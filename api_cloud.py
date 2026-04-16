"""Render 雲端版 LINE Bot API — 不依賴任何 ML 套件，可部署到免費雲端。

啟動方式：
    uvicorn api_cloud:app --host 0.0.0.0 --port $PORT
"""

import os
import json
import hashlib
import hmac
import base64
import tempfile
from urllib.parse import quote as _url_quote

import httpx
from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pyvis.network import Network

# ---------------------------------------------------------------------------
# Config (all from environment variables)
# ---------------------------------------------------------------------------

RENDER_EXTERNAL_URL = os.environ.get("RENDER_EXTERNAL_URL", "")


def _line_default_scheme() -> str:
    raw = os.environ.get("LINE_DEFAULT_SCHEME")
    if raw is None or str(raw).strip() == "":
        return "甲班"
    return str(raw).strip()


def _line_teacher_password() -> str:
    raw = os.environ.get("LINE_TEACHER_PASSWORD")
    if raw is None or str(raw).strip() == "":
        return "teacher123"
    return str(raw).strip()


def _line_channel_secret() -> str:
    return str(os.environ.get("LINE_CHANNEL_SECRET", "") or "").strip()


def _line_channel_access_token() -> str:
    return str(os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "") or "").strip()


def _sync_secret_expected() -> str:
    """每次請求讀取，避免程序啟動時尚未注入 SYNC_SECRET 而永遠用預設值。"""
    raw = os.environ.get("SYNC_SECRET")
    if raw is None or str(raw).strip() == "":
        return "changeme"
    return str(raw).strip()

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cloud_data")

# Ego graph defaults (mirrors config.py)
EGO_MIN_COOCCURRENCE = 3
EGO_MIN_EDGE_WEIGHT = 0.0
EGO_MAX_NEIGHBORS = None

# Relationship graph visual defaults
GRAPH_HEIGHT = "780px"
NODE_SIZE = 30
PHYSICS_SPRING_LENGTH = 280
PHYSICS_REPULSION = -120
PHYSICS_CENTRAL_GRAVITY = 0.004
PHYSICS_STABILIZATION_ITERS = 320

PERSONALITY_COLORS = {
    "社交型": "#4CAF50",
    "親密型": "#E91E63",
    "活躍型": "#FF9800",
    "內向型": "#2196F3",
    "觀察中": "#9E9E9E",
}

# ---------------------------------------------------------------------------
# In-memory session store
# ---------------------------------------------------------------------------

_user_sessions: dict[str, dict] = {}

# ---------------------------------------------------------------------------
# Helper: paths
# ---------------------------------------------------------------------------


def _scheme_dir(scheme: str) -> str:
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in scheme)
    d = os.path.join(DATA_DIR, "schemes", safe)
    os.makedirs(d, exist_ok=True)
    return d


def _graph_dir(scheme: str) -> str:
    d = os.path.join(_scheme_dir(scheme), "graph")
    os.makedirs(d, exist_ok=True)
    return d


def _load_kg(scheme: str) -> dict | None:
    p = os.path.join(_graph_dir(scheme), "knowledge_graph.json")
    if not os.path.isfile(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_registered_names(scheme: str) -> list[str]:
    p = os.path.join(_scheme_dir(scheme), "registered_names.json")
    if not os.path.isfile(p):
        return []
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _list_schemes() -> list[str]:
    d = os.path.join(DATA_DIR, "schemes")
    if not os.path.isdir(d):
        return []
    return sorted(x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x)))


# ---------------------------------------------------------------------------
# Ego knowledge graph (ported from knowledge_graph.py — pure dict ops)
# ---------------------------------------------------------------------------


def _filter_timeline(timeline: dict | None, node_ids: set) -> dict | None:
    if not timeline or not isinstance(timeline, dict) or not node_ids:
        return timeline
    ps = timeline.get("person_series") or {}
    filt = {k: v for k, v in ps.items() if k in node_ids}
    if not filt:
        return None
    out = dict(timeline)
    out["person_series"] = filt
    return out


def _ego_knowledge_graph(
    kg_data: dict,
    focal_id: str,
    min_cooccurrence: int = EGO_MIN_COOCCURRENCE,
    min_edge_weight: float = EGO_MIN_EDGE_WEIGHT,
    max_neighbors: int | None = EGO_MAX_NEIGHBORS,
) -> dict:
    node_map = {n["id"]: n for n in kg_data.get("nodes", [])}
    if focal_id not in node_map:
        return {
            "nodes": [{"id": focal_id, "label": focal_id, "personality": ["觀察中"]}],
            "edges": [],
            "personality": kg_data.get("personality", {}),
            "focal_id": focal_id,
            "interaction_timeline": _filter_timeline(
                kg_data.get("interaction_timeline"), {focal_id}
            ),
        }

    candidates = []
    for e in kg_data.get("edges", []):
        u, v = e.get("source"), e.get("target")
        if u != focal_id and v != focal_id:
            continue
        co = e.get("cooccurrence", 0)
        w = float(e.get("weight", 0))
        if co < min_cooccurrence or w < min_edge_weight:
            continue
        peer = v if u == focal_id else u
        candidates.append((w, co, peer, e))

    candidates.sort(key=lambda x: (-x[0], -x[1], x[2]))

    chosen = []
    seen = set()
    for w, co, peer, e in candidates:
        if peer in seen:
            continue
        if max_neighbors is not None and len(chosen) >= max_neighbors:
            break
        seen.add(peer)
        chosen.append((peer, e))

    peer_ids = {p for p, _ in chosen}
    nodes_out = [node_map[focal_id]]
    for pid in sorted(peer_ids):
        if pid in node_map:
            nodes_out.append(node_map[pid])

    edges_out = [dict(e) for _, e in chosen]
    node_ids = {n["id"] for n in nodes_out}
    return {
        "nodes": nodes_out,
        "edges": edges_out,
        "personality": kg_data.get("personality", {}),
        "focal_id": focal_id,
        "interaction_timeline": _filter_timeline(
            kg_data.get("interaction_timeline"), node_ids
        ),
    }


# ---------------------------------------------------------------------------
# Relationship graph drawing (ported from relationship_graph.py)
# ---------------------------------------------------------------------------


def _node_color(personality_list):
    if personality_list and personality_list[0] in PERSONALITY_COLORS:
        return PERSONALITY_COLORS[personality_list[0]]
    return PERSONALITY_COLORS["觀察中"]


def _inject_timeline_chart(html_path: str, kg_data: dict) -> None:
    if not os.path.isfile(html_path):
        return
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()
    if "</body>" not in html:
        return
    timeline = kg_data.get("interaction_timeline")
    payload = timeline if isinstance(timeline, dict) else {}
    json_str = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
    snippet = f"""
        <div id="kg-hover-chart-panel" style="display:none;position:fixed;z-index:9999;right:12px;bottom:12px;width:380px;max-width:96vw;background:#fff;border:1px solid #bdbdbd;border-radius:10px;box-shadow:0 4px 18px rgba(0,0,0,0.12);padding:12px 14px;font-family:Microsoft JhengHei,SimHei,sans-serif;">
          <div id="kg-hover-chart-title" style="font-weight:600;margin-bottom:8px;color:#333;font-size:15px;"></div>
          <div style="position:relative;height:220px;">
            <canvas id="kg-hover-chart-canvas"></canvas>
          </div>
          <div id="kg-hover-chart-hint" style="margin-top:8px;font-size:12px;color:#666;line-height:1.45;"></div>
        </div>
        <script type="application/json" id="kg-interaction-timeline-data">{json_str}</script>
        <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js" crossorigin="anonymous"></script>
        <script type="text/javascript">
(function () {{
  function kgParseTimeline() {{
    try {{
      var el = document.getElementById('kg-interaction-timeline-data');
      if (!el) return null;
      return JSON.parse(el.textContent || '{{}}');
    }} catch (e) {{ return null; }}
  }}
  var __kgTimeline = kgParseTimeline();
  var __kgChart = null;
  function kgFormatTime(sec) {{
    var s = Math.floor(Number(sec) || 0);
    var m = Math.floor(s / 60);
    var r = s % 60;
    return m + ':' + (r < 10 ? '0' : '') + r;
  }}
  function kgShowTimelineChart(nodeId) {{
    var panel = document.getElementById('kg-hover-chart-panel');
    var titleEl = document.getElementById('kg-hover-chart-title');
    var hint = document.getElementById('kg-hover-chart-hint');
    var canvas = document.getElementById('kg-hover-chart-canvas');
    if (!panel || !canvas || !titleEl || !hint) return;
    if (typeof Chart === 'undefined') {{
      titleEl.textContent = nodeId + ' · 互動時間軸';
      hint.textContent = '無法載入圖表程式庫。';
      panel.style.display = 'block';
      return;
    }}
    var TL = __kgTimeline;
    if (!TL || !TL.person_series) {{
      titleEl.textContent = nodeId + ' · 互動時間軸';
      hint.textContent = '無時間軸互動資料。';
      panel.style.display = 'block';
      if (__kgChart) {{ __kgChart.destroy(); __kgChart = null; }}
      return;
    }}
    var series = TL.person_series[nodeId];
    if (!series || !series.length) {{
      titleEl.textContent = nodeId + ' · 互動時間軸';
      hint.textContent = '此節點尚無逐時段互動紀錄。';
      panel.style.display = 'block';
      if (__kgChart) {{ __kgChart.destroy(); __kgChart = null; }}
      return;
    }}
    titleEl.textContent = nodeId + ' · 互動時間軸';
    var binHint = TL.bin_sec ? ('時間桶：約 ' + TL.bin_sec + ' 秒／格。') : '';
    hint.textContent = binHint + '游標移到折線數據點上可查看該時段資訊。';
    var labels = series.map(function (p) {{ return kgFormatTime(p.t); }});
    var values = series.map(function (p) {{ return p.unique_peer_count; }});
    var detail = series.map(function (p) {{ return p.with || {{}}; }});
    var ctx = canvas.getContext('2d');
    if (__kgChart) {{ __kgChart.destroy(); __kgChart = null; }}
    __kgChart = new Chart(ctx, {{
      type: 'line',
      data: {{
        labels: labels,
        datasets: [{{
          label: (TL.y_label || '同框同伴人數'),
          data: values,
          borderColor: '#1565C0',
          backgroundColor: 'rgba(21,101,192,0.12)',
          tension: 0.2, fill: true, pointRadius: 3, pointHoverRadius: 5
        }}]
      }},
      options: {{
        responsive: true, maintainAspectRatio: false, animation: false,
        plugins: {{
          legend: {{ display: true }},
          tooltip: {{
            callbacks: {{
              title: function (items) {{
                if (!items.length) return '';
                return '時段終點約 ' + kgFormatTime(series[items[0].dataIndex].t);
              }},
              label: function (ctx) {{ return (TL.y_label || '同框同伴人數') + '：' + ctx.parsed.y; }},
              afterBody: function (items) {{
                if (!items.length) return [];
                var w = detail[items[0].dataIndex];
                if (!w) return [];
                var lines = ['— 同框對象（幀數）—'];
                Object.keys(w).sort(function (a, b) {{ return w[b] - w[a]; }}).forEach(function (k) {{
                  lines.push(k + '：' + w[k]);
                }});
                return lines;
              }}
            }}
          }}
        }},
        scales: {{
          x: {{ title: {{ display: true, text: '時間' }} }},
          y: {{ title: {{ display: true, text: (TL.y_label || '人數') }}, beginAtZero: true, ticks: {{ precision: 0 }} }}
        }}
      }}
    }});
    panel.style.display = 'block';
  }}
  function kgHideTimelineChart() {{
    var panel = document.getElementById('kg-hover-chart-panel');
    if (panel) panel.style.display = 'none';
    if (__kgChart) {{ __kgChart.destroy(); __kgChart = null; }}
  }}
  function kgAttachNetworkHover() {{
    if (typeof network === 'undefined' || !network) {{ setTimeout(kgAttachNetworkHover, 50); return; }}
    network.on('hoverNode', function (p) {{ kgShowTimelineChart(p.node); }});
    network.on('blurNode', function () {{ kgHideTimelineChart(); }});
  }}
  if (document.readyState === 'loading')
    document.addEventListener('DOMContentLoaded', kgAttachNetworkHover);
  else
    kgAttachNetworkHover();
}})();
        </script>
"""
    html = html.replace("</body>", snippet + "\n</body>", 1)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)


def _draw_relationship_graph(
    kg_data: dict,
    output_html: str,
    title: str = "幼兒關係圖",
    focal_node_id: str = None,
) -> str:
    focal = focal_node_id or kg_data.get("focal_id")
    net = Network(
        height=GRAPH_HEIGHT, width="100%",
        bgcolor="#fafafa", font_color="#333", directed=False,
    )
    sl = PHYSICS_SPRING_LENGTH
    rep = PHYSICS_REPULSION
    cg = PHYSICS_CENTRAL_GRAVITY
    stab = PHYSICS_STABILIZATION_ITERS
    net.set_options(
        f"""var options = {{
  "interaction": {{ "hover": true, "tooltipDelay": 120, "zoomView": true, "dragView": true }},
  "nodes": {{
    "font": {{ "size": 15, "face": "Microsoft JhengHei, SimHei, sans-serif" }},
    "shape": "dot", "shadow": true, "margin": 14,
    "scaling": {{ "min": 12, "max": 48 }}
  }},
  "edges": {{
    "smooth": {{ "type": "dynamic", "roundness": 0.35 }},
    "color": {{ "inherit": false, "color": "#9E9E9E", "highlight": "#616161" }}
  }},
  "physics": {{
    "enabled": true, "solver": "forceAtlas2Based",
    "forceAtlas2Based": {{
      "gravitationalConstant": {rep},
      "centralGravity": {cg},
      "springLength": {sl},
      "springConstant": 0.045,
      "damping": 0.55,
      "avoidOverlap": 1
    }},
    "minVelocity": 0.5, "maxVelocity": 22,
    "stabilization": {{ "enabled": true, "iterations": {stab}, "updateInterval": 40, "fit": true }}
  }}
}}"""
    )

    node_size = NODE_SIZE
    for n in kg_data.get("nodes", []):
        pid = n["id"]
        label = n["label"]
        pers = n.get("personality", [])
        pers_str = "、".join(pers) if pers else "觀察中"
        is_focal = focal and pid == focal
        title_str = f"{'【中心】' if is_focal else ''}{label}\n個性：{pers_str}"
        color = _node_color(pers)
        sz = int(node_size * 1.45) if is_focal else node_size
        kwargs = dict(label=label, title=title_str, size=sz, color=color)
        if is_focal:
            kwargs["borderWidth"] = 3
            kwargs["color"] = {"background": color, "border": "#1565C0"}
        net.add_node(pid, **kwargs)

    for e in kg_data.get("edges", []):
        w = e.get("weight", 1)
        co = e.get("cooccurrence", 0)
        width = max(1.0, min(1.4 + w / 12.0, 7.0))
        net.add_edge(
            e["source"], e["target"], value=w,
            title=f"朋友關係 · 同框 {co} 次", width=width,
        )

    os.makedirs(os.path.dirname(output_html) or ".", exist_ok=True)
    net.save_graph(output_html)
    _inject_timeline_chart(output_html, kg_data)
    return output_html


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="幼兒知識圖譜 LINE Bot（雲端版）",
    description="輕量版 API，部署於 Render，提供 LINE Bot webhook 與圖譜瀏覽。",
    version="2.0.0",
)


def _get_base_url(request: Request) -> str:
    if RENDER_EXTERNAL_URL:
        return RENDER_EXTERNAL_URL.rstrip("/")
    return str(request.base_url).rstrip("/")


# ---------------------------------------------------------------------------
# Data sync endpoint
# ---------------------------------------------------------------------------

@app.post("/api/sync", summary="從本機同步資料到雲端")
async def sync_data(request: Request):
    body = await request.json()
    secret = str(body.get("secret", "")).strip()
    expected = _sync_secret_expected()
    if not hmac.compare_digest(secret, expected):
        raise HTTPException(403, "Invalid sync secret")

    scheme = body.get("scheme", "").strip()
    if not scheme:
        raise HTTPException(400, "scheme is required")

    kg_data = body.get("knowledge_graph")
    names = body.get("registered_names")

    if kg_data is None and names is None:
        raise HTTPException(400, "No data provided")

    sd = _scheme_dir(scheme)
    gd = _graph_dir(scheme)

    saved = []
    if kg_data is not None:
        p = os.path.join(gd, "knowledge_graph.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(kg_data, f, ensure_ascii=False, indent=1)
        saved.append("knowledge_graph.json")

        _draw_relationship_graph(
            kg_data, os.path.join(gd, "relationship_graph.html"),
            title=f"幼兒關係圖（{scheme}）",
        )
        saved.append("relationship_graph.html")

        _draw_relationship_graph(
            kg_data, os.path.join(gd, "knowledge_graph.html"),
            title=f"幼兒知識圖譜（{scheme}）",
        )
        saved.append("knowledge_graph.html")

    if names is not None:
        p = os.path.join(sd, "registered_names.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(names, f, ensure_ascii=False)
        saved.append("registered_names.json")

    return {"message": "同步完成", "scheme": scheme, "saved": saved}


# ---------------------------------------------------------------------------
# Graph viewing endpoints
# ---------------------------------------------------------------------------

@app.get("/api/schemes", summary="列出所有方案")
def list_schemes():
    return {"schemes": _list_schemes()}


@app.get("/api/graph/data", summary="取得知識圖譜 JSON")
def get_graph_data(scheme: str = Query("", description="方案名稱")):
    scheme = scheme or _line_default_scheme()
    kg = _load_kg(scheme)
    if kg is None:
        raise HTTPException(404, f"方案「{scheme}」尚未同步圖譜資料。")
    return JSONResponse(content=kg)


@app.get("/api/graph/relationship", response_class=HTMLResponse,
         summary="取得關係圖 HTML")
def get_relationship_html(scheme: str = Query("", description="方案名稱")):
    scheme = scheme or _line_default_scheme()
    p = os.path.join(_graph_dir(scheme), "relationship_graph.html")
    if not os.path.isfile(p):
        raise HTTPException(404, "尚未同步關係圖。")
    with open(p, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/api/graph/knowledge", response_class=HTMLResponse,
         summary="取得知識圖譜 HTML")
def get_knowledge_html(scheme: str = Query("", description="方案名稱")):
    scheme = scheme or _line_default_scheme()
    p = os.path.join(_graph_dir(scheme), "knowledge_graph.html")
    if not os.path.isfile(p):
        raise HTTPException(404, "尚未同步知識圖譜。")
    with open(p, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/api/graph/ego", response_class=HTMLResponse,
         summary="取得指定幼兒的 ego 圖譜 HTML")
def get_ego_html(
    scheme: str = Query("", description="方案名稱"),
    name: str = Query(..., description="幼兒名字"),
):
    scheme = scheme or _line_default_scheme()
    kg = _load_kg(scheme)
    if kg is None:
        raise HTTPException(404, "尚未同步圖譜資料。")
    ego_data = _ego_knowledge_graph(kg, name)
    ego_dir = os.path.join(_graph_dir(scheme), "ego")
    os.makedirs(ego_dir, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)
    out_path = os.path.join(ego_dir, f"ego_{safe}.html")
    _draw_relationship_graph(
        ego_data, out_path,
        title=f"幼兒關係圖 · {name}", focal_node_id=name,
    )
    with open(out_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/api/graph/summary", summary="取得圖譜純文字摘要")
def get_graph_summary(scheme: str = Query("", description="方案名稱")):
    scheme = scheme or _line_default_scheme()
    kg = _load_kg(scheme)
    if kg is None:
        raise HTTPException(404, "尚未同步圖譜資料。")
    return {"scheme": scheme, "summary": _build_full_summary_text(scheme, kg)}


# ---------------------------------------------------------------------------
# LINE Bot helpers
# ---------------------------------------------------------------------------

def _verify_line_signature(body: bytes, signature: str) -> bool:
    secret = _line_channel_secret()
    if not secret or not signature:
        return False
    mac = hmac.new(secret.encode("utf-8"), body, hashlib.sha256)
    return hmac.compare_digest(
        base64.b64encode(mac.digest()).decode("utf-8"), signature
    )


async def _line_reply(reply_token: str, messages: list[dict]):
    token = _line_channel_access_token()
    if not token:
        return
    async with httpx.AsyncClient() as client:
        await client.post(
            "https://api.line.me/v2/bot/message/reply",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
            },
            json={"replyToken": reply_token, "messages": messages},
        )


def _get_registered_names_cloud(scheme: str) -> set[str]:
    return set(_load_registered_names(scheme))


def _build_full_summary_text(scheme: str, kg: dict | None = None) -> str:
    if kg is None:
        kg = _load_kg(scheme)
    if kg is None:
        return f"方案「{scheme}」尚未同步圖譜資料。"
    nodes = kg.get("nodes", [])
    edges = kg.get("edges", [])
    lines = [f"📊 知識圖譜摘要（{scheme}）", ""]
    lines.append(f"👶 共 {len(nodes)} 位幼兒：")
    for n in nodes:
        pers = "、".join(n.get("personality", ["觀察中"]))
        lines.append(f"  · {n['label']}（{pers}）")
    lines.append("")
    if edges:
        es = sorted(edges, key=lambda e: e.get("cooccurrence", 0), reverse=True)
        lines.append(f"🤝 共 {len(edges)} 組朋友關係：")
        for e in es[:10]:
            lines.append(f"  · {e['source']} ↔ {e['target']}：同框 {e.get('cooccurrence', 0)} 次")
        if len(edges) > 10:
            lines.append(f"  ...（還有 {len(edges) - 10} 組）")
    else:
        lines.append("尚無朋友關係。")
    return "\n".join(lines)


def _build_ego_summary_text(scheme: str, child_name: str) -> str:
    kg = _load_kg(scheme)
    if kg is None:
        return f"方案「{scheme}」尚未同步圖譜資料。"
    ego = _ego_knowledge_graph(kg, child_name)
    nodes = ego.get("nodes", [])
    edges = ego.get("edges", [])
    focal = next((n for n in nodes if n["id"] == child_name), None)
    pers = "、".join(focal["personality"]) if focal else "觀察中"
    lines = [f"📊 {child_name} 的個人圖譜（{scheme}）", ""]
    lines.append(f"👤 個性：{pers}")
    lines.append("")
    if edges:
        es = sorted(edges, key=lambda e: e.get("cooccurrence", 0), reverse=True)
        lines.append(f"🤝 共 {len(edges)} 位朋友：")
        for e in es:
            peer = e["target"] if e["source"] == child_name else e["source"]
            lines.append(f"  · {peer}：同框 {e.get('cooccurrence', 0)} 次")
    else:
        lines.append("尚無朋友關係紀錄。")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LINE Bot Webhook
# ---------------------------------------------------------------------------

AUTH_PROMPT = (
    "🔐 請先驗證身份：\n\n"
    "老師請輸入密碼\n"
    "學生/家長請輸入幼兒名字（如：小孩1）"
)

TEACHER_HELP = (
    "📋 老師可用指令：\n"
    "  「圖譜」→ 查看完整知識圖譜\n"
    "  「名單」→ 查看已註冊幼兒\n"
    "  「方案」→ 列出所有方案\n"
    "  「登出」→ 切換身份\n"
    "  「說明」→ 顯示此說明"
)

STUDENT_HELP_TPL = (
    "📋 {name} 可用指令：\n"
    "  「圖譜」→ 查看 {name} 的個人圖譜\n"
    "  「名單」→ 查看已註冊幼兒\n"
    "  「登出」→ 切換身份\n"
    "  「說明」→ 顯示此說明"
)


@app.post("/webhook", summary="LINE Bot Webhook")
async def line_webhook(request: Request):
    body = await request.body()
    signature = request.headers.get("x-line-signature", "")
    if not _verify_line_signature(body, signature):
        raise HTTPException(403, "Invalid signature")

    payload = json.loads(body)
    events = payload.get("events", [])
    base_url = _get_base_url(request)

    for event in events:
        if event.get("type") != "message" or event["message"].get("type") != "text":
            continue

        text = event["message"]["text"].strip()
        reply_token = event["replyToken"]
        user_id = event["source"].get("userId", "")
        scheme = _line_default_scheme()
        session = _user_sessions.get(user_id)

        # --- 登出 ---
        if text in ("登出", "logout", "切換身份"):
            _user_sessions.pop(user_id, None)
            await _line_reply(reply_token, [
                {"type": "text", "text": "已登出。\n\n" + AUTH_PROMPT},
            ])
            continue

        # --- 未驗證：嘗試驗證 ---
        if session is None:
            if text == _line_teacher_password():
                _user_sessions[user_id] = {"role": "teacher"}
                await _line_reply(reply_token, [
                    {"type": "text", "text": "✅ 老師身份驗證成功！\n\n" + TEACHER_HELP},
                ])
                continue

            registered = _get_registered_names_cloud(scheme)
            if text in registered:
                _user_sessions[user_id] = {"role": "student", "child_name": text}
                help_text = STUDENT_HELP_TPL.format(name=text)
                await _line_reply(reply_token, [
                    {"type": "text", "text": f"✅ 歡迎，{text}！\n\n" + help_text},
                ])
                continue

            await _line_reply(reply_token, [
                {"type": "text", "text": AUTH_PROMPT},
            ])
            continue

        # --- 已驗證：處理指令 ---
        role = session["role"]
        child_name = session.get("child_name", "")

        if text in ("圖譜", "查圖譜", "知識圖譜", "關係圖"):
            if role == "teacher":
                summary = _build_full_summary_text(scheme)
                messages = [{"type": "text", "text": summary}]
                link_kg = f"{base_url}/api/graph/knowledge?scheme={_url_quote(scheme)}"
                link_rel = f"{base_url}/api/graph/relationship?scheme={_url_quote(scheme)}"
                messages.append({
                    "type": "text",
                    "text": (
                        f"📈 點擊查看互動式圖譜：\n"
                        f"知識圖譜：{link_kg}\n"
                        f"關係圖：{link_rel}"
                    ),
                })
                await _line_reply(reply_token, messages)
            else:
                summary = _build_ego_summary_text(scheme, child_name)
                messages = [{"type": "text", "text": summary}]
                link_ego = f"{base_url}/api/graph/ego?scheme={_url_quote(scheme)}&name={_url_quote(child_name)}"
                messages.append({
                    "type": "text",
                    "text": f"📈 點擊查看 {child_name} 的互動式圖譜：\n{link_ego}",
                })
                await _line_reply(reply_token, messages)

        elif text in ("名單", "查名單", "註冊"):
            names = _load_registered_names(scheme)
            if not names:
                msg = f"方案「{scheme}」尚無註冊資料。"
            else:
                msg = f"📋 {scheme} 已註冊 {len(names)} 人：\n"
                msg += "\n".join(f"  · {n}" for n in names)
            await _line_reply(reply_token, [{"type": "text", "text": msg}])

        elif text in ("方案", "查方案", "列表"):
            if role != "teacher":
                await _line_reply(reply_token, [
                    {"type": "text", "text": "此指令僅限老師使用。"},
                ])
            else:
                schemes = _list_schemes()
                if schemes:
                    msg = "📂 已有方案：\n" + "\n".join(f"  · {s}" for s in schemes)
                else:
                    msg = "尚無任何方案。"
                await _line_reply(reply_token, [{"type": "text", "text": msg}])

        elif text in ("說明", "help", "幫助", "指令"):
            if role == "teacher":
                await _line_reply(reply_token, [{"type": "text", "text": TEACHER_HELP}])
            else:
                await _line_reply(reply_token, [
                    {"type": "text", "text": STUDENT_HELP_TPL.format(name=child_name)},
                ])

        elif text in ("我是誰", "身份", "whoami"):
            if role == "teacher":
                msg = "👩‍🏫 目前身份：老師"
            else:
                msg = f"👤 目前身份：{child_name}"
            await _line_reply(reply_token, [{"type": "text", "text": msg}])

        else:
            help_text = TEACHER_HELP if role == "teacher" else STUDENT_HELP_TPL.format(name=child_name)
            await _line_reply(reply_token, [
                {"type": "text", "text": f"收到：「{text}」\n\n" + help_text},
            ])

    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/", summary="健康檢查")
def health():
    return {"status": "ok", "service": "child_tracker_kg cloud API"}
