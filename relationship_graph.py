# 關係圖視覺化：誰跟誰是朋友、關係強度、個性標籤
import json
import os
from pyvis.network import Network

from config import (
    graph_dir,
    ensure_dirs,
    RELATIONSHIP_GRAPH_HEIGHT,
    RELATIONSHIP_GRAPH_NODE_SIZE,
    GRAPH_PHYSICS_SPRING_LENGTH,
    GRAPH_PHYSICS_REPULSION,
    GRAPH_PHYSICS_CENTRAL_GRAVITY,
    GRAPH_PHYSICS_STABILIZATION_ITERS,
)
from knowledge_graph import build_knowledge_graph, ego_knowledge_graph
from config import EGO_GRAPH_SUBDIR

# 個性標籤對應顏色（方便一眼看出小孩個性）
PERSONALITY_COLORS = {
    "社交型": "#4CAF50",
    "親密型": "#E91E63",
    "活躍型": "#FF9800",
    "內向型": "#2196F3",
    "觀察中": "#9E9E9E",
}


def _node_color(personality_list):
    """依第一個個性標籤決定節點顏色。"""
    if personality_list and personality_list[0] in PERSONALITY_COLORS:
        return PERSONALITY_COLORS[personality_list[0]]
    return PERSONALITY_COLORS["觀察中"]


def _safe_filename_segment(name: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in str(name))


def _inject_interaction_timeline_chart(output_html: str, kg_data: dict) -> None:
    """在 pyvis 輸出之後嵌入互動時間軸 JSON、Chart.js 與 hover 折線圖。"""
    if not os.path.isfile(output_html):
        return
    try:
        with open(output_html, "r", encoding="utf-8") as f:
            html = f.read()
    except OSError:
        return
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
      hint.textContent = '無法載入圖表程式庫，請檢查網路連線。';
      panel.style.display = 'block';
      return;
    }}

    var TL = __kgTimeline;
    if (!TL || !TL.person_series) {{
      titleEl.textContent = nodeId + ' · 互動時間軸';
      hint.textContent = '無時間軸互動資料。請使用 YOLO 流程並開啟「回溯互動」處理影片後，再執行建圖。';
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
    hint.textContent = binHint + '游標移到折線數據點上可查看該時段與哪些同伴同框（同框幀數）。';

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
          tension: 0.2,
          fill: true,
          pointRadius: 3,
          pointHoverRadius: 5
        }}]
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        plugins: {{
          legend: {{ display: true }},
          tooltip: {{
            callbacks: {{
              title: function (items) {{
                if (!items.length) return '';
                var i = items[0].dataIndex;
                return '時段終點約 ' + kgFormatTime(series[i].t);
              }},
              label: function (ctx) {{
                return (TL.y_label || '同框同伴人數') + '：' + ctx.parsed.y;
              }},
              afterBody: function (items) {{
                if (!items.length) return [];
                var i = items[0].dataIndex;
                var w = detail[i];
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
          y: {{
            title: {{ display: true, text: (TL.y_label || '人數') }},
            beginAtZero: true,
            ticks: {{ precision: 0 }}
          }}
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
    if (typeof network === 'undefined' || !network) {{
      setTimeout(kgAttachNetworkHover, 50);
      return;
    }}
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
    try:
        with open(output_html, "w", encoding="utf-8") as f:
            f.write(html)
    except OSError:
        pass


def draw_relationship_graph(
    kg_data: dict = None,
    output_html: str = None,
    title: str = "幼兒關係圖",
    focal_node_id: str = None,
) -> str:
    """
    產生關係圖 HTML（pyvis 互動圖）。
    若未傳 kg_data 會先呼叫 build_knowledge_graph 取得資料。
    """
    ensure_dirs()
    if kg_data is None:
        kg_data = build_knowledge_graph()
    if output_html is None:
        output_html = os.path.join(graph_dir(), "relationship_graph.html")

    focal = focal_node_id or kg_data.get("focal_id")

    net = Network(
        height=RELATIONSHIP_GRAPH_HEIGHT,
        width="100%",
        bgcolor="#fafafa",
        font_color="#333",
        directed=False,
    )
    # 節點散開：forceAtlas2Based + 長彈簧 + 斥力；避免全擠成一團
    try:
        sl = int(GRAPH_PHYSICS_SPRING_LENGTH)
        rep = float(GRAPH_PHYSICS_REPULSION)
        cg = float(GRAPH_PHYSICS_CENTRAL_GRAVITY)
        stab = int(GRAPH_PHYSICS_STABILIZATION_ITERS)
        net.set_options(
            f"""var options = {{
  "interaction": {{ "hover": true, "tooltipDelay": 120, "zoomView": true, "dragView": true }},
  "nodes": {{
    "font": {{ "size": 15, "face": "Microsoft JhengHei, SimHei, sans-serif" }},
    "shape": "dot",
    "shadow": true,
    "margin": 14,
    "scaling": {{ "min": 12, "max": 48 }}
  }},
  "edges": {{
    "smooth": {{ "type": "dynamic", "roundness": 0.35 }},
    "color": {{ "inherit": false, "color": "#9E9E9E", "highlight": "#616161" }}
  }},
  "physics": {{
    "enabled": true,
    "solver": "forceAtlas2Based",
    "forceAtlas2Based": {{
      "gravitationalConstant": {rep},
      "centralGravity": {cg},
      "springLength": {sl},
      "springConstant": 0.045,
      "damping": 0.55,
      "avoidOverlap": 1
    }},
    "minVelocity": 0.5,
    "maxVelocity": 22,
    "stabilization": {{
      "enabled": true,
      "iterations": {stab},
      "updateInterval": 40,
      "fit": true
    }}
  }}
}}"""
        )
    except Exception:
        pass

    # 節點：畫面上只顯示名字（減少字疊在一起）；個性放在 hover
    node_size = int(RELATIONSHIP_GRAPH_NODE_SIZE)
    for n in kg_data.get("nodes", []):
        pid = n["id"]
        label = n["label"]
        pers = n.get("personality", [])
        pers_str = "、".join(pers) if pers else "觀察中"
        is_focal = focal and pid == focal
        title_str = f"{label}\n個性：{pers_str}\n（可拖曳節點、滾輪縮放）"
        if is_focal:
            title_str = f"【中心】{label}\n個性：{pers_str}\n（可拖曳節點、滾輪縮放）"
        color = _node_color(pers)
        sz = int(node_size * 1.45) if is_focal else node_size
        kwargs = dict(label=label, title=title_str, size=sz, color=color)
        if is_focal:
            kwargs["borderWidth"] = 3
            kwargs["color"] = {"background": color, "border": "#1565C0"}
        net.add_node(pid, **kwargs)

    # 邊：權重愈大線愈粗（上限降低，避免線條糊成一塊）
    for e in kg_data.get("edges", []):
        w = e.get("weight", 1)
        co = e.get("cooccurrence", 0)
        width = max(1.0, min(1.4 + w / 12.0, 7.0))
        net.add_edge(
            e["source"],
            e["target"],
            value=w,
            title=f"朋友關係 · 同框 {co} 次（線愈粗關係愈好）",
            width=width,
        )

    os.makedirs(os.path.dirname(output_html), exist_ok=True)
    net.save_graph(output_html)
    _inject_interaction_timeline_chart(output_html, kg_data)
    return output_html


def draw_all_ego_graphs(kg_data: dict, output_dir: str = None) -> list:
    """
    為 kg_data 中每個節點各產一張「以該人為中心」的子圖 HTML。
    同伴是否出現依 config.EGO_GRAPH_* 門檻。
    """
    output_dir = output_dir or graph_dir()
    ego_dir = os.path.join(output_dir, EGO_GRAPH_SUBDIR)
    os.makedirs(ego_dir, exist_ok=True)
    paths = []
    for n in kg_data.get("nodes", []):
        fid = n["id"]
        ego = ego_knowledge_graph(kg_data, fid)
        safe = _safe_filename_segment(fid)
        out = os.path.join(ego_dir, f"ego_{safe}.html")
        draw_relationship_graph(
            kg_data=ego,
            output_html=out,
            title=f"幼兒關係圖 · {fid}",
            focal_node_id=fid,
        )
        paths.append(out)
    return paths


def run_build_and_draw(output_dir: str = None, build_ego: bool = False) -> tuple:
    """建知識圖譜並畫關係圖，回傳 (kg_path, html_path)。build_ego=True 時另寫入 data/graph/ego/。"""
    output_dir = output_dir or graph_dir()
    ensure_dirs()
    kg_path = os.path.join(output_dir, "knowledge_graph.json")
    html_path = os.path.join(output_dir, "relationship_graph.html")
    kg_path_kg = os.path.join(output_dir, "knowledge_graph.html")
    kg_data = build_knowledge_graph(output_path=kg_path)
    draw_relationship_graph(kg_data=kg_data, output_html=html_path, title="幼兒關係圖")
    draw_relationship_graph(kg_data=kg_data, output_html=kg_path_kg, title="幼兒知識圖譜")
    ego_paths = []
    if build_ego:
        ego_paths = draw_all_ego_graphs(kg_data, output_dir=output_dir)
    return kg_path, html_path, ego_paths
