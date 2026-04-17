"""FastAPI REST API — 提供知識圖譜、關係圖等端點，供 n8n / LINE Bot 串接。

啟動方式：
    python api.py                    # http://localhost:8000
    python api.py --port 8080        # 自訂 port
"""

import os, sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from runtime_hardening import apply as _runtime_hardening_apply
_runtime_hardening_apply()

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
import uvicorn

from config import (
    ensure_dirs,
    SCHEMES_PARENT,
    PROJECT_ROOT,
    set_active_scheme,
    active_scheme,
    graph_dir,
    scheme_root,
    interactions_file,
    registry_file,
)
from face_registry import get_registry_encodings, get_unique_registered_names

app = FastAPI(
    title="幼兒辨識與知識圖譜 API",
    description="提供知識圖譜、關係圖、註冊名單等查詢端點",
    version="1.0.0",
)


def _activate(scheme: str | None):
    s = (scheme or "").strip()
    set_active_scheme(s if s else None)
    ensure_dirs()


def _list_schemes() -> list[str]:
    if not os.path.isdir(SCHEMES_PARENT):
        return []
    return sorted(
        d for d in os.listdir(SCHEMES_PARENT)
        if os.path.isdir(os.path.join(SCHEMES_PARENT, d))
    )


# ---------------------------------------------------------------------------
# 端點
# ---------------------------------------------------------------------------

@app.get("/api/schemes", summary="列出所有方案")
def list_schemes():
    return {"schemes": _list_schemes()}


@app.get("/api/registry", summary="查看某方案的註冊名單")
def get_registry(scheme: str = Query("", description="方案名稱，空字串為預設")):
    _activate(scheme)
    names, _ = get_registry_encodings()
    if not names:
        return {"scheme": active_scheme() or "預設", "count": 0, "names": []}
    from collections import Counter
    c = Counter(names)
    return {
        "scheme": active_scheme() or "預設",
        "count": len(c),
        "total_templates": len(names),
        "names": [{"name": n, "templates": cnt} for n, cnt in c.most_common()],
    }


@app.post("/api/build-graph", summary="建構圖譜（需先有 interactions.json）")
def build_graph(scheme: str = Query("", description="方案名稱"), ego: bool = Query(False)):
    _activate(scheme)
    int_file = interactions_file()
    if not os.path.isfile(int_file):
        raise HTTPException(404, f"找不到互動記錄：{int_file}。請先執行影片處理。")

    from relationship_graph import run_build_and_draw
    kg_path, html_path, ego_paths = run_build_and_draw(
        output_dir=graph_dir(), build_ego=ego
    )
    return {
        "message": "圖譜建構完成",
        "knowledge_graph_json": kg_path,
        "relationship_graph_html": html_path,
        "ego_graphs": len(ego_paths),
    }


@app.get("/api/graph/data", summary="取得知識圖譜 JSON 資料")
def get_graph_data(scheme: str = Query("", description="方案名稱")):
    _activate(scheme)
    json_path = os.path.join(graph_dir(), "knowledge_graph.json")
    if not os.path.isfile(json_path):
        raise HTTPException(404, "尚未建構圖譜。請先呼叫 POST /api/build-graph")
    import json
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return JSONResponse(content=data)


@app.get("/api/graph/relationship", response_class=HTMLResponse,
         summary="取得關係圖 HTML（可直接在瀏覽器顯示）")
def get_relationship_html(scheme: str = Query("", description="方案名稱")):
    _activate(scheme)
    html_path = os.path.join(graph_dir(), "relationship_graph.html")
    if not os.path.isfile(html_path):
        raise HTTPException(404, "尚未建構關係圖。請先呼叫 POST /api/build-graph")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/api/graph/knowledge", response_class=HTMLResponse,
         summary="取得知識圖譜 HTML（可直接在瀏覽器顯示）")
def get_knowledge_html(scheme: str = Query("", description="方案名稱")):
    _activate(scheme)
    html_path = os.path.join(graph_dir(), "knowledge_graph.html")
    if not os.path.isfile(html_path):
        raise HTTPException(404, "尚未建構知識圖譜。請先呼叫 POST /api/build-graph")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/api/graph/summary", summary="取得圖譜摘要（適合發送到 LINE 的純文字）")
def get_graph_summary(scheme: str = Query("", description="方案名稱")):
    """回傳純文字摘要，方便 n8n 直接轉發給 LINE。"""
    _activate(scheme)
    json_path = os.path.join(graph_dir(), "knowledge_graph.json")
    if not os.path.isfile(json_path):
        raise HTTPException(404, "尚未建構圖譜。請先呼叫 POST /api/build-graph")
    import json
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    personality = data.get("personality", {})

    lines = [f"📊 知識圖譜摘要（方案：{active_scheme() or '預設'}）", ""]

    lines.append(f"👶 共 {len(nodes)} 位幼兒：")
    for n in nodes:
        pers = "、".join(n.get("personality", ["觀察中"]))
        lines.append(f"  · {n['label']}（{pers}）")

    lines.append("")
    if edges:
        edges_sorted = sorted(edges, key=lambda e: e.get("cooccurrence", 0), reverse=True)
        lines.append(f"🤝 共 {len(edges)} 組朋友關係：")
        for e in edges_sorted[:10]:
            lines.append(f"  · {e['source']} ↔ {e['target']}：同框 {e.get('cooccurrence', 0)} 次")
        if len(edges) > 10:
            lines.append(f"  ...（還有 {len(edges) - 10} 組）")
    else:
        lines.append("尚無朋友關係（同框次數未達門檻）。")

    return {"scheme": active_scheme() or "預設", "summary": "\n".join(lines)}


# ---------------------------------------------------------------------------
# Ego graph API endpoint
# ---------------------------------------------------------------------------

@app.get("/api/graph/ego", response_class=HTMLResponse,
         summary="取得指定幼兒的 ego 圖譜 HTML")
def get_ego_html(
    scheme: str = Query("", description="方案名稱"),
    name: str = Query(..., description="幼兒名字"),
):
    _activate(scheme)
    json_path = os.path.join(graph_dir(), "knowledge_graph.json")
    if not os.path.isfile(json_path):
        raise HTTPException(404, "尚未建構圖譜。請先呼叫 POST /api/build-graph")
    import json
    with open(json_path, "r", encoding="utf-8") as f:
        kg_data = json.load(f)
    from knowledge_graph import ego_knowledge_graph
    from relationship_graph import draw_relationship_graph
    ego_data = ego_knowledge_graph(kg_data, name)
    ego_dir = os.path.join(graph_dir(), "ego")
    os.makedirs(ego_dir, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)
    out_path = os.path.join(ego_dir, f"ego_{safe}.html")
    draw_relationship_graph(
        kg_data=ego_data, output_html=out_path,
        title=f"幼兒關係圖 · {name}", focal_node_id=name,
    )
    with open(out_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


# ---------------------------------------------------------------------------
# LINE Bot Webhook
# ---------------------------------------------------------------------------

import httpx
import json as _json
import hashlib
import hmac
import base64
from fastapi import Request, Header
from urllib.parse import quote as _url_quote

LINE_CHANNEL_SECRET = os.environ.get(
    "LINE_CHANNEL_SECRET", "3040d808dcea3b9a99a3680a36148399"
)
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get(
    "LINE_CHANNEL_ACCESS_TOKEN",
    "7KnVvgcIozMf+Q82k7NztXikUUfb4hl8BbsWoYlEPVV/XrIR83Ok/1dw86zV/jrgHUxOpNZOkPh6ko2+4lJkrmDZZUKgI5lxxWwR3Ps6tw6nl3TN/aq7zi1kptSgrPUPJVUgonqwWiMTm44+APc2WgdB04t89/1O/w1cDnyilFU=",
)
LINE_DEFAULT_SCHEME = os.environ.get("LINE_DEFAULT_SCHEME", "甲班")
LINE_TEACHER_PASSWORD = os.environ.get("LINE_TEACHER_PASSWORD", "teacher123")
NGROK_PUBLIC_URL = os.environ.get("NGROK_PUBLIC_URL", "")

# 使用者 session：LINE User ID -> {"role": "teacher"} 或 {"role": "student", "child_name": "小孩1"}
_user_sessions: dict[str, dict] = {}


def _get_public_url() -> str:
    if NGROK_PUBLIC_URL:
        return NGROK_PUBLIC_URL.rstrip("/")
    try:
        import httpx as _httpx
        r = _httpx.get("http://localhost:4040/api/tunnels", timeout=2)
        tunnels = r.json().get("tunnels", [])
        for t in tunnels:
            if t.get("proto") == "https":
                return t["public_url"].rstrip("/")
    except Exception:
        pass
    return ""


def _get_registered_names(scheme: str) -> set[str]:
    _activate(scheme)
    try:
        names = get_unique_registered_names()
        return set(names) if names else set()
    except Exception:
        return set()


def _verify_line_signature(body: bytes, signature: str) -> bool:
    mac = hmac.new(
        LINE_CHANNEL_SECRET.encode("utf-8"), body, hashlib.sha256
    )
    return hmac.compare_digest(
        base64.b64encode(mac.digest()).decode("utf-8"), signature
    )


async def _line_reply(reply_token: str, messages: list[dict]):
    async with httpx.AsyncClient() as client:
        await client.post(
            "https://api.line.me/v2/bot/message/reply",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}",
            },
            json={"replyToken": reply_token, "messages": messages},
        )


def _build_full_summary(scheme: str) -> str:
    _activate(scheme)
    json_path = os.path.join(graph_dir(), "knowledge_graph.json")
    if not os.path.isfile(json_path):
        return f"方案「{scheme}」尚未建構圖譜。請先在電腦上執行影片處理與建構圖譜。"
    with open(json_path, "r", encoding="utf-8") as f:
        data = _json.load(f)
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    lines = [f"📊 知識圖譜摘要（{scheme}）", ""]
    lines.append(f"👶 共 {len(nodes)} 位幼兒：")
    for n in nodes:
        pers = "、".join(n.get("personality", ["觀察中"]))
        lines.append(f"  · {n['label']}（{pers}）")
    lines.append("")
    if edges:
        edges_sorted = sorted(edges, key=lambda e: e.get("cooccurrence", 0), reverse=True)
        lines.append(f"🤝 共 {len(edges)} 組朋友關係：")
        for e in edges_sorted[:10]:
            lines.append(f"  · {e['source']} ↔ {e['target']}：同框 {e.get('cooccurrence', 0)} 次")
        if len(edges) > 10:
            lines.append(f"  ...（還有 {len(edges) - 10} 組）")
    else:
        lines.append("尚無朋友關係。")
    return "\n".join(lines)


def _build_ego_summary(scheme: str, child_name: str) -> str:
    _activate(scheme)
    json_path = os.path.join(graph_dir(), "knowledge_graph.json")
    if not os.path.isfile(json_path):
        return f"方案「{scheme}」尚未建構圖譜。"
    with open(json_path, "r", encoding="utf-8") as f:
        kg_data = _json.load(f)
    from knowledge_graph import ego_knowledge_graph
    ego = ego_knowledge_graph(kg_data, child_name)
    nodes = ego.get("nodes", [])
    edges = ego.get("edges", [])
    focal = next((n for n in nodes if n["id"] == child_name), None)
    pers = "、".join(focal["personality"]) if focal else "觀察中"
    lines = [f"📊 {child_name} 的個人圖譜（{scheme}）", ""]
    lines.append(f"👤 個性：{pers}")
    lines.append("")
    if edges:
        edges_sorted = sorted(edges, key=lambda e: e.get("cooccurrence", 0), reverse=True)
        lines.append(f"🤝 共 {len(edges)} 位朋友：")
        for e in edges_sorted:
            peer = e["target"] if e["source"] == child_name else e["source"]
            lines.append(f"  · {peer}：同框 {e.get('cooccurrence', 0)} 次")
    else:
        lines.append("尚無朋友關係紀錄。")
    return "\n".join(lines)


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


@app.post("/webhook", summary="LINE Bot Webhook（接收使用者訊息並回覆）")
async def line_webhook(request: Request):
    body = await request.body()
    signature = request.headers.get("x-line-signature", "")
    if not _verify_line_signature(body, signature):
        raise HTTPException(403, "Invalid signature")

    payload = _json.loads(body)
    events = payload.get("events", [])

    for event in events:
        if event.get("type") != "message" or event["message"].get("type") != "text":
            continue

        text = event["message"]["text"].strip()
        reply_token = event["replyToken"]
        user_id = event["source"].get("userId", "")
        scheme = LINE_DEFAULT_SCHEME
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
            if text == LINE_TEACHER_PASSWORD:
                _user_sessions[user_id] = {"role": "teacher"}
                await _line_reply(reply_token, [
                    {"type": "text", "text": "✅ 老師身份驗證成功！\n\n" + TEACHER_HELP},
                ])
                continue

            registered = _get_registered_names(scheme)
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
                summary = _build_full_summary(scheme)
                messages = [{"type": "text", "text": summary}]
                pub = _get_public_url()
                if pub:
                    link_kg = f"{pub}/api/graph/knowledge?scheme={_url_quote(scheme)}"
                    link_rel = f"{pub}/api/graph/relationship?scheme={_url_quote(scheme)}"
                    messages.append({
                        "type": "text",
                        "text": (
                            "📈 互動式圖譜（請長按或點選下方完整網址）：\n"
                            "\n"
                            f"{link_kg}\n"
                            "\n"
                            f"{link_rel}"
                        ),
                    })
                await _line_reply(reply_token, messages)
            else:
                summary = _build_ego_summary(scheme, child_name)
                messages = [{"type": "text", "text": summary}]
                pub = _get_public_url()
                if pub:
                    link_ego = f"{pub}/api/graph/ego?scheme={_url_quote(scheme)}&name={_url_quote(child_name)}"
                    messages.append({
                        "type": "text",
                        "text": (
                            f"📈 {child_name} 的互動式圖譜（請長按或點選下方完整網址）：\n"
                            "\n"
                            f"{link_ego}"
                        ),
                    })
                await _line_reply(reply_token, messages)

        elif text in ("名單", "查名單", "註冊"):
            _activate(scheme)
            names, _ = get_registry_encodings()
            if not names:
                msg = f"方案「{scheme}」尚無註冊資料。"
            else:
                from collections import Counter
                c = Counter(names)
                msg = f"📋 {scheme} 已註冊 {len(c)} 人：\n"
                msg += "\n".join(f"  · {n}（{cnt} 筆模版）" for n, cnt in c.most_common())
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
                help_text = STUDENT_HELP_TPL.format(name=child_name)
                await _line_reply(reply_token, [{"type": "text", "text": help_text}])

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
# main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="0.0.0.0")
    args = p.parse_args()

    ensure_dirs()
    print(f"API 文件：http://localhost:{args.port}/docs")
    uvicorn.run(app, host=args.host, port=args.port)
