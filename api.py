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
