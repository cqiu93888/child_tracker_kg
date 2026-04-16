"""Gradio 網頁介面 — 讓組員透過瀏覽器使用完整功能（註冊、處理影片、建構圖譜、查看結果）。

啟動方式：
    python app_gradio.py            # http://localhost:7860
    python app_gradio.py --share    # 產生公開連結（可傳給他人）
"""

import os, sys, subprocess, shutil, tempfile, json, glob

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from runtime_hardening import apply as _runtime_hardening_apply
_runtime_hardening_apply()

import gradio as gr

from config import (
    ensure_dirs,
    SCHEMES_PARENT,
    PROJECT_ROOT,
    set_active_scheme,
    active_scheme,
    output_dir,
    graph_dir,
    registry_file,
    registered_dir,
    scheme_root,
)
from face_registry import register_face, get_registry_encodings

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _list_schemes() -> list[str]:
    if not os.path.isdir(SCHEMES_PARENT):
        return []
    return sorted(
        d for d in os.listdir(SCHEMES_PARENT)
        if os.path.isdir(os.path.join(SCHEMES_PARENT, d))
    )


def _activate(scheme: str | None):
    s = (scheme or "").strip()
    set_active_scheme(s if s else None)
    ensure_dirs()


def _registered_names(scheme: str | None) -> list[str]:
    _activate(scheme)
    names, _ = get_registry_encodings()
    return names or []


# ---------------------------------------------------------------------------
# Tab 1 — 臉部註冊
# ---------------------------------------------------------------------------

def fn_register(photo_path: str, name: str, scheme: str):
    if not photo_path:
        return "請上傳照片。"
    name = (name or "").strip()
    if not name:
        return "請輸入名字。"
    _activate(scheme)
    try:
        register_face(photo_path, name, prefer_cnn=False, allow_no_face=None)
    except (FileNotFoundError, ValueError, OSError) as e:
        return f"註冊失敗：{e}"
    return f"已成功註冊「{name}」（方案：{active_scheme() or '預設'}）\n註冊庫：{registry_file()}"


def fn_registry_info(scheme: str):
    _activate(scheme)
    names, encs = get_registry_encodings()
    if not names:
        return "（此方案尚無任何註冊資料）"
    from collections import Counter
    c = Counter(names)
    lines = [f"方案：{active_scheme() or '預設'}　共 {len(names)} 筆模版、{len(c)} 人"]
    for n, cnt in c.most_common():
        lines.append(f"  · {n}：{cnt} 筆模版")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tab 2 — 影片處理 (process)
# ---------------------------------------------------------------------------

def fn_process(video_path: str, scheme: str, seconds: float | None,
               start: float | None, use_yolo: bool, yolo_cpu: bool):
    if not video_path:
        return "請上傳影片。", None
    _activate(scheme)

    names, _ = get_registry_encodings()
    if not names:
        return "錯誤：此方案尚未註冊任何人臉。請先到「臉部註冊」頁完成註冊。", None

    out_name = "output_gradio.mp4"
    out_path = os.path.join(output_dir(), out_name)
    os.makedirs(output_dir(), exist_ok=True)

    cmd = [
        sys.executable, os.path.join(PROJECT_ROOT, "main.py"),
        "process",
        "--video", video_path,
        "--output", out_path,
    ]
    if scheme and scheme.strip():
        cmd.extend(["--scheme", scheme.strip()])
    if seconds:
        cmd.extend(["--seconds", str(seconds)])
    if start:
        cmd.extend(["--start", str(start)])
    if use_yolo:
        cmd.append("--yolo")
        if yolo_cpu:
            cmd.append("--yolo-cpu")
    else:
        cmd.append("--no-yolo")

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
    log = (result.stdout or "") + "\n" + (result.stderr or "")
    log = log.strip()

    if result.returncode != 0:
        return f"處理失敗（exit code {result.returncode}）：\n{log}", None

    if os.path.isfile(out_path):
        return f"處理完成。\n{log}", out_path
    return f"處理完成，但找不到輸出檔 {out_path}。\n{log}", None


# ---------------------------------------------------------------------------
# Tab 3 — 建構圖譜
# ---------------------------------------------------------------------------

def fn_build_graph(scheme: str, build_ego: bool):
    _activate(scheme)
    interactions = os.path.join(scheme_root(), "interactions.json") if active_scheme() else os.path.join(PROJECT_ROOT, "data", "interactions.json")
    if not os.path.isfile(interactions):
        return "找不到互動記錄 (interactions.json)。請先執行「影片處理」產生互動資料。", None

    from relationship_graph import run_build_and_draw
    kg_path, html_path, ego_paths = run_build_and_draw(
        output_dir=graph_dir(), build_ego=build_ego
    )
    msg = f"圖譜建構完成。\n知識圖譜 JSON：{kg_path}\n關係圖 HTML：{html_path}"
    if ego_paths:
        msg += f"\n個人關係圖：{len(ego_paths)} 份"
    return msg, html_path


# ---------------------------------------------------------------------------
# Tab 4 — 查看結果
# ---------------------------------------------------------------------------

def fn_list_outputs(scheme: str):
    _activate(scheme)
    od = output_dir()
    if not os.path.isdir(od):
        return gr.update(choices=[], value=None)
    vids = sorted(
        f for f in os.listdir(od)
        if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
    )
    paths = [os.path.join(od, v) for v in vids]
    return gr.update(choices=paths, value=paths[0] if paths else None)


def fn_load_video(path: str):
    if path and os.path.isfile(path):
        return path
    return None


def fn_load_graph_html(scheme: str):
    _activate(scheme)
    html_path = os.path.join(graph_dir(), "relationship_graph.html")
    if not os.path.isfile(html_path):
        return "<p style='padding:20px;color:#999;'>尚未建構關係圖。請先到「建構圖譜」頁執行。</p>"
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Build UI
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    schemes = _list_schemes()
    default_scheme = schemes[0] if schemes else ""

    with gr.Blocks(
        title="幼兒辨識追蹤與知識圖譜",
    ) as app:
        gr.Markdown("# 幼兒辨識追蹤與知識圖譜系統")
        gr.Markdown("上傳照片註冊幼兒 → 處理影片 → 建構關係圖，全部在瀏覽器完成。")

        # ---- Tab 1: 臉部註冊 ----
        with gr.Tab("臉部註冊"):
            with gr.Row():
                with gr.Column(scale=1):
                    reg_scheme = gr.Dropdown(
                        label="方案", choices=schemes, value=default_scheme,
                        allow_custom_value=True,
                    )
                    reg_photo = gr.Image(label="上傳照片（正面清晰為佳）", type="filepath")
                    reg_name = gr.Textbox(label="幼兒名字", placeholder="例如：小孩1")
                    reg_btn = gr.Button("註冊", variant="primary")
                with gr.Column(scale=1):
                    reg_result = gr.Textbox(label="結果", lines=4, interactive=False)
                    reg_info_btn = gr.Button("查看目前註冊名單")
                    reg_info = gr.Textbox(label="註冊名單", lines=10, interactive=False)

            reg_btn.click(fn_register, inputs=[reg_photo, reg_name, reg_scheme], outputs=reg_result)
            reg_info_btn.click(fn_registry_info, inputs=[reg_scheme], outputs=reg_info)

        # ---- Tab 2: 影片處理 ----
        with gr.Tab("影片處理"):
            with gr.Row():
                with gr.Column(scale=1):
                    proc_scheme = gr.Dropdown(
                        label="方案", choices=schemes, value=default_scheme,
                        allow_custom_value=True,
                    )
                    proc_video = gr.Video(label="上傳影片")
                    with gr.Row():
                        proc_start = gr.Number(label="起始秒數（可選）", value=None, precision=1)
                        proc_seconds = gr.Number(label="處理秒數（可選）", value=None, precision=1)
                    proc_yolo = gr.Checkbox(label="使用 YOLO 追蹤（建議開啟）", value=True)
                    proc_yolo_cpu = gr.Checkbox(label="YOLO 使用 CPU（較穩定、較慢）", value=True)
                    proc_btn = gr.Button("開始處理", variant="primary")
                with gr.Column(scale=1):
                    proc_log = gr.Textbox(label="處理紀錄", lines=12, interactive=False)
                    proc_output = gr.Video(label="輸出影片")

            proc_btn.click(
                fn_process,
                inputs=[proc_video, proc_scheme, proc_seconds, proc_start, proc_yolo, proc_yolo_cpu],
                outputs=[proc_log, proc_output],
            )

        # ---- Tab 3: 建構圖譜 ----
        with gr.Tab("建構圖譜"):
            with gr.Row():
                with gr.Column(scale=1):
                    graph_scheme = gr.Dropdown(
                        label="方案", choices=schemes, value=default_scheme,
                        allow_custom_value=True,
                    )
                    graph_ego = gr.Checkbox(label="同時建構每位幼兒的個人關係圖", value=False)
                    graph_btn = gr.Button("建構圖譜", variant="primary")
                with gr.Column(scale=1):
                    graph_log = gr.Textbox(label="結果", lines=6, interactive=False)
                    graph_html = gr.HTML(label="關係圖預覽")

            def _build_and_preview(scheme, ego):
                msg, html_path = fn_build_graph(scheme, ego)
                html = ""
                if html_path and os.path.isfile(html_path):
                    with open(html_path, "r", encoding="utf-8") as f:
                        html = f.read()
                return msg, f"<iframe srcdoc=\"{_escape_iframe(html)}\" style='width:100%;height:600px;border:1px solid #ddd;border-radius:8px;'></iframe>" if html else ""

            graph_btn.click(_build_and_preview, inputs=[graph_scheme, graph_ego], outputs=[graph_log, graph_html])

        # ---- Tab 4: 查看結果 ----
        with gr.Tab("查看結果"):
            with gr.Row():
                view_scheme = gr.Dropdown(
                    label="方案", choices=schemes, value=default_scheme,
                    allow_custom_value=True,
                )
                view_refresh = gr.Button("重新整理")

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 輸出影片")
                    view_video_dd = gr.Dropdown(label="選擇影片", choices=[], interactive=True)
                    view_video = gr.Video(label="影片播放")
                with gr.Column(scale=1):
                    gr.Markdown("### 關係圖")
                    view_graph = gr.HTML()

            def _refresh(scheme):
                dd_update = fn_list_outputs(scheme)
                html_content = fn_load_graph_html(scheme)
                iframe = f"<iframe srcdoc=\"{_escape_iframe(html_content)}\" style='width:100%;height:600px;border:1px solid #ddd;border-radius:8px;'></iframe>" if html_content and not html_content.startswith("<p") else html_content
                return dd_update, None, iframe

            view_refresh.click(_refresh, inputs=[view_scheme], outputs=[view_video_dd, view_video, view_graph])
            view_video_dd.change(fn_load_video, inputs=[view_video_dd], outputs=[view_video])

    return app


def _escape_iframe(html: str) -> str:
    return html.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--share", action="store_true", help="產生公開分享連結")
    p.add_argument("--port", type=int, default=7870)
    args = p.parse_args()

    ensure_dirs()
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)
