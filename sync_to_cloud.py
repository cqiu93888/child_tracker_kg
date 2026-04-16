"""將本機建好的圖譜資料同步到 Render 雲端 API。

用法：
    python sync_to_cloud.py --scheme 甲班 --url https://your-app.onrender.com --secret 你的密碼

也可在本機設定環境變數後只打方案名（減少重複輸入）：
    set CLOUD_SYNC_URL=https://your-app.onrender.com
    set SYNC_SECRET=你的密碼
    python sync_to_cloud.py --scheme 甲班

--url 請填 Render 服務根網址（不要加 /webhook；LINE Webhook 才用 /webhook）。

首次使用前，請先在 Render 上設定好環境變數 SYNC_SECRET，
然後在本機用相同的 secret 值執行此腳本。

想「不必因主機休眠而反覆 sync」：請在 Render 掛 Persistent Disk 並設 CLOUD_DATA_DIR（見 README）。

加上 --with-videos 可上傳該方案相關目錄內**全部**追蹤輸出影片；若只要某一檔請用 --video output3.mp4（可重複）。
影片預設以 POST /api/sync/video-chunk 分塊上傳（避免 Render 502）；小檔亦可手動用 multipart 的 POST /api/sync。
預設先找 data/schemes/<方案>/output；若為空，會改找 data/output。
"""

import argparse
import glob
import io
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

_VIDEO_GLOB = ("*.mp4", "*.mov", "*.webm", "*.avi", "*.mkv")

_CHUNK_PATHS = ("/api/sync/video-chunk", "/api/sync-video-chunk")


def _resolve_video_chunk_url(httpx_mod, base_url: str) -> str:
    """回傳可用的分塊上傳 POST 基底 URL；若雲端為舊版則回傳空字串。"""
    override = (os.environ.get("CLOUD_SYNC_VIDEO_CHUNK_URL") or "").strip().rstrip("/")
    if override:
        return override
    root = base_url.rstrip("/")
    for path in _CHUNK_PATHS:
        url = root + path
        try:
            g = httpx_mod.get(url, timeout=25.0)
        except Exception:
            continue
        if g.status_code == 200:
            try:
                data = g.json()
            except Exception:
                data = {}
            if data.get("video_chunk") is True:
                return url
        # 僅註冊 POST、未加 GET 探測的舊版：GET 會 405，仍視為端點存在
        if g.status_code == 405:
            return url
    return ""


def _upload_videos_in_chunks(httpx_mod, base_url: str, secret: str, scheme: str, paths: list[str]) -> None:
    """大檔分塊 POST，減少 Render 閘道 502（單請求過大／過久）。"""
    chunk_url = _resolve_video_chunk_url(httpx_mod, base_url)
    if not chunk_url:
        print("[ERROR] 雲端找不到「分塊上傳」API（GET /api/sync/video-chunk 回傳 404）。")
        print("        代表 Render 上跑的還不是 GitHub 最新版（尚未含 video-chunk）。")
        print("        請到：Render Dashboard → 你的 Web Service → Manual Deploy → Deploy latest commit")
        print("        並確認連線的 Branch 為 main、Build 成功後再重跑本指令。")
        sys.exit(1)
    print(f"[INFO] 分塊上傳端點：{chunk_url}")
    chunk_mb = int(os.environ.get("SYNC_UPLOAD_CHUNK_MB", "8"))
    chunk_bytes = max(1, chunk_mb) * 1024 * 1024
    timeout = float(os.environ.get("SYNC_UPLOAD_CHUNK_TIMEOUT", "180.0"))
    for local_path in paths:
        bn = os.path.basename(local_path)
        sz = os.path.getsize(local_path)
        print(f"    · {bn}（約 {sz / (1024 * 1024):.1f} MB）分塊上傳（每塊 {chunk_mb} MB）…")
        r = httpx_mod.post(
            chunk_url,
            data={
                "secret": secret,
                "scheme": scheme,
                "filename": bn,
                "phase": "start",
            },
            timeout=timeout,
        )
        if r.status_code != 200:
            print(f"[ERROR] 影片 start 失敗（HTTP {r.status_code}）")
            print(f"     {r.text}")
            sys.exit(1)
        n = 0
        with open(local_path, "rb") as f:
            while True:
                buf = f.read(chunk_bytes)
                if not buf:
                    break
                n += 1
                bio = io.BytesIO(buf)
                r = httpx_mod.post(
                    chunk_url,
                    data={
                        "secret": secret,
                        "scheme": scheme,
                        "filename": bn,
                        "phase": "append",
                    },
                    files=[
                        (
                            "chunk",
                            (f"part{n}.bin", bio, "application/octet-stream"),
                        )
                    ],
                    timeout=timeout,
                )
                if r.status_code != 200:
                    print(f"[ERROR] 影片第 {n} 塊 append 失敗（HTTP {r.status_code}）")
                    print(f"     {r.text}")
                    sys.exit(1)
        r = httpx_mod.post(
            chunk_url,
            data={
                "secret": secret,
                "scheme": scheme,
                "filename": bn,
                "phase": "finish",
            },
            timeout=timeout,
        )
        if r.status_code != 200:
            print(f"[ERROR] 影片 finish 失敗（HTTP {r.status_code}）")
            print(f"     {r.text}")
            sys.exit(1)
        print(f"    [OK] 已寫入雲端：{', '.join(r.json().get('saved', []))}")


def _video_paths_in_dir(out_dir: str) -> list[str]:
    """掃描目錄根層與一層子資料夾內的常見影片副檔名。"""
    if not os.path.isdir(out_dir):
        return []
    found: list[str] = []
    for pat in _VIDEO_GLOB:
        found.extend(glob.glob(os.path.join(out_dir, pat)))
        found.extend(glob.glob(os.path.join(out_dir, "*", pat)))
    return sorted(set(found))


def main():
    p = argparse.ArgumentParser(description="同步本機圖譜資料到雲端 API")
    p.add_argument("--scheme", required=True, help="方案名稱（如：甲班）")
    p.add_argument(
        "--url",
        default=os.environ.get("CLOUD_SYNC_URL", "").strip(),
        help="雲端 API 根網址；若省略則讀環境變數 CLOUD_SYNC_URL（勿加 /webhook）",
    )
    p.add_argument("--secret", default=os.environ.get("SYNC_SECRET", "changeme"),
                   help="同步密碼（需與雲端 SYNC_SECRET 環境變數一致）")
    p.add_argument(
        "--with-videos",
        action="store_true",
        help="上傳 output 內全部可辨識的追蹤輸出影片",
    )
    p.add_argument(
        "--video",
        action="append",
        default=None,
        metavar="檔名",
        help="只上傳指定檔名（如 output3.mp4）；可重複此參數指定多檔",
    )
    args = p.parse_args()
    video_names = [str(x).strip() for x in (args.video or []) if str(x).strip()]

    scheme = args.scheme.strip()
    base_url = (args.url or "").strip().rstrip("/")
    if not base_url:
        print("[ERROR] 請提供 --url 或在環境變數設定 CLOUD_SYNC_URL（Render 服務根網址）。")
        sys.exit(1)
    if base_url.lower().endswith("/webhook"):
        base_url = base_url[: -len("/webhook")].rstrip("/")
        print("[INFO] 已將 --url 的 /webhook 尾碼移除（同步應使用服務根網址）。")
    secret = str(args.secret).strip()

    scheme_safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in scheme)
    scheme_dir = os.path.join(PROJECT_ROOT, "data", "schemes", scheme)
    if not os.path.isdir(scheme_dir):
        scheme_dir = os.path.join(PROJECT_ROOT, "data", "schemes", scheme_safe)
    if not os.path.isdir(scheme_dir):
        print(f"[ERROR] 找不到方案目錄：data/schemes/{scheme}")
        sys.exit(1)

    # --- 讀取 knowledge_graph.json ---
    kg_path = os.path.join(scheme_dir, "graph", "knowledge_graph.json")
    kg_data = None
    if os.path.isfile(kg_path):
        with open(kg_path, "r", encoding="utf-8") as f:
            kg_data = json.load(f)
        print(f"[OK] 已讀取 knowledge_graph.json（{len(kg_data.get('nodes', []))} 個節點）")
    else:
        print(f"[WARN] 找不到 {kg_path}，跳過圖譜資料")

    # --- 讀取已註冊名字（從 face_registry.json）---
    reg_path = os.path.join(scheme_dir, "face_registry.json")
    names = []
    if os.path.isfile(reg_path):
        with open(reg_path, "r", encoding="utf-8") as f:
            reg_data = json.load(f)
        if isinstance(reg_data, dict):
            raw_names = reg_data.get("names", [])
        elif isinstance(reg_data, list):
            raw_names = [e.get("name", "") for e in reg_data if isinstance(e, dict)]
        else:
            raw_names = []
        seen = set()
        for n in raw_names:
            if n and n not in seen:
                seen.add(n)
                names.append(n)
        print(f"[OK] 已讀取 face_registry.json（{len(names)} 人）")
    else:
        print(f"[WARN] 找不到 {reg_path}，跳過註冊名單")

    out_dir = os.path.join(scheme_dir, "output")
    all_video_paths = _video_paths_in_dir(out_dir)
    if not all_video_paths:
        fallback = os.path.join(PROJECT_ROOT, "data", "output")
        alt = _video_paths_in_dir(fallback)
        if alt:
            print(f"\n[INFO] 方案 output 無影片：{out_dir}")
            print(f"[INFO] 改從全域輸出目錄讀取候選：{fallback}（共 {len(alt)} 支）")
            all_video_paths = alt

    if video_names:
        want = {os.path.basename(n) for n in video_names}
        paths = [p for p in all_video_paths if os.path.basename(p) in want]
        found_names = {os.path.basename(p) for p in paths}
        missing = want - found_names
        if missing:
            print(f"[ERROR] 找不到下列影片（請確認檔名與副檔名）：{', '.join(sorted(missing))}")
            print(f"       已搜尋：{out_dir} 與 data/output/")
            sys.exit(1)
    elif args.with_videos:
        paths = list(all_video_paths)
    else:
        paths = []

    has_json_payload = kg_data is not None or bool(names)
    if not has_json_payload and not paths:
        print("[ERROR] 沒有圖譜／名單可同步，也未指定要上傳的影片。")
        print("       請先建構圖譜，或使用 --with-videos 或 --video 檔名.mp4")
        sys.exit(1)

    try:
        import httpx
    except ImportError:
        print("[INFO] 安裝 httpx...")
        os.system(f"{sys.executable} -m pip install httpx -q")
        import httpx

    sync_url = f"{base_url}/api/sync"

    if has_json_payload:
        payload = {"secret": secret, "scheme": scheme}
        if kg_data is not None:
            payload["knowledge_graph"] = kg_data
        if names:
            payload["registered_names"] = names
        print(f"\n正在上傳圖譜／名單到 {sync_url} ...")
        try:
            r = httpx.post(sync_url, json=payload, timeout=60)
            if r.status_code == 200:
                result = r.json()
                print("[OK] 同步成功！")
                print(f"     方案：{result.get('scheme')}")
                print(f"     已儲存：{', '.join(result.get('saved', []))}")
            else:
                print(f"[ERROR] 同步失敗（HTTP {r.status_code}）")
                print(f"     {r.text}")
                sys.exit(1)
        except httpx.ConnectError:
            print(f"[ERROR] 無法連線到 {base_url}")
            print("     請確認 Render 服務已啟動，且網址正確。")
            sys.exit(1)
        except Exception as e:
            print(f"[ERROR] {e}")
            sys.exit(1)
        print(f"\n完成！LINE Bot 現在可以使用「{scheme}」方案的圖譜資料。")
        print(f"Webhook URL：{base_url}/webhook")
    elif paths:
        print("[INFO] 本次僅上傳影片（略過圖譜 JSON）。")

    if paths:
        print(
            f"\n正在上傳 {len(paths)} 支影片（分塊 API，避免單次請求過大導致 Render 502）…"
        )
        try:
            _upload_videos_in_chunks(httpx, base_url, secret, scheme, paths)
            print("[OK] 影片同步完成。")
        except httpx.ConnectError:
            print(f"[ERROR] 無法連線到 {base_url}")
            sys.exit(1)
        except Exception as e:
            print(f"[ERROR] 影片同步：{e}")
            sys.exit(1)
        print(
            "\n若仍出現 502：請在 Render 重新部署最新程式，或縮小影片、"
            "或設定環境變數 SYNC_UPLOAD_CHUNK_MB=4 改小每塊大小。"
        )
        if not has_json_payload:
            print(f"\n完成！Webhook：{base_url}/webhook")
    elif args.with_videos:
        print("\n[WARN] --with-videos：找不到可上傳的影片（.mp4 / .mov / .webm / .avi / .mkv）。")
        print(f"       請確認檔案在：\n         {out_dir}\n       或 data/output/")


if __name__ == "__main__":
    main()
