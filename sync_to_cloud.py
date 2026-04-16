"""將本機建好的圖譜資料同步到 Render 雲端 API。

用法：
    python sync_to_cloud.py --scheme 甲班 --url https://your-app.onrender.com --secret 你的密碼

--url 請填 Render 服務根網址（不要加 /webhook；LINE Webhook 才用 /webhook）。

首次使用前，請先在 Render 上設定好環境變數 SYNC_SECRET，
然後在本機用相同的 secret 值執行此腳本。
"""

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    p = argparse.ArgumentParser(description="同步本機圖譜資料到雲端 API")
    p.add_argument("--scheme", required=True, help="方案名稱（如：甲班）")
    p.add_argument(
        "--url",
        required=True,
        help="Render 服務根網址（如：https://xxx.onrender.com，勿加 /webhook）",
    )
    p.add_argument("--secret", default=os.environ.get("SYNC_SECRET", "changeme"),
                   help="同步密碼（需與雲端 SYNC_SECRET 環境變數一致）")
    args = p.parse_args()

    scheme = args.scheme.strip()
    base_url = args.url.rstrip("/")
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

    if kg_data is None and not names:
        print("[ERROR] 沒有任何資料可以同步。請先處理影片、建構圖譜。")
        sys.exit(1)

    # --- 上傳 ---
    payload = {
        "secret": secret,
        "scheme": scheme,
    }
    if kg_data is not None:
        payload["knowledge_graph"] = kg_data
    if names:
        payload["registered_names"] = names

    try:
        import httpx
    except ImportError:
        print("[INFO] 安裝 httpx...")
        os.system(f"{sys.executable} -m pip install httpx -q")
        import httpx

    sync_url = f"{base_url}/api/sync"
    print(f"\n正在上傳到 {sync_url} ...")

    try:
        r = httpx.post(sync_url, json=payload, timeout=60)
        if r.status_code == 200:
            result = r.json()
            print(f"[OK] 同步成功！")
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


if __name__ == "__main__":
    main()
