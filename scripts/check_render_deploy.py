#!/usr/bin/env python3
"""檢查遠端 Render 是否為本專案最新的 api_cloud（含分塊上傳等）。

用法：
    python scripts/check_render_deploy.py https://child-tracker-kg.onrender.com
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request


def _get(url: str, timeout: float = 25.0) -> tuple[int, str]:
    """回傳 (HTTP 狀態碼, 內文)。連線失敗時狀態碼為 -1，內文為錯誤說明。"""
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            body = str(e)
        return e.code, body
    except urllib.error.URLError as e:
        reason = getattr(e, "reason", e)
        return -1, f"URLError: {reason}"
    except TimeoutError:
        return -1, "TimeoutError: 連線逾時（服務休眠或網路慢，請稍後再試）"
    except OSError as e:
        return -1, f"OSError: {e}"
    except Exception as e:
        return -1, f"{type(e).__name__}: {e}"


def main() -> int:
    if len(sys.argv) < 2:
        print("用法: python scripts/check_render_deploy.py https://你的服務.onrender.com")
        return 2
    base = sys.argv[1].strip().rstrip("/")
    print(f"檢查：{base}\n")

    code, body = _get(base + "/")
    print(f"[1] GET /  → HTTP {code}")
    if code < 0:
        print(f"\n[ERROR] 無法連到雲端根路徑：{body}")
        print("   請檢查：網址是否正確、電腦網路／防火牆／校園網是否擋外連、或服務正在喚醒。")
        print("   可改用手機熱點或瀏覽器直接開同一網址比對。")
        return 3
    try:
        j = json.loads(body)
    except json.JSONDecodeError:
        print("    回應不是 JSON，可能不是本專案的 api_cloud。")
        return 1
    print(f"    app: {j.get('app', '(missing)')}")
    print(f"    deploy_mark: {j.get('deploy_mark', '(missing)')}")
    print(f"    video_chunk_probe: {j.get('video_chunk_probe', '(missing)')}")

    if j.get("service") != "child_tracker_kg cloud API":
        print("\n[FAIL] 回應不是本專案 api_cloud 的健康檢查 JSON。")
        return 1

    # 與本機 api_cloud 常數對照
    try:
        import os
        import importlib.util

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(root, "api_cloud.py")
        spec = importlib.util.spec_from_file_location("api_cloud_check", path)
        mod = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(mod)
        expected = getattr(mod, "API_CLOUD_DEPLOY_MARK", "")
    except Exception:
        expected = ""

    remote_mark = j.get("deploy_mark", "")
    if not remote_mark or (expected and remote_mark != expected):
        print(f"\n[FAIL] 雲端仍是舊版 api_cloud，或未部署到與本機相同的 commit。")
        print(f"   本機 deploy_mark 預期: {expected!r}")
        print(f"   線上 deploy_mark 實際: {remote_mark!r}")
        print("   請在 Render: Settings -> Build & Deploy -> 確認 Repository/Branch")
        print("   Start Command 須為: uvicorn api_cloud:app --host 0.0.0.0 --port $PORT")
        print("   （勿用 api:app；專案內 api.py 為另一支程式）")
        print("   然後 Manual Deploy -> Clear build cache & deploy")
        return 1

    paths = j.get("video_chunk_paths")
    if not isinstance(paths, list) or not paths:
        paths = [
            "/api/sync/video-chunk",
            "/api/sync-video-chunk",
            "/api/v1/video-chunk",
        ]
    c2, b2 = 404, ""
    probe = ""
    for rel in paths:
        probe = base + str(rel)
        c2, b2 = _get(probe)
        print(f"\n[2] GET {probe} → HTTP {c2}")
        if c2 < 0:
            print(f"    連線失敗：{b2}")
            continue
        if c2 == 200:
            try:
                j2 = json.loads(b2)
                print(f"    video_chunk: {j2.get('video_chunk')}")
            except json.JSONDecodeError:
                print(f"    內容: {b2[:200]}")
            break
        if c2 == 405:
            break
        print(f"    內容: {b2[:200]}")

    if c2 < 0:
        print("\n[FAIL] 分塊路徑皆無法連線（網路／防火牆／SSL），與 Render 版本無關。請換網路或稍後再試。")
        return 3

    if c2 not in (200, 405):
        print("\n[FAIL] 上述分塊探測路徑皆不可用（多為 404）。請在 Render：Manual Deploy -> Clear build cache & deploy")
        print("   並確認 Repository / Branch / Root Directory 正確（見 README「徹底檢查」）。")
        return 1

    print("\n[OK] 遠端看起來已為含分塊上傳的新版 api_cloud。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
