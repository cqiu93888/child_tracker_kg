"""MP4／ISO-BMFF 檔頭快速檢查（不讀整檔）。用於雲端同步驗證與 OpenCV 寫入探針。"""

from __future__ import annotations

import os


def inspect_mp4_path(path: str) -> dict:
    """回傳診斷 dict：是否存在、大小、是否像標準 MP4（ftyp）。"""
    info: dict = {
        "path_exists": False,
        "size_bytes": 0,
        "prefix_hex": "",
        "looks_like_mp4": False,
        "major_brand": None,
        "hint": "",
    }
    if not path or not os.path.isfile(path):
        info["hint"] = "檔案不存在或路徑無效"
        return info
    info["path_exists"] = True
    try:
        sz = os.path.getsize(path)
    except OSError as e:
        info["hint"] = f"無法讀取大小：{e}"
        return info
    info["size_bytes"] = sz
    if sz < 32:
        info["hint"] = "檔案過小（未完成上傳或內容損毀）"
        return info
    try:
        with open(path, "rb") as f:
            head = f.read(min(4096, sz))
    except OSError as e:
        info["hint"] = f"無法讀取檔頭：{e}"
        return info
    info["prefix_hex"] = head[:32].hex()
    if head[:4] == b"RIFF":
        info["hint"] = "實際為 AVI（RIFF），副檔名卻是 .mp4，瀏覽器不會當 MP4 播放"
        return info
    if len(head) >= 12 and head[4:8] == b"ftyp":
        info["looks_like_mp4"] = True
        info["major_brand"] = head[8:12].decode("ascii", errors="replace")
        return info
    idx = head.find(b"ftyp")
    if idx >= 4:
        info["looks_like_mp4"] = True
        info["hint"] = "偵測到 ftyp 但不在檔案最前；少數檔案仍可播"
        return info
    info["hint"] = "不是標準 ISO-BMFF／MP4（檔頭無 ftyp），瀏覽器通常無法播放"
    return info


def is_likely_playable_mp4(path: str, min_bytes: int = 32) -> bool:
    """給 OpenCV 探針用：檔案夠大且像 MP4。"""
    inf = inspect_mp4_path(path)
    if inf["size_bytes"] < min_bytes:
        return False
    return bool(inf["looks_like_mp4"])
