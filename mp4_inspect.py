"""MP4／ISO-BMFF 檔頭快速檢查（不讀整檔）。用於雲端同步驗證與 OpenCV 寫入探針。"""

from __future__ import annotations

import os

# 掃描前 N bytes 尋找 stsd 常見影像 fourcc（啟發式，足供瀏覽器相容性提示）
_CODEC_SCAN_BYTES = min(2 * 1024 * 1024, 8 * 1024 * 1024)


def _guess_codec_from_chunk(chunk: bytes) -> tuple[str, bool]:
    """回傳 (codec_label, chrome_inline_video_likely)。

    chrome_inline_video_likely：Chrome／多數 Android WebView 的 <video> 對 H.264 內嵌支援佳；
    mp4v（MPEG-4 Part 2）常出現「可下載、VLC 可播，但瀏覽器線上播失敗」。
    """
    if b"avc1" in chunk or b"AVC1" in chunk:
        return "avc1_h264", True
    if b"hvc1" in chunk or b"hev1" in chunk or b"HVC1" in chunk or b"HEV1" in chunk:
        return "hevc_h265", False
    if b"mp4v" in chunk or b"MP4V" in chunk:
        return "mp4v_mpeg4part2", False
    return "unknown", False


def _moov_before_first_mdat(path: str, file_size: int) -> bool | None:
    """若第一個 mdat 出現在 moov 之前，多數瀏覽器較難邊下邊播（moov 在檔尾）。"""
    try:
        with open(path, "rb") as f:
            pos = 0
            first_mdat: int | None = None
            first_moov: int | None = None
            while pos + 8 <= file_size:
                f.seek(pos)
                h = f.read(8)
                if len(h) < 8:
                    break
                size32 = int.from_bytes(h[0:4], "big")
                typ = h[4:8]
                if size32 < 8:
                    break
                box_size = size32
                if size32 == 1:
                    ext = f.read(8)
                    if len(ext) < 8:
                        break
                    box_size = int.from_bytes(ext, "big")
                    if box_size < 16:
                        break
                if typ == b"mdat" and first_mdat is None:
                    first_mdat = pos
                if typ == b"moov" and first_moov is None:
                    first_moov = pos
                prev = pos
                pos += box_size
                if pos <= prev or pos > file_size:
                    break
            if first_mdat is None or first_moov is None:
                return None
            return first_moov < first_mdat
    except OSError:
        return None


def inspect_mp4_path(path: str) -> dict:
    """回傳診斷 dict：是否存在、大小、是否像標準 MP4（ftyp）、編碼與串流友善度啟發式。"""
    info: dict = {
        "path_exists": False,
        "size_bytes": 0,
        "prefix_hex": "",
        "looks_like_mp4": False,
        "major_brand": None,
        "hint": "",
        "codec_guess": None,
        "browser_inline_video_likely": None,
        "moov_before_mdat": None,
        "playback_hint_zh": "",
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
    else:
        idx = head.find(b"ftyp")
        if idx >= 4:
            info["looks_like_mp4"] = True
            info["hint"] = "偵測到 ftyp 但不在檔案最前；少數檔案仍可播"
        else:
            info["hint"] = "不是標準 ISO-BMFF／MP4（檔頭無 ftyp），瀏覽器通常無法播放"
            return info

    # --- 編碼／結構（僅在判定為 MP4 後）---
    try:
        with open(path, "rb") as f:
            scan_n = min(_CODEC_SCAN_BYTES, sz)
            f.seek(0)
            chunk = f.read(scan_n)
        label, browser_ok = _guess_codec_from_chunk(chunk)
        info["codec_guess"] = label
        info["browser_inline_video_likely"] = browser_ok
        info["moov_before_mdat"] = _moov_before_first_mdat(path, sz)
    except OSError:
        pass

    hints: list[str] = []
    if info.get("hint"):
        hints.append(str(info["hint"]))
    cg = info.get("codec_guess")
    if cg == "mp4v_mpeg4part2":
        hints.append(
            "偵測到 mp4v（MPEG-4 Part 2）編碼：VLC／下載後可播，但 Chrome 與多數手機內建瀏覽器常無法線上播放。"
            " 請在本機追蹤時使用預設 VIDEO_OUTPUT_FOURCC=auto（會優先嘗試 avc1/H264），"
            "或手動設 avc1／H264 後重跑輸出並再 sync；亦可用 ffmpeg 轉成 libx264（-c:v libx264 -movflags +faststart）。"
        )
    elif cg == "hevc_h265":
        hints.append(
            "偵測到 HEVC（hvc1/hev1）：Safari／部分裝置可播，Chrome 桌面版常無法內嵌播放。"
            " 建議改輸出 H.264（avc1）後再同步。"
        )
    mb = info.get("moov_before_mdat")
    if mb is False:
        hints.append(
            "偵測到 moov 在大型 mdat 之後（典型「moov 在檔尾」）：部分瀏覽器線上播放較不穩。"
            " 可在本機執行：ffmpeg -i 輸入.mp4 -c copy -movflags +faststart 輸出.mp4 後再同步。"
        )
    info["playback_hint_zh"] = " ".join(hints).strip()
    return info


def is_likely_playable_mp4(path: str, min_bytes: int = 32) -> bool:
    """給 OpenCV 探針用：檔案夠大且像 MP4。"""
    inf = inspect_mp4_path(path)
    if inf["size_bytes"] < min_bytes:
        return False
    return bool(inf["looks_like_mp4"])
