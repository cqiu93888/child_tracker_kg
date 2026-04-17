"""OpenCV 追蹤輸出 MP4：優先嘗試 H.264 fourcc，但須通過「寫一幀＋檢查 ftyp」才採用。

部分 Windows／pip 版 OpenCV 會出現 isOpened()==True 卻寫出損壞或極小的檔案，
導致電腦與手機皆無法播放。auto 模式會用暫存檔驗證後再建立正式 VideoWriter。

環境變數 VIDEO_OUTPUT_FOURCC：auto（預設，優先 avc1/H264 以利瀏覽器線上播放）、或四字元如 avc1、H264、mp4v（mp4v 常導致僅 VLC 可播）。
"""

from __future__ import annotations

import os
import tempfile

import cv2
import numpy as np

from mp4_inspect import is_likely_playable_mp4

_TAG_FROM_ENV = (os.environ.get("VIDEO_OUTPUT_FOURCC") or "auto").strip().lower()


def _try_writer_with_probe(tag: str, fps: float, w: int, h: int) -> bool:
    """寫入一張全黑畫面後檢查是否像有效 MP4。"""
    fcc = cv2.VideoWriter_fourcc(*tag)
    fd, tmp = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        out = cv2.VideoWriter(tmp, fcc, float(fps), (w, h))
        if not out.isOpened():
            return False
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        out.write(frame)
        out.write(frame)
        out.release()
        return is_likely_playable_mp4(tmp, min_bytes=32)
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def create_cv_video_writer(
    output_path: str,
    fps: float,
    frame_size: tuple[int, int],
) -> cv2.VideoWriter:
    w, h = int(frame_size[0]), int(frame_size[1])
    fps_f = float(fps)
    pref = _TAG_FROM_ENV
    if pref not in ("", "auto"):
        tag = (pref + "xxxx")[:4]
        fcc = cv2.VideoWriter_fourcc(*tag)
        out = cv2.VideoWriter(output_path, fcc, fps_f, (w, h))
        if not out.isOpened():
            raise IOError(
                f"無法建立追蹤輸出 VideoWriter（VIDEO_OUTPUT_FOURCC={pref!r}）：{output_path}"
            )
        return out

    for tag in ("avc1", "H264", "X264", "mp4v"):
        if not _try_writer_with_probe(tag, fps_f, w, h):
            continue
        fcc = cv2.VideoWriter_fourcc(*tag)
        out = cv2.VideoWriter(output_path, fcc, fps_f, (w, h))
        if out.isOpened():
            return out

    # 探測全失敗時退回舊行為（僅 isOpened）；仍優先 avc1/H264 以利瀏覽器內嵌播放
    for tag in ("avc1", "H264", "X264", "mp4v"):
        fcc = cv2.VideoWriter_fourcc(*tag)
        out = cv2.VideoWriter(output_path, fcc, fps_f, (w, h))
        if out.isOpened():
            return out

    raise IOError(
        f"無法建立追蹤輸出 VideoWriter：{output_path}。"
        "請確認 opencv 可寫入該路徑；或安裝含 ffmpeg 的 OpenCV 建置。"
    )
