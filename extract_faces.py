# 從影片自動採樣、偵測臉部並裁成小圖，供手動分類後再 register
from __future__ import annotations

import os
import json
import argparse
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
import face_recognition
from PIL import Image

from config import DATA_DIR, ensure_dirs


def _crop_face_rgb(rgb: np.ndarray, loc, padding: float = 0.35):
    """loc: (top, right, bottom, left)，回傳裁切後 RGB 圖（可能為空）。"""
    top, right, bottom, left = loc
    h, w = rgb.shape[:2]
    fh, fw = bottom - top, right - left
    if fh <= 0 or fw <= 0:
        return None
    pad_y = int(fh * padding)
    pad_x = int(fw * padding)
    t = max(0, top - pad_y)
    l = max(0, left - pad_x)
    b = min(h, bottom + pad_y)
    r = min(w, right + pad_x)
    if b <= t or r <= l:
        return None
    return rgb[t:b, l:r].copy()


def extract_faces_from_video(
    video_path: str,
    out_dir: Optional[str] = None,
    every_n_frames: int = 20,
    max_crops: int = 400,
    min_face_area_frac: float = 0.0012,
    padding: float = 0.35,
    dedup_distance: float = 0.22,
    model: str = "hog",
    start_seconds: Optional[float] = None,
    max_seconds: Optional[float] = None,
) -> dict:
    """
    每隔 every_n_frames 取一幀，偵測臉並存成 jpg。
    同一支影片內用 encoding 去重（距離 < dedup_distance 視為重複角度，不另存）。
    回傳統計 dict。
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"找不到影片: {video_path}")

    ensure_dirs()
    base = os.path.join(DATA_DIR, "extracted")
    os.makedirs(base, exist_ok=True)
    if out_dir is None:
        stem = os.path.splitext(os.path.basename(video_path))[0]
        safe = "".join(c if c.isalnum() or c in "._- " else "_" for c in stem)[:80]
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(base, f"{safe}_{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    out_dir = os.path.abspath(out_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟影片: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    start_frame = 0
    if start_seconds is not None and start_seconds > 0:
        start_frame = int(start_seconds * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    end_frame = total_frames if total_frames > 0 else None
    if max_seconds is not None and max_seconds > 0 and end_frame is not None:
        end_frame = min(end_frame, start_frame + int(max_seconds * fps))

    saved = 0
    scanned = 0
    skipped_dup = 0
    skipped_small = 0
    no_face_frames = 0
    write_failed = 0
    manifest = []
    known_encodings: list = []

    frame_idx = start_frame
    model_kw = model if model in ("hog", "cnn") else "hog"

    while saved < max_crops:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if end_frame is not None and frame_idx >= end_frame:
            break

        if (frame_idx - start_frame) % every_n_frames != 0:
            frame_idx += 1
            continue

        scanned += 1
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        fh, fw = rgb.shape[0], rgb.shape[1]
        frame_area = fh * fw

        locations = face_recognition.face_locations(rgb, model=model_kw)
        if not locations:
            no_face_frames += 1
            frame_idx += 1
            continue

        encodings = face_recognition.face_encodings(rgb, locations)
        for fi, (loc, enc) in enumerate(zip(locations, encodings)):
            top, right, bottom, left = loc
            face_area = (bottom - top) * (right - left)
            if face_area < min_face_area_frac * frame_area:
                skipped_small += 1
                continue

            if known_encodings:
                dists = face_recognition.face_distance(known_encodings, np.array(enc))
                if float(np.min(dists)) < dedup_distance:
                    skipped_dup += 1
                    continue

            crop = _crop_face_rgb(rgb, loc, padding=padding)
            if crop is None or crop.size == 0:
                continue

            t_sec = frame_idx / fps if fps else 0.0
            fname = f"f{frame_idx:06d}_face{fi}_{saved:04d}.jpg"
            fpath = os.path.join(out_dir, fname)
            bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
            ok_write = bool(cv2.imwrite(fpath, bgr))
            if not ok_write or not os.path.isfile(fpath) or os.path.getsize(fpath) == 0:
                try:
                    Image.fromarray(crop).save(fpath, format="JPEG", quality=92, subsampling=0)
                except OSError:
                    pass
                ok_write = os.path.isfile(fpath) and os.path.getsize(fpath) > 0
            if not ok_write:
                write_failed += 1
                continue

            known_encodings.append(np.array(enc))
            manifest.append(
                {
                    "file": fname,
                    "frame": frame_idx,
                    "time_sec": round(t_sec, 3),
                    "bbox_top": top,
                    "bbox_right": right,
                    "bbox_bottom": bottom,
                    "bbox_left": left,
                }
            )
            saved += 1
            if saved >= max_crops:
                break

        frame_idx += 1

    cap.release()

    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "video": os.path.abspath(video_path),
                "every_n_frames": every_n_frames,
                "saved_count": saved,
                "items": manifest,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    readme = os.path.join(out_dir, "README_請手動分類.txt")
    with open(readme, "w", encoding="utf-8") as f:
        f.write(
            "這些小圖是程式從影片自動裁切的臉部，請依幼兒手動分資料夾後再註冊：\n\n"
            "  python main.py register --photo \"本資料夾內某張.jpg\" --name \"幼兒名字\"\n\n"
            "同一人可對多張圖重複 register（多模版）。\n"
        )

    return {
        "out_dir": out_dir,
        "saved": saved,
        "frames_scanned": scanned,
        "skipped_duplicate": skipped_dup,
        "skipped_too_small": skipped_small,
        "frames_without_face": no_face_frames,
        "write_failed": write_failed,
        "manifest": manifest_path,
    }


def main_cli():
    p = argparse.ArgumentParser(description="從影片自動擷取臉部小圖（供手動分類後 register）")
    p.add_argument("--video", "-v", required=True, help="輸入影片路徑")
    p.add_argument("--out-dir", "-o", default=None, help="輸出資料夾（預設 data/extracted/影片名_時間）")
    p.add_argument("--every-n", type=int, default=20, help="每隔幾幀採樣一次（預設 20）")
    p.add_argument("--max-crops", type=int, default=400, help="最多存幾張臉（預設 400）")
    p.add_argument(
        "--min-face",
        type=float,
        default=0.0012,
        help="臉框面積至少占畫面比例（預設 0.0012，過小遠景會略過）",
    )
    p.add_argument("--padding", type=float, default=0.35, help="裁切時外扩比例（預設 0.35）")
    p.add_argument(
        "--dedup",
        type=float,
        default=0.22,
        help="與已存臉 encoding 距離小於此值視為重複、不存（預設 0.22）",
    )
    p.add_argument("--model", choices=["hog", "cnn"], default="hog", help="偵測模型：hog 較快、cnn 較準但慢")
    p.add_argument("--start", type=float, default=None, metavar="S", help="從第 S 秒開始")
    p.add_argument("--seconds", type=float, default=None, metavar="N", help="只處理前 N 秒（自 start 起算）")
    args = p.parse_args()
    r = extract_faces_from_video(
        args.video,
        out_dir=args.out_dir,
        every_n_frames=args.every_n,
        max_crops=args.max_crops,
        min_face_area_frac=args.min_face,
        padding=args.padding,
        dedup_distance=args.dedup,
        model=args.model,
        start_seconds=args.start,
        max_seconds=args.seconds,
    )
    print("擷取完成。")
    print(f"  輸出目錄: {r['out_dir']}")
    print(f"  已存臉圖: {r['saved']} 張")
    print(f"  採樣幀數: {r['frames_scanned']}")
    print(f"  略過（重複角度）: {r['skipped_duplicate']}")
    print(f"  略過（臉太小）: {r['skipped_too_small']}")
    if r.get("write_failed"):
        print(f"  寫入失敗（磁碟／權限／雲端同步）: {r['write_failed']} 次")
    print(f"  採樣幀內完全無臉: {r['frames_without_face']}")
    print(f"  清單: {r['manifest']}")
    print("\n請打開輸出資料夾，手動分類後執行：")
    print('  python main.py register --photo "路徑\\圖.jpg" --name "名字"')


if __name__ == "__main__":
    main_cli()
