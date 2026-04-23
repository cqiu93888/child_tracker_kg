# 影片追蹤：辨識已註冊幼兒、疊加名字、收集同框/互動資料
import os
import json
import time
from collections import defaultdict

import cv2
import numpy as np

from config import (
    USE_APPEARANCE_AUXILIARY,
    ensure_dirs,
    output_dir,
    interactions_file,
    FACE_MATCH_TOLERANCE,
    NEAR_DISTANCE_PX,
    SAME_FRAME_COOLDOWN,
    TRACK_SMOOTHING_FRAMES,
    TRACK_PERSISTENCE_FRAMES,
    TRACK_USE_OPENCV_TRACKER,
    DETECT_EVERY_N_FRAMES,
)
from face_registry import get_registry_bundle, match_face_encoding
from appearance_features import signature_from_face_box
import face_engine as _fe
from draw_text_cn import draw_label_cn
from cv_video_writer import create_cv_video_writer


# OpenCV 追蹤器 API 依版本可能在 cv2 或 cv2.legacy
def _create_tracker():
    for create in [
        lambda: cv2.legacy.TrackerKCF_create(),
        lambda: cv2.legacy.TrackerCSRT_create(),
        lambda: getattr(cv2, "TrackerKCF_create", lambda: None)(),
        lambda: getattr(cv2, "TrackerCSRT_create", lambda: None)(),
    ]:
        try:
            tracker = create()
            if tracker is not None:
                return tracker
        except Exception:
            continue
    return None


def _smooth_box(prev_box, new_box, alpha=0.4):
    """
    用指數移動平均平滑 bbox，減少追蹤框抖動。
    alpha 愈小愈平滑，愈大愈跟隨當前偵測。
    """
    if prev_box is None or new_box is None:
        return new_box or prev_box
    top = int(prev_box[0] * (1 - alpha) + new_box[0] * alpha)
    right = int(prev_box[1] * (1 - alpha) + new_box[1] * alpha)
    bottom = int(prev_box[2] * (1 - alpha) + new_box[2] * alpha)
    left = int(prev_box[3] * (1 - alpha) + new_box[3] * alpha)
    return (top, right, bottom, left)


def _distance(box1, box2):
    """兩個 bbox (top, right, bottom, left) 中心點距離。"""
    c1 = ((box1[3] + box1[1]) / 2, (box1[0] + box1[2]) / 2)
    c2 = ((box2[3] + box2[1]) / 2, (box2[0] + box2[2]) / 2)
    return np.hypot(c1[0] - c2[0], c1[1] - c2[1])


def _box_to_xywh(box):
    """(top, right, bottom, left) -> (x, y, w, h) for OpenCV tracker."""
    top, right, bottom, left = box
    return (left, top, right - left, bottom - top)


def _xywh_to_box(rect):
    """(x, y, w, h) -> (top, right, bottom, left)."""
    x, y, w, h = rect
    return (y, x + w, y + h, x)


def _match_detections_to_previous(candidates, smoothed_boxes):
    """依與上一幀位置最近對應偵測，避免多人時名字對調。回傳 {name: box}。"""
    if not candidates:
        return {}
    name_to_box = {}
    used = [False] * len(candidates)
    for name in list(smoothed_boxes.keys()):
        prev = smoothed_boxes[name]
        best_idx, best_dist = None, float("inf")
        for i, (box, n) in enumerate(candidates):
            if used[i] or n != name:
                continue
            d = _distance(box, prev)
            if d < best_dist:
                best_dist = d
                best_idx = i
        if best_idx is not None:
            used[best_idx] = True
            name_to_box[name] = candidates[best_idx][0]
    for i, (box, name) in enumerate(candidates):
        if not used[i] and name != "未知" and name not in name_to_box:
            name_to_box[name] = box
    return name_to_box


def _draw_label(frame, box, label, color, font_scale=0.8, thickness=2):
    """在 bbox 上方畫名字（支援中文）。box = (top, right, bottom, left)。"""
    top, right, bottom, left = box
    box_xyxy = (left, top, right, bottom)
    font_size = max(18, int(24 * (frame.shape[0] / 480)))
    draw_label_cn(frame, box_xyxy, label, color, font_size=font_size)


def process_video(
    video_path: str,
    output_path: str = None,
    primary_name: str = None,
    collect_interactions: bool = True,
    max_seconds: float = None,
    start_seconds: float = None,
) -> dict:
    """
    處理影片：
    - 辨識已註冊的臉，在畫面上疊加名字
    - 若指定 primary_name，優先穩定追蹤該幼兒
    - 若 collect_interactions 為 True，會寫入互動記錄（config.interactions_file()）
    回傳統計：{ "frames": N, "primary_visible_frames": M, "interactions": ... }
    """
    ensure_dirs()
    if output_path is None:
        output_path = os.path.join(output_dir(), "output.mp4")

    names, known_encodings, known_appearances = get_registry_bundle()
    if not names:
        raise ValueError("尚未註冊任何幼兒，請先執行 register")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"無法開啟影片: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 25.0
    fps = float(fps)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if start_seconds is not None and start_seconds > 0:
        start_frame = int(start_seconds * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        total_frames = max(0, total_frames - start_frame)
    out = create_cv_video_writer(output_path, fps, (w, h))

    # 互動記錄：同框 (name1, name2) -> 次數；靠近 (name1, name2) -> 次數
    cooccurrence = defaultdict(int)  # (n1, n2) 且 n1 < n2
    near_count = defaultdict(int)
    last_cooccur_time = {}  # (n1,n2) -> last timestamp
    frame_index = 0
    primary_visible = 0
    # 平滑：上一幀辨識到的 primary 的 bbox，用於暫時遺失時仍顯示名字
    primary_last_box = None
    primary_last_name = primary_name
    smooth_alpha = 0.22
    smoothed_boxes = {}
    last_seen_frame = {}
    trackers = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 本幀時間（秒），只用於同框統計，勿被其他變數覆寫
        time_sec = frame_index / fps if fps else 0.0

        run_detection = (frame_index % DETECT_EVERY_N_FRAMES == 0) or frame_index < 2
        name_to_box = {}
        current_names = []

        if run_detection:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, encodings_this = _fe.detect_and_encode(rgb)
            candidates = []
            for (top, right, bottom, left), enc in zip(boxes, encodings_this):
                sig = (
                    signature_from_face_box(rgb, (top, right, bottom, left))
                    if USE_APPEARANCE_AUXILIARY
                    else None
                )
                mname, _, _ = match_face_encoding(
                    names,
                    known_encodings,
                    enc,
                    FACE_MATCH_TOLERANCE,
                    query_appearance=sig,
                    known_appearances=known_appearances,
                )
                name = mname if mname is not None else "未知"
                current_names.append(name)
                candidates.append(((top, right, bottom, left), name))
            name_to_box = _match_detections_to_previous(candidates, smoothed_boxes)
            if not name_to_box and candidates:
                for (box, name) in candidates:
                    if name != "未知":
                        name_to_box[name] = box
        elif TRACK_USE_OPENCV_TRACKER and trackers:
            for name in list(trackers.keys()):
                tracker = trackers[name]
                if tracker is None:
                    continue
                ok, rect = tracker.update(frame)
                if ok and rect is not None and len(rect) >= 4 and rect[2] > 5 and rect[3] > 5:
                    box = _xywh_to_box(tuple(map(int, rect)))
                    smoothed_boxes[name] = _smooth_box(smoothed_boxes.get(name), box, alpha=smooth_alpha)
                    name_to_box[name] = box
                    last_seen_frame[name] = frame_index
            current_names = list(name_to_box.keys())

        if name_to_box:
            if run_detection:
                for name, raw_box in name_to_box.items():
                    smoothed_boxes[name] = _smooth_box(smoothed_boxes.get(name), raw_box, alpha=smooth_alpha)
                    last_seen_frame[name] = frame_index
                    if TRACK_USE_OPENCV_TRACKER:
                        try:
                            roi = _box_to_xywh(smoothed_boxes[name])
                            if name not in trackers:
                                new_tracker = _create_tracker()
                                if new_tracker is not None:
                                    trackers[name] = new_tracker
                            if trackers.get(name) is not None:
                                trackers[name].init(frame, roi)
                        except Exception:
                            pass

        # 穩定追蹤：若有 primary_name，優先找該人並用上一幀位置平滑
        if primary_name:
            if primary_name in name_to_box:
                raw_box = name_to_box[primary_name]
                primary_last_box = _smooth_box(primary_last_box, raw_box, alpha=smooth_alpha)
                primary_last_name = primary_name
                primary_visible += 1
            elif primary_last_box is not None and frame_index < 1000:
                # 短暫遺失：仍在上次位置畫名字（可選）
                pass
            else:
                primary_last_box = None

        # 同框與靠近統計（僅對已註冊且非「未知」的）
        registered_in_frame = [n for n in current_names if n != "未知"]
        if collect_interactions and len(registered_in_frame) >= 2:
            for i, n1 in enumerate(registered_in_frame):
                for n2 in registered_in_frame[i + 1 :]:
                    key = tuple(sorted([n1, n2]))
                    last_t = last_cooccur_time.get(key, -10.0)
                    if (time_sec - last_t) >= SAME_FRAME_COOLDOWN:
                        cooccurrence[key] += 1
                        last_cooccur_time[key] = time_sec
                    # 靠近
                    if n1 in name_to_box and n2 in name_to_box:
                        d = _distance(name_to_box[n1], name_to_box[n2])
                        if d < NEAR_DISTANCE_PX:
                            near_count[key] += 1

        # 畫 bbox 與名字（用平滑後的位置，並顯示「是誰」）
        font_scale = max(0.6, h / 500.0)
        names_to_draw = set(name_to_box.keys()) | {
            n for n, last in last_seen_frame.items()
            if frame_index - last <= TRACK_PERSISTENCE_FRAMES and smoothed_boxes.get(n)
        }
        for name in names_to_draw:
            box = smoothed_boxes.get(name)
            if box is None:
                continue
            top, right, bottom, left = box
            # 有名字 = 綠色（確定）；未知 = 橘色（未確定）
            color = (0, 255, 0) if not name.startswith("未知") else (0, 165, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            _draw_label(frame, box, name, color, font_scale=font_scale)

        out.write(frame)
        frame_index += 1
        if max_seconds is not None and time_sec >= max_seconds:
            break
        if total_frames > 0 and (frame_index % 30 == 0 or frame_index <= 1):
            pct = 100 * frame_index / total_frames
            print(f"\r處理中: {frame_index} / {total_frames} ({pct:.1f}%)", end="", flush=True)

    if total_frames > 0:
        print()
    cap.release()
    out.release()

    # 寫入互動記錄
    if collect_interactions:
        interactions = {
            "cooccurrence": {f"{k[0]},{k[1]}": v for k, v in cooccurrence.items()},
            "near_count": {f"{k[0]},{k[1]}": v for k, v in near_count.items()},
            "frame_count": frame_index,
            "fps": fps,
            "primary_visible_frames": primary_visible if primary_name else None,
            "interaction_timeline": None,
        }
        ip = interactions_file()
        os.makedirs(os.path.dirname(ip), exist_ok=True)
        with open(ip, "w", encoding="utf-8") as f:
            json.dump(interactions, f, ensure_ascii=False, indent=2)

    return {
        "frames": frame_index,
        "primary_visible_frames": primary_visible if primary_name else None,
        "output_path": output_path,
        "interactions_file": interactions_file() if collect_interactions else None,
    }
