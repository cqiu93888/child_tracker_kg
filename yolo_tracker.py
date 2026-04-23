# YOLO 人體追蹤 + 臉部辨識：先用人體追蹤穩住 ID，再在框內辨識臉對應名字
# 做法類似 yolo_person_track：追蹤穩定、再掛上「是誰」
import os
import json
from collections import defaultdict

from runtime_hardening import apply as _runtime_hardening_apply

_runtime_hardening_apply()

import cv2
import numpy as np

# 先載入 PyTorch（YOLO），再載入 face_recognition（dlib）；順序反了在部分 Windows 環境易無聲閃退
try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except ImportError:
    YOLO = None  # type: ignore
    YOLO_AVAILABLE = False

import face_recognition

from cv_video_writer import create_cv_video_writer
from config import (
    ensure_dirs,
    output_dir,
    interactions_file,
    FACE_MATCH_TOLERANCE,
    YOLO_FACE_TOLERANCE_CAP,
    YOLO_FACE_TOLERANCE_EXTRA,
    FACE_PERSON_MIN_IOU_WHEN_CENTER_INSIDE,
    FACE_PERSON_MIN_IOU_WHEN_CENTER_OUTSIDE,
    FIRST_ASSIGN_MAX_DISTANCE,
    GRAPH_CONFIRMED_MAX_DISTANCE,
    GRAPH_TIMELINE_BIN_SEC,
    CORRECTION_INTERVAL_FRAMES,
    CORRECTION_MAX_DISTANCE,
    FIXED_NAME_PER_TRACK,
    STRONG_MATCH_OVERRIDE_DISTANCE,
    STRONG_OVERRIDE_MIN_IMPROVEMENT,
    STRONG_OVERRIDE_CONSECUTIVE_HITS,
    STRONG_OVERRIDE_COOLDOWN_FRAMES,
    STRONG_OVERRIDE_RESPECT_NEVER_SWITCH,
    FIRST_ASSIGN_MIN_MARGIN,
    STRONG_OVERRIDE_MIN_MARGIN,
    FIRST_ASSIGN_CONSECUTIVE_HITS,
    MIN_MARGIN_BEFORE_ASSIGN,
    NEVER_SWITCH_NAME,
    NAME_STICKY_THRESHOLD,
    NEAR_DISTANCE_PX,
    SAME_FRAME_COOLDOWN,
    TRACK_PERSISTENCE_FRAMES,
    TRACK_IOU_MIN,
    TRACK_AREA_MIN,
    TRACK_AREA_MAX,
    TRACK_CENTER_MAX,
    TRACK_IOU_FALLBACK,
    YOLO_PERSON_MODEL_SIZE,
    YOLO_IMGSZ,
    YOLO_CONF,
    YOLO_IOU,
    YOLO_MAX_DET,
    YOLO_TRACKER,
    YOLO_MIN_PERSON_AREA_FRAC,
    YOLO_FACE_UPSAMPLE_PRIMARY,
    YOLO_FACE_UPSAMPLE_FALLBACK,
    YOLO_FACE_UPSAMPLE_FALLBACK_MAX_PERSONS,
    TRACK_FINAL_NAME_RETROSPECTIVE_INTERACTIONS,
    NAME_SCORE_DECAY_PER_FRAME,
    NAME_SCORE_INCREMENT_SCALE,
    NAME_MIN_SCORE_TO_SHOW,
    NAME_SWITCH_SCORE_DELTA,
    NAME_SWITCH_COOLDOWN_FRAMES,
    NAME_SWITCH_TOP2_DELTA,
    YOLO_DEVICE,
    YOLO_TORCH_INTRAOP_THREADS,
    REQUIRE_FACE_CENTER_IN_PERSON_TO_ASSIGN,
    CONFIRMED_TRACK_NAME_CARRY_FRAMES,
)
from face_registry import get_registry_bundle, match_face_encoding
import face_engine as _fe
from draw_text_cn import draw_label_cn
from runtime_hardening import limit_torch_threads, torch_mitigations_for_mixed_dlib

if YOLO_AVAILABLE:
    try:
        limit_torch_threads(YOLO_TORCH_INTRAOP_THREADS)
        torch_mitigations_for_mixed_dlib()
    except Exception:
        pass

try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def _assign_curr_to_prev_global(curr_xyxys, prev_boxes, next_display_id):
    """
    用全域最佳指派（Hungarian）將當前框一對一對應到上一幀，最大化總 IoU，減少兩人交叉時 ID 互換。
    回傳 (curr_boxes, next_display_id)；若 prev_boxes 為空或未安裝 scipy 則回傳 (None, next_display_id)。
    """
    M = len(prev_boxes)
    if M == 0:
        n = len(curr_xyxys)
        return [(next_display_id + i, xyxy) for i, xyxy in enumerate(curr_xyxys)], next_display_id + n
    if not SCIPY_AVAILABLE:
        return None, next_display_id

    N = len(curr_xyxys)
    if N == 0:
        return [], next_display_id
    cost = np.full((N, M), 1e9)
    for i in range(N):
        for j in range(M):
            iou = _iou_xyxy(curr_xyxys[i], prev_boxes[j][1])
            if iou >= TRACK_IOU_MIN:
                cost[i, j] = 1.0 - iou
    row_ind, col_ind = linear_sum_assignment(cost)
    our_ids = [None] * N
    assigned_prev = set()
    for k in range(len(row_ind)):
        i, j = int(row_ind[k]), int(col_ind[k])
        if cost[i, j] < 1e8:
            our_ids[i] = prev_boxes[j][0]
            assigned_prev.add(j)
    for i in range(N):
        if our_ids[i] is not None:
            continue
        best_j = -1
        best_center = float("inf")
        xyxy = curr_xyxys[i]
        area_curr = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
        for j in range(M):
            if j in assigned_prev:
                continue
            oid, pbox = prev_boxes[j][0], prev_boxes[j][1]
            if _iou_xyxy(xyxy, pbox) < TRACK_IOU_FALLBACK:
                continue
            area_prev = (pbox[2] - pbox[0]) * (pbox[3] - pbox[1])
            if area_prev <= 0:
                continue
            ratio = area_curr / area_prev
            if ratio < TRACK_AREA_MIN or ratio > TRACK_AREA_MAX:
                continue
            d = _distance(xyxy, pbox)
            if d < best_center and d < TRACK_CENTER_MAX:
                best_center = d
                best_j = j
        if best_j >= 0:
            our_ids[i] = prev_boxes[best_j][0]
            assigned_prev.add(best_j)
    for i in range(N):
        if our_ids[i] is None:
            our_ids[i] = next_display_id
            next_display_id += 1
    curr_boxes = [(our_ids[i], curr_xyxys[i]) for i in range(N)]
    return curr_boxes, next_display_id


def _iou_xyxy(box1, box2):
    """計算兩個 xyxy bbox 的 IoU。"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (a1 + a2 - inter) if (a1 + a2 - inter) > 0 else 0.0


def _distance(box1, box2):
    c1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
    c2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)
    return np.hypot(c1[0] - c2[0], c1[1] - c2[1])


def _canonical_name_per_track_oid(frame_names_list):
    """
    每條人體軌道 oid 的「回溯用名字」：
    - 若該軌道**最後一次出現在畫面上**的那幀已有註冊名 → 用該名。
    - 若最後一幀是未知（None）→ 用**該軌道曾出現過的最後一個註冊名**（未知前最後的小孩）。
    - 從未掛上過註冊名 → 不納入（無該 oid 鍵）。
    """
    all_oids = set()
    for m in frame_names_list:
        all_oids.update(m.keys())
    canonical = {}
    for oid in all_oids:
        last_registered = None
        last_presence_name = None
        for m in frame_names_list:
            if oid not in m:
                continue
            nm = m[oid]
            last_presence_name = nm
            if nm is not None and not str(nm).startswith("未知"):
                last_registered = nm
        if last_presence_name is not None and not str(last_presence_name).startswith("未知"):
            canonical[oid] = last_presence_name
        elif last_registered is not None:
            canonical[oid] = last_registered
    return canonical


def _interaction_timeline_to_json(bin_tl, bin_sec: float) -> dict:
    """bin_tl: bin_idx -> person -> partner -> 該時間桶內同框幀數（雙向累加）。"""
    if not bin_tl:
        return None
    max_bin = max(bin_tl.keys())
    all_people = set()
    for b in bin_tl:
        all_people.update(bin_tl[b].keys())
    person_series = {}
    bs = float(bin_sec) if bin_sec and bin_sec > 0 else 1.0
    for p in sorted(all_people):
        arr = []
        for b in range(max_bin + 1):
            with_map = {k: int(v) for k, v in sorted(dict(bin_tl[b].get(p, {})).items())}
            arr.append(
                {
                    "t": round((b + 0.5) * bs, 3),
                    "unique_peer_count": len(with_map),
                    "with": with_map,
                }
            )
        person_series[p] = arr
    return {"bin_sec": bs, "unit": "time_sec", "y_label": "同框同伴人數", "person_series": person_series}


def _rebuild_interactions_retrospective_final_name(
    frame_boxes_list, frame_names_list, fps, timeline_bin_sec: float = None
):
    """
    依每條軌道 oid 的「回溯用名字」（見 _canonical_name_per_track_oid）：
    該軌道所有有框的幀都視為該生，再重算同框／靠近。
    同一幀若兩個 oid 回溯到同一人，保留面積較大的人框。
    另產出 interaction_timeline：時間桶內與各同伴同框的幀數，供關係圖 hover 折線圖。
    """
    bin_sec = (
        float(timeline_bin_sec)
        if timeline_bin_sec is not None and float(timeline_bin_sec) > 0
        else float(GRAPH_TIMELINE_BIN_SEC)
    )
    canonical = _canonical_name_per_track_oid(frame_names_list)

    cooccurrence = defaultdict(int)
    near_count = defaultdict(int)
    last_cooccur_time = {}
    bin_tl = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    for frame_index, boxes_map in enumerate(frame_boxes_list):
        time_sec = frame_index / fps if fps else 0.0
        bin_idx = int(time_sec / bin_sec) if bin_sec > 0 else 0
        # oid -> (xyxy, area) 再解決同人兩框
        best = {}
        for oid, xyxy in boxes_map.items():
            cn = canonical.get(oid)
            if cn is None:
                continue
            x1, y1, x2, y2 = xyxy
            area = max(0.0, float(x2 - x1) * float(y2 - y1))
            if cn not in best or area > best[cn][1]:
                best[cn] = (xyxy, area)
        name_to_box = {k: v[0] for k, v in best.items()}
        registered = sorted(name_to_box.keys())
        if len(registered) < 2:
            continue
        for i, n1 in enumerate(registered):
            for n2 in registered[i + 1 :]:
                key = tuple(sorted([n1, n2]))
                last_t = last_cooccur_time.get(key, -10.0)
                if (time_sec - last_t) >= SAME_FRAME_COOLDOWN:
                    cooccurrence[key] += 1
                    last_cooccur_time[key] = time_sec
                d = _distance(name_to_box[n1], name_to_box[n2])
                if d < NEAR_DISTANCE_PX:
                    near_count[key] += 1
                bin_tl[bin_idx][n1][n2] += 1
                bin_tl[bin_idx][n2][n1] += 1

    timeline_payload = _interaction_timeline_to_json(bin_tl, bin_sec)
    return cooccurrence, near_count, timeline_payload


def _face_box_to_xyxy(face_box):
    """face_recognition 為 (top, right, bottom, left)，轉成 (x1,y1,x2,y2)。"""
    top, right, bottom, left = face_box
    return (left, top, right, bottom)


def _face_center_in_person(face_xyxy, person_xyxy, expand_top_ratio=0.28):
    """臉部框中心是否落在人體框內；人框往上擴多一點以涵蓋坐姿時的頭部。"""
    cx = (face_xyxy[0] + face_xyxy[2]) / 2
    cy = (face_xyxy[1] + face_xyxy[3]) / 2
    x1, y1, x2, y2 = person_xyxy
    h = y2 - y1
    y1_expanded = y1 - int(h * expand_top_ratio)
    return x1 <= cx <= x2 and y1_expanded <= cy <= y2


def _allowed_face_person_pair(face_xyxy, person_xyxy):
    """臉—人體框是否允許配對。
    臉中心在人體框內時不檢查 IoU：小臉對全身框的 IoU 天生極小，舊邏輯會導致全部無法配對→整片未知。
    僅當臉中心不在該人框內時，才用較高 IoU 排除乱配。
    """
    if _face_center_in_person(face_xyxy, person_xyxy):
        if FACE_PERSON_MIN_IOU_WHEN_CENTER_INSIDE <= 0:
            return True
        iou = _iou_xyxy(face_xyxy, person_xyxy)
        return iou >= FACE_PERSON_MIN_IOU_WHEN_CENTER_INSIDE
    iou = _iou_xyxy(face_xyxy, person_xyxy)
    return iou >= FACE_PERSON_MIN_IOU_WHEN_CENTER_OUTSIDE


def _match_faces_to_persons(
    face_results,
    person_boxes,
    known_names,
    known_encodings,
    tolerance,
    rgb=None,
    known_appearances=None,
):
    """
    辨識臉 -> 取距離 < 門檻的匹配 -> 以「臉為中心」做一對一對應到人框（先算每臉最佳人框，再依分數排序指派，減少左右對調）。
    回傳 (id_to_name, id_to_best_distance, id_to_best_margin)。
    rgb + known_appearances：可選外觀輔助（髮／頸／上衣色調），加權見 config。
    """
    face_boxes = face_results["boxes"]
    encodings = face_results["encodings"]
    face_list = []
    for idx, enc in enumerate(encodings):
        box = face_boxes[idx]
        sig = None
        if rgb is not None and known_appearances is not None:
            from config import USE_APPEARANCE_AUXILIARY
            from appearance_features import signature_from_face_box

            if USE_APPEARANCE_AUXILIARY:
                sig = signature_from_face_box(rgb, box)
        # 一人多模版：每人取所有模版距離的最小值，再算人與人之間的 margin
        name, best_dist, margin = match_face_encoding(
            known_names,
            known_encodings,
            enc,
            tolerance,
            query_appearance=sig,
            known_appearances=known_appearances,
        )
        if name is None:
            continue
        face_xyxy = _face_box_to_xyxy(box)
        face_list.append((face_xyxy, name, best_dist, margin))

    from config import YOLO_FACE_LOOSE_FALLBACK_EXTRA, LOOSE_FALLBACK_MIN_MARGIN

    if (
        not face_list
        and encodings
        and float(YOLO_FACE_LOOSE_FALLBACK_EXTRA) > 0
    ):
        loose_tol = min(
            0.68,
            float(tolerance) + float(YOLO_FACE_LOOSE_FALLBACK_EXTRA),
        )
        min_mar = float(LOOSE_FALLBACK_MIN_MARGIN)
        for idx, enc in enumerate(encodings):
            box = face_boxes[idx]
            sig = None
            if rgb is not None and known_appearances is not None:
                from config import USE_APPEARANCE_AUXILIARY
                from appearance_features import signature_from_face_box

                if USE_APPEARANCE_AUXILIARY:
                    sig = signature_from_face_box(rgb, box)
            name, best_dist, margin = match_face_encoding(
                known_names,
                known_encodings,
                enc,
                loose_tol,
                query_appearance=sig,
                known_appearances=known_appearances,
                skip_min_margin=True,
            )
            if name is None:
                continue
            if margin is None or float(margin) < min_mar:
                continue
            face_xyxy = _face_box_to_xyxy(box)
            face_list.append((face_xyxy, name, best_dist, margin))

    if not person_boxes:
        return {}, {}, {}

    # 整帧偵測不到臉：僅依人體框外觀嘗試掛名
    if not face_list:
        from config import USE_APPEARANCE_AUXILIARY, APPEARANCE_ONLY_TOLERANCE
        from appearance_features import signature_from_person_crop

        id_to_name, id_to_best_dist, id_to_best_margin = {}, {}, {}
        if (
            rgb is not None
            and known_appearances is not None
            and USE_APPEARANCE_AUXILIARY
        ):
            for our_id, pxyxy in person_boxes:
                sig = signature_from_person_crop(rgb, pxyxy)
                if sig is None:
                    continue
                aname, adist, amargin = match_face_encoding(
                    known_names,
                    known_encodings,
                    None,
                    APPEARANCE_ONLY_TOLERANCE,
                    query_appearance=sig,
                    known_appearances=known_appearances,
                    appearance_only_mode=True,
                )
                if aname is not None:
                    id_to_name[our_id] = aname
                    id_to_best_dist[our_id] = adist
                    id_to_best_margin[our_id] = amargin
        return id_to_name, id_to_best_dist, id_to_best_margin

    # 以臉為中心：臉->人框做一對一指派
    id_to_name = {}
    id_to_best_dist = {}
    id_to_best_margin = {}

    if SCIPY_AVAILABLE:
        # Hungarian：在交疊/並排臉部時，比貪婪排序更不容易左右對調
        Nf = len(face_list)
        Np = len(person_boxes)
        cost = np.full((Nf, Np), 1e6, dtype=float)
        for fi, (fxyxy, fname, fdist, fmargin) in enumerate(face_list):
            for pi, (our_id, pxyxy) in enumerate(person_boxes):
                if not _allowed_face_person_pair(fxyxy, pxyxy):
                    continue
                if REQUIRE_FACE_CENTER_IN_PERSON_TO_ASSIGN and not _face_center_in_person(
                    fxyxy, pxyxy
                ):
                    continue
                if _face_center_in_person(fxyxy, pxyxy):
                    score = _iou_xyxy(fxyxy, pxyxy) + 1.0
                else:
                    score = _iou_xyxy(fxyxy, pxyxy)
                cost[fi, pi] = 1.0 - score

        row_ind, col_ind = linear_sum_assignment(cost)
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] < 5e5:
                our_id = person_boxes[c][0]
                fname = face_list[r][1]
                fdist = face_list[r][2]
                fmargin = face_list[r][3]
                id_to_name[our_id] = fname
                id_to_best_dist[our_id] = fdist
                id_to_best_margin[our_id] = fmargin
    else:
        candidates = []
        for fi, (fxyxy, fname, fdist, fmargin) in enumerate(face_list):
            for pi, (our_id, pxyxy) in enumerate(person_boxes):
                if not _allowed_face_person_pair(fxyxy, pxyxy):
                    continue
                if REQUIRE_FACE_CENTER_IN_PERSON_TO_ASSIGN and not _face_center_in_person(
                    fxyxy, pxyxy
                ):
                    continue
                if _face_center_in_person(fxyxy, pxyxy):
                    score = _iou_xyxy(fxyxy, pxyxy) + 1.0
                else:
                    score = _iou_xyxy(fxyxy, pxyxy)
                candidates.append((score, fi, pi, our_id, fname, fdist, fmargin))
        candidates.sort(key=lambda x: -x[0])

        used_face = [False] * len(face_list)
        used_person = [False] * len(person_boxes)
        for score, fi, pi, our_id, fname, fdist, fmargin in candidates:
            if used_face[fi] or used_person[pi]:
                continue
            used_face[fi] = True
            used_person[pi] = True
            id_to_name[our_id] = fname
            id_to_best_dist[our_id] = fdist
            id_to_best_margin[our_id] = fmargin

    from config import USE_APPEARANCE_AUXILIARY, APPEARANCE_ONLY_TOLERANCE
    from appearance_features import signature_from_person_crop

    if (
        rgb is not None
        and known_appearances is not None
        and person_boxes
        and USE_APPEARANCE_AUXILIARY
    ):
        assigned = set(id_to_name.keys())
        for our_id, pxyxy in person_boxes:
            if our_id in assigned:
                continue
            sig = signature_from_person_crop(rgb, pxyxy)
            if sig is None:
                continue
            aname, adist, amargin = match_face_encoding(
                known_names,
                known_encodings,
                None,
                APPEARANCE_ONLY_TOLERANCE,
                query_appearance=sig,
                known_appearances=known_appearances,
                appearance_only_mode=True,
            )
            if aname is not None:
                id_to_name[our_id] = aname
                id_to_best_dist[our_id] = adist
                id_to_best_margin[our_id] = amargin
    return id_to_name, id_to_best_dist, id_to_best_margin


def _one_name_one_person_per_frame(id_to_name, id_to_best_dist, id_to_best_margin):
    """
    同一幀內，每個註冊名最多只掛一個人框（取臉距離最佳者）。
    否則相似臉／口罩場景常出現「兩個真人都顯示小孩1」。
    """
    if not id_to_name:
        return id_to_name, id_to_best_dist, id_to_best_margin
    by_name = defaultdict(list)
    for oid, nm in id_to_name.items():
        by_name[nm].append((oid, float(id_to_best_dist.get(oid, 1.0))))
    for nm, lst in by_name.items():
        if len(lst) <= 1:
            continue
        lst.sort(key=lambda x: x[1])
        for oid, _dist in lst[1:]:
            id_to_name.pop(oid, None)
            id_to_best_dist.pop(oid, None)
            id_to_best_margin.pop(oid, None)
    return id_to_name, id_to_best_dist, id_to_best_margin


def process_video(
    video_path: str,
    output_path: str = None,
    primary_name: str = None,
    collect_interactions: bool = True,
    model_size: str = None,
    max_seconds: float = None,
    start_seconds: float = None,
    yolo_device=None,
) -> dict:
    if not YOLO_AVAILABLE:
        raise RuntimeError("請安裝 ultralytics：pip install ultralytics")

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

    ms = (model_size or YOLO_PERSON_MODEL_SIZE or "n").strip().lower()
    if ms not in ("n", "s", "m", "l", "x"):
        ms = "n"
    _dev = yolo_device if yolo_device is not None else YOLO_DEVICE
    model = YOLO(f"yolov8{ms}.pt")
    if _dev is not None and str(_dev).strip().lower() == "cpu":
        try:
            model.to("cpu")
        except Exception:
            pass
    track_id_to_name = {}
    track_id_last_name_frame = {}
    track_id_last_best_dist = {}  # 該軌道目前名字的「最近一次/最接近」距離（用來判斷是否明顯更像）
    track_id_last_switch_frame = {}  # 強匹配覆蓋換名冷卻
    track_id_override_candidate = {}  # oid -> (candidate_name, hits)
    track_id_unknown_candidate = {}  # oid -> (candidate_name, hits)
    track_id_unknown_candidate_last_frame = {}  # oid -> last frame_index（用來判斷是否連續命中）
    # 跨幀累積命名分數：每個軌道對每個名字累積證據，最後用分數差決定掛名/切換
    track_id_name_scores = {}  # oid -> {name: score}
    track_id_last_score_update_frame = {}  # oid -> last frame_index
    track_id_last_face_confirmed_frame = {}  # oid -> 最近一次臉部比對成功且已掛名的 frame_index
    # 自訂穩定編號：依位置對應上一幀，每人一個編號 (1,2,3,...)，避免 YOLO 回傳重複 id
    next_display_id = 1
    prev_boxes = []           # 上一幀 [(our_id, xyxy), ...]，僅用框的 IoU/位置延續 ID
    cooccurrence = defaultdict(int)
    near_count = defaultdict(int)
    last_cooccur_time = {}
    interaction_frame_boxes = []
    interaction_frame_track_names = []
    frame_index = 0
    primary_visible = 0
    recognize_every_n = 1   # 每幀都在整張畫面做臉部辨識，才能掛上名字

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        time_sec = frame_index / fps if fps else 0.0
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        min_person_area = w * h * YOLO_MIN_PERSON_AREA_FRAC
        track_kwargs = dict(
            persist=True,
            classes=[0],
            verbose=False,
            imgsz=YOLO_IMGSZ,
            conf=YOLO_CONF,
            iou=YOLO_IOU,
            max_det=YOLO_MAX_DET,
        )
        if _dev is not None and str(_dev).strip().lower() not in ("", "auto", "none"):
            track_kwargs["device"] = _dev
        # tracker 參數在舊版 ultralytics 可能不存在，做相容處理
        try:
            results = model.track(frame, tracker=YOLO_TRACKER, **track_kwargs)
        except TypeError:
            try:
                results = model.track(frame, **track_kwargs)
            except TypeError:
                tk = {k: v for k, v in track_kwargs.items() if k != "device"}
                try:
                    results = model.track(frame, tracker=YOLO_TRACKER, **tk)
                except TypeError:
                    results = model.track(frame, **tk)
        name_to_box = {}
        draw_list = []
        confirmed_names_this_frame = set()  # 本幀辨識距離 ≤ 門檻的名字，才計入知識圖譜
        snap_retro_boxes = {}
        snap_retro_names = {}

        if results and len(results) > 0:
            boxes = results[0].boxes
            # 不依賴 ultralytics 的 track id（本專案用幾何自編 our_id）；id 常為 None 時舊版會整段跳過→全片無辨識
            if boxes is not None and len(boxes) > 0:
                # 過濾過小人框（常是誤偵），減少「人框亂飄／臉掛錯人」
                sel = []
                for i in range(len(boxes)):
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    a = float((xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1]))
                    if a >= min_person_area:
                        sel.append(i)
                curr_xyxys = [boxes.xyxy[i].cpu().numpy() for i in sel]
                curr_boxes, next_display_id = _assign_curr_to_prev_global(
                    curr_xyxys, prev_boxes, next_display_id
                )
                if curr_boxes is None:
                    curr_boxes = []
                    used_prev = [False] * len(prev_boxes)
                    for i in range(len(curr_xyxys)):
                        xyxy = curr_xyxys[i]
                        our_id = None
                        best_j = -1
                        best_iou = TRACK_IOU_MIN
                        for j, (oid, pbox) in enumerate(prev_boxes):
                            if used_prev[j]:
                                continue
                            iou = _iou_xyxy(xyxy, pbox)
                            if iou > best_iou:
                                best_iou = iou
                                best_j = j
                                our_id = oid
                        if best_j < 0 and prev_boxes:
                            best_center = float("inf")
                            area_curr = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
                            for j, (oid, pbox) in enumerate(prev_boxes):
                                if used_prev[j]:
                                    continue
                                if _iou_xyxy(xyxy, pbox) < TRACK_IOU_FALLBACK:
                                    continue
                                area_prev = (pbox[2] - pbox[0]) * (pbox[3] - pbox[1])
                                if area_prev <= 0:
                                    continue
                                ratio = area_curr / area_prev
                                if ratio < TRACK_AREA_MIN or ratio > TRACK_AREA_MAX:
                                    continue
                                d = _distance(xyxy, pbox)
                                if d < best_center and d < TRACK_CENTER_MAX:
                                    best_center = d
                                    best_j = j
                                    our_id = oid
                        if best_j >= 0:
                            used_prev[best_j] = True
                        if our_id is None:
                            our_id = next_display_id
                            next_display_id += 1
                        curr_boxes.append((our_id, xyxy))
                prev_boxes = curr_boxes

                # 整張畫面做臉部辨識，再對應到人框（與註冊時同一套，較易辨識出名字）
                if frame_index % recognize_every_n == 0:
                    if _fe._current_engine() == "insightface":
                        face_boxes, face_encodings = _fe.detect_and_encode(rgb)
                    else:
                        up_pri = int(YOLO_FACE_UPSAMPLE_PRIMARY)
                        face_boxes = face_recognition.face_locations(
                            rgb, number_of_times_to_upsample=max(1, up_pri)
                        )
                        n_persons = len(curr_boxes)
                        up_fb = int(YOLO_FACE_UPSAMPLE_FALLBACK)
                        if (
                            up_fb > up_pri
                            and len(face_boxes) < n_persons
                            and n_persons <= int(YOLO_FACE_UPSAMPLE_FALLBACK_MAX_PERSONS)
                        ):
                            face_boxes_hi = face_recognition.face_locations(
                                rgb, number_of_times_to_upsample=up_fb
                            )
                            if len(face_boxes_hi) > len(face_boxes):
                                face_boxes = face_boxes_hi
                        face_encodings = face_recognition.face_encodings(rgb, face_boxes)
                    face_results = {"rgb": rgb, "boxes": face_boxes, "encodings": face_encodings}
                    tolerance = min(
                        float(YOLO_FACE_TOLERANCE_CAP),
                        float(FACE_MATCH_TOLERANCE) + float(YOLO_FACE_TOLERANCE_EXTRA),
                    )
                    # best_dist 與 best_margin（第二名比最好的差多少）
                    # 用於：未知掛名較寬鬆、強匹配覆蓋較嚴格
                    id_to_name, id_to_best_dist, id_to_best_margin = _match_faces_to_persons(
                        face_results,
                        curr_boxes,
                        names,
                        known_encodings,
                        tolerance,
                        rgb=rgb,
                        known_appearances=known_appearances,
                    )
                    id_to_name, id_to_best_dist, id_to_best_margin = (
                        _one_name_one_person_per_frame(
                            id_to_name, id_to_best_dist, id_to_best_margin
                        )
                    )
                    # Belief / Score accumulation：把單幀誤判「平均化」
                    # 用每條軌道累積不同名字的證據，最後用分數差決定是否掛名/切換。
                    frame_new_info = {}  # oid -> (new_name, new_dist, new_margin)
                    for oid, new_name in id_to_name.items():
                        new_dist = id_to_best_dist.get(oid, 1.0)
                        new_margin = id_to_best_margin.get(oid, 0.0)
                        frame_new_info[oid] = (new_name, new_dist, new_margin)

                        scores = track_id_name_scores.get(oid)
                        if scores is None:
                            scores = {}

                        last_upd = track_id_last_score_update_frame.get(oid, frame_index)
                        delta_frames = frame_index - last_upd
                        if delta_frames > 0:
                            decay = NAME_SCORE_DECAY_PER_FRAME ** delta_frames
                            for k in list(scores.keys()):
                                scores[k] *= decay

                        # 距離越小越好；margin 越大代表「最佳與第二名拉開」，證據更可信
                        conf = max(0.0, tolerance - new_dist) / max(1e-6, tolerance)
                        sep_factor = min(2.0, new_margin / max(1e-6, MIN_MARGIN_BEFORE_ASSIGN))
                        inc = conf * (1.0 + sep_factor) * NAME_SCORE_INCREMENT_SCALE
                        scores[new_name] = scores.get(new_name, 0.0) + inc

                        track_id_name_scores[oid] = scores
                        track_id_last_score_update_frame[oid] = frame_index

                    # 依累積分數決策掛名（未知才掛上；錯名才在「明顯更好」時切換）
                    for oid, (new_name, new_dist, new_margin) in frame_new_info.items():
                        scores = track_id_name_scores.get(oid, {})
                        if not scores:
                            continue

                        current = track_id_to_name.get(oid)
                        items_sorted = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                        best_name, best_score = items_sorted[0]
                        second_score = items_sorted[1][1] if len(items_sorted) > 1 else 0.0

                        curr_score = 0.0
                        if current is not None and not str(current).startswith("未知"):
                            curr_score = scores.get(current, 0.0)

                        can_switch = (frame_index - track_id_last_switch_frame.get(oid, -10**9)) >= NAME_SWITCH_COOLDOWN_FRAMES

                        # 未知 -> 顯示：同時要求 top1 與 top2 分離，避免 13/8 這種相似臉把未知立刻釘死
                        if current is None or str(current).startswith("未知"):
                            if (
                                best_score >= NAME_MIN_SCORE_TO_SHOW
                                and (best_score - second_score) >= NAME_SWITCH_TOP2_DELTA
                            ):
                                track_id_to_name[oid] = best_name
                                track_id_last_name_frame[oid] = frame_index
                                track_id_last_switch_frame[oid] = frame_index
                        else:
                            # 已知 -> 切換：要求分離 + 明顯優於目前 + 冷卻期
                            if (
                                best_name != current
                                and can_switch
                                and best_score >= NAME_MIN_SCORE_TO_SHOW
                                and (best_score - second_score) >= NAME_SWITCH_TOP2_DELTA
                                and (best_score - curr_score) >= NAME_SWITCH_SCORE_DELTA
                            ):
                                track_id_to_name[oid] = best_name
                                track_id_last_name_frame[oid] = frame_index
                                track_id_last_switch_frame[oid] = frame_index

                        # 若此軌道已確定（非未知），就延續其生命週期
                        chosen = track_id_to_name.get(oid)
                        if chosen is not None and not str(chosen).startswith("未知"):
                            track_id_last_name_frame[oid] = frame_index
                            track_id_last_face_confirmed_frame[oid] = frame_index

                        # 知識圖譜/互動：只用「最終決策後的名字」且該幀距離要高信心
                        if chosen is not None and not str(chosen).startswith("未知"):
                            if new_dist <= GRAPH_CONFIRMED_MAX_DISTANCE:
                                confirmed_names_this_frame.add(chosen)

                    # 已確認軌道延續：人框還在但這幀臉偵測失敗時，延續已有名字
                    _carry_limit = int(CONFIRMED_TRACK_NAME_CARRY_FRAMES)
                    if _carry_limit > 0:
                        for oid, _xyxy in curr_boxes:
                            if oid in frame_new_info:
                                continue
                            cname = track_id_to_name.get(oid)
                            if cname is None or str(cname).startswith("未知"):
                                continue
                            last_cf = track_id_last_face_confirmed_frame.get(oid, -1)
                            if last_cf < 0:
                                continue
                            if frame_index - last_cf <= _carry_limit:
                                track_id_last_name_frame[oid] = frame_index

                for (our_id, xyxy) in curr_boxes:
                    x1, y1, x2, y2 = xyxy
                    name = track_id_to_name.get(our_id)
                    display_name = name if name is not None else f"未知-{our_id}"
                    draw_list.append((display_name, (x1, y1, x2, y2)))
                    if name is not None:
                        name_to_box[name] = (x1, y1, x2, y2)
                        if name == primary_name:
                            primary_visible += 1

                if collect_interactions and TRACK_FINAL_NAME_RETROSPECTIVE_INTERACTIONS:
                    for oid, xyxy in curr_boxes:
                        snap_retro_boxes[oid] = tuple(float(x) for x in xyxy)
                        snap_retro_names[oid] = track_id_to_name.get(oid)

            else:
                prev_boxes = []
        else:
            prev_boxes = []

        # 互動：預設僅本幀高信心；回溯模式改於全片跑完後重算
        if collect_interactions and not TRACK_FINAL_NAME_RETROSPECTIVE_INTERACTIONS:
            registered_in_frame = [
                n for n in name_to_box
                if n != "未知" and n in confirmed_names_this_frame
            ]
            if len(registered_in_frame) >= 2:
                for i, n1 in enumerate(registered_in_frame):
                    for n2 in registered_in_frame[i + 1 :]:
                        key = tuple(sorted([n1, n2]))
                        last_t = last_cooccur_time.get(key, -10.0)
                        if (time_sec - last_t) >= SAME_FRAME_COOLDOWN:
                            cooccurrence[key] += 1
                            last_cooccur_time[key] = time_sec
                        if n1 in name_to_box and n2 in name_to_box:
                            d = _distance(name_to_box[n1], name_to_box[n2])
                            if d < NEAR_DISTANCE_PX:
                                near_count[key] += 1

        if collect_interactions and TRACK_FINAL_NAME_RETROSPECTIVE_INTERACTIONS:
            interaction_frame_boxes.append(snap_retro_boxes)
            interaction_frame_track_names.append(snap_retro_names)

        for tid in list(track_id_last_name_frame.keys()):
            if frame_index - track_id_last_name_frame[tid] > TRACK_PERSISTENCE_FRAMES * 2:
                track_id_to_name.pop(tid, None)
                track_id_last_name_frame.pop(tid, None)
                track_id_last_best_dist.pop(tid, None)
                track_id_name_scores.pop(tid, None)
                track_id_last_score_update_frame.pop(tid, None)
                track_id_override_candidate.pop(tid, None)
                track_id_unknown_candidate.pop(tid, None)
                track_id_unknown_candidate_last_frame.pop(tid, None)
                track_id_last_switch_frame.pop(tid, None)
                track_id_last_face_confirmed_frame.pop(tid, None)

        font_scale = max(0.6, h / 500.0)
        font_size = max(18, int(24 * (h / 480)))
        for display_name, (x1, y1, x2, y2) in draw_list:
            # 有名字 = 綠色（確定）；未知-N = 橘色（未確定），一眼看出誰被辨識
            color = (0, 255, 0) if not display_name.startswith("未知") else (0, 165, 255)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            draw_label_cn(frame, (x1, y1, x2, y2), display_name, color, font_size=font_size)

        out.write(frame)
        frame_index += 1
        # 只跑 N 秒：達到即停止
        if max_seconds is not None and time_sec >= max_seconds:
            break
        # 進度輸出：每 30 幀或首幀更新一次，避免刷屏
        if total_frames > 0 and (frame_index % 30 == 0 or frame_index <= 1):
            pct = 100 * frame_index / total_frames
            print(f"\r處理中: {frame_index} / {total_frames} ({pct:.1f}%)", end="", flush=True)

    if total_frames > 0:
        print()  # 換行，避免覆蓋最後一行進度
    cap.release()
    out.release()

    if collect_interactions:
        interaction_timeline = None
        if TRACK_FINAL_NAME_RETROSPECTIVE_INTERACTIONS and interaction_frame_boxes:
            cooccurrence, near_count, interaction_timeline = (
                _rebuild_interactions_retrospective_final_name(
                    interaction_frame_boxes, interaction_frame_track_names, fps
                )
            )
        interactions = {
            "cooccurrence": {f"{k[0]},{k[1]}": v for k, v in cooccurrence.items()},
            "near_count": {f"{k[0]},{k[1]}": v for k, v in near_count.items()},
            "frame_count": frame_index,
            "fps": fps,
            "primary_visible_frames": primary_visible if primary_name else None,
            "interaction_counting": (
                "retrospective_final_name_per_track"
                if TRACK_FINAL_NAME_RETROSPECTIVE_INTERACTIONS
                else "per_frame_confirmed_face_only"
            ),
            "interaction_timeline": interaction_timeline,
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
