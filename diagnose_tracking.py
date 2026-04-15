# 追蹤診斷：對輸入影片取一幀，檢查 YOLO 人框、臉部偵測、名字匹配狀況
# 用法: python diagnose_tracking.py --video data/input/test2.mp4 [--frame 30] [--save debug.png]
import os
import sys
import argparse

# 與 main 相同，避免 OpenMP 衝突
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import numpy as np
import face_recognition

from config import FACE_MATCH_TOLERANCE, USE_APPEARANCE_AUXILIARY, set_active_scheme
from face_registry import get_registry_bundle, match_face_encoding, min_distance_per_person
from appearance_features import signature_from_face_box


def run(video_path: str, frame_index: int = 30, save_path: str = None):
    if not os.path.isfile(video_path):
        print(f"錯誤：找不到影片 {video_path}")
        return
    names, known_encodings, known_appearances = get_registry_bundle()
    if not names:
        print("錯誤：尚未註冊任何人臉，請先執行 register 或 register_all.py")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"錯誤：無法開啟影片 {video_path}")
        return
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"錯誤：無法讀取第 {frame_index} 幀")
        return

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 1) YOLO 人體偵測
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")
        results = model(frame, classes=[0], verbose=False)
        n_persons = 0
        if results and len(results) > 0 and results[0].boxes is not None:
            n_persons = len(results[0].boxes)
        print(f"[YOLO] 偵測到人體框數: {n_persons}")
    except Exception as e:
        print(f"[YOLO] 無法執行: {e}")
        n_persons = 0

    # 2) 臉部偵測（預設 + 加強小臉 upsample=2 / upsample=3，與 yolo_tracker 一致）
    boxes_default = face_recognition.face_locations(rgb)
    boxes_upsample2 = face_recognition.face_locations(rgb, number_of_times_to_upsample=2)
    boxes_upsample3 = face_recognition.face_locations(rgb, number_of_times_to_upsample=3)
    print(f"[臉部] 預設偵測到臉數: {len(boxes_default)}")
    print(f"[臉部] 加強小臉(upsample=2) 偵測到臉數: {len(boxes_upsample2)}")
    print(f"[臉部] 加強小臉(upsample=3) 偵測到臉數: {len(boxes_upsample3)}")

    # 取偵測數最多的結果用於匹配與建議
    boxes = max([boxes_default, boxes_upsample2, boxes_upsample3], key=len)

    # 3) 與註冊名單匹配（用 face_distance 看距離）
    encodings = face_recognition.face_encodings(rgb, boxes)
    matched = 0
    for box, enc in zip(boxes, encodings):
        sig = signature_from_face_box(rgb, box) if USE_APPEARANCE_AUXILIARY else None
        mname, best_dist, margin = match_face_encoding(
            names,
            known_encodings,
            enc,
            FACE_MATCH_TOLERANCE,
            query_appearance=sig,
            known_appearances=known_appearances,
        )
        if mname is not None:
            matched += 1
        distances_list = []
        for ke in known_encodings:
            if ke is None:
                distances_list.append(1.0)
            else:
                distances_list.append(float(face_recognition.face_distance([ke], enc)[0]))
        distances = np.array(distances_list, dtype=float)
        if USE_APPEARANCE_AUXILIARY and sig is not None and known_appearances:
            from appearance_features import appearance_similarity
            from config import APPEARANCE_FACE_DISTANCE_BONUS

            d = np.array(distances, dtype=float)
            for i in range(len(d)):
                if i < len(known_appearances) and known_appearances[i]:
                    sim = appearance_similarity(sig, known_appearances[i])
                    d[i] = max(0.0, d[i] - APPEARANCE_FACE_DISTANCE_BONUS * sim)
            distances = d
        by_person = min_distance_per_person(names, distances)
        parts = [f"{n}:{by_person[n]:.3f}" for n in sorted(by_person.keys())]
        if mname is not None:
            print(
                f"  臉→每人最小距離: {', '.join(parts)} | 門檻內最佳: {mname} dist={best_dist:.3f} margin={margin:.3f}"
            )
        else:
            print(
                f"  臉→每人最小距離: {', '.join(parts)} | 門檻內無匹配 (門檻={FACE_MATCH_TOLERANCE})"
            )
    print(f"[匹配] 符合門檻的臉數: {matched} / {len(encodings)}")

    # 4) 建議
    print("\n--- 建議 ---")
    if n_persons == 0:
        print("- YOLO 未偵測到人：請確認影片中有人、且畫面清晰。")
    if n_persons > 0 and len(boxes) < n_persons:
        print(f"- YOLO 偵測到 {n_persons} 人但只偵測到 {len(boxes)} 張臉：")
        print("  可試 upsample=3（本診斷已使用）、或換其他幀（--frame）；側臉、遮擋、表情大時易漏偵測。")
    if len(boxes) == 0:
        print("- 未偵測到臉：可嘗試 number_of_times_to_upsample=3 或檢查該幀是否臉部可見。")
    if len(boxes) > 0 and matched == 0:
        print("- 有臉但都未匹配：建議將 config.py 的 FACE_MATCH_TOLERANCE 調大（例如 0.65）。")
        print("  或確認註冊照與影片中的人臉角度、光線接近。")

    if save_path:
        out = frame.copy()
        for (top, right, bottom, left) in boxes:
            cv2.rectangle(out, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.imwrite(save_path, out)
        print(f"\n已儲存偵測結果圖: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="追蹤診斷：檢查一幀的 YOLO、臉部、匹配")
    parser.add_argument("--video", "-v", default="data/input/test2.mp4", help="輸入影片路徑")
    parser.add_argument("--frame", "-f", type=int, default=30, help="檢查第幾幀")
    parser.add_argument("--save", "-s", help="儲存偵測結果圖路徑")
    parser.add_argument(
        "--scheme",
        default=None,
        metavar="名稱",
        help="使用該方案的 face_registry（與 main register/process 相同名稱）",
    )
    args = parser.parse_args()
    if args.scheme:
        try:
            set_active_scheme(str(args.scheme).strip())
        except ValueError as e:
            print(f"方案名稱無效：{e}")
            sys.exit(1)
    run(args.video, args.frame, args.save)


if __name__ == "__main__":
    main()
