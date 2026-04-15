# 臉部註冊：從一張照片 + 名字建立可追蹤的幼兒資料
# 支援「一人多模版」：同一名字可多次 register，比對時取該人所有模版距離的最小值
import os
import json
import shutil
import uuid
import face_recognition
import numpy as np
from PIL import Image

from config import (
    ensure_dirs,
    active_scheme,
    registered_dir,
    registry_file,
    MAX_TEMPLATES_PER_PERSON,
    USE_APPEARANCE_AUXILIARY,
    APPEARANCE_FACE_DISTANCE_BONUS,
    REGISTER_WITHOUT_FACE_USE_APPEARANCE_ONLY,
    APPEARANCE_ONLY_VS_FACE_TEMPLATE_WEIGHT,
    APPEARANCE_FACE_QUERY_VS_APP_ONLY_TEMPLATE,
    FACE_MATCH_MIN_MARGIN,
)
from appearance_features import signature_from_face_box, signature_from_person_crop
import face_engine as _fe


def _first_face_encoding(rgb_img: np.ndarray, prefer_cnn: bool = False):
    """
    多階段偵測臉並回傳第一個 encoding（多人同框時取偵測順序第一個）。
    小臉、側臉、遠景時預設 HOG 常失敗，會依序嘗試：upsample、cnn、放大 2 倍再偵測。
    """
    if rgb_img is None or rgb_img.size == 0:
        return None

    def enc_from_locs(img, locs):
        if not locs:
            return None
        encs = face_recognition.face_encodings(img, known_face_locations=locs)
        return encs[0] if encs else None

    stages = []
    if prefer_cnn:
        stages.append(("cnn_1", lambda im: enc_from_locs(im, face_recognition.face_locations(im, number_of_times_to_upsample=1, model="cnn"))))
    stages += [
        ("hog_default", lambda im: face_recognition.face_encodings(im)),
        ("hog_up1", lambda im: enc_from_locs(im, face_recognition.face_locations(im, number_of_times_to_upsample=1, model="hog"))),
        ("hog_up2", lambda im: enc_from_locs(im, face_recognition.face_locations(im, number_of_times_to_upsample=2, model="hog"))),
    ]
    if not prefer_cnn:
        stages.append(
            (
                "cnn_1",
                lambda im: enc_from_locs(im, face_recognition.face_locations(im, number_of_times_to_upsample=1, model="cnn")),
            )
        )

    for _name, fn in stages:
        try:
            out = fn(rgb_img)
            if isinstance(out, list) and out:
                return out[0]
            if out is not None and hasattr(out, "shape"):
                return out
        except Exception:
            continue

    # 圖太小時放大再試 HOG（不改變原檔，僅記憶體內運算）
    h, w = rgb_img.shape[:2]
    if max(h, w) < 1400:
        try:
            pil = Image.fromarray(rgb_img)
            scaled = np.array(pil.resize((w * 2, h * 2), Image.Resampling.LANCZOS))
            for ups in (1, 2):
                locs = face_recognition.face_locations(scaled, number_of_times_to_upsample=ups, model="hog")
                e = enc_from_locs(scaled, locs)
                if e is not None:
                    return e
            locs = face_recognition.face_locations(scaled, number_of_times_to_upsample=1, model="cnn")
            e = enc_from_locs(scaled, locs)
            if e is not None:
                return e
        except Exception:
            pass

    return None


def load_registry():
    ensure_dirs()
    rf = registry_file()
    if os.path.isfile(rf):
        with open(rf, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {"names": [], "encodings": [], "photo_paths": []}
    saved_engine = data.get("engine", "dlib")
    cur_engine = _fe._current_engine()
    if saved_engine != cur_engine and data.get("names"):
        print(
            f"[警告] 註冊庫是用 {saved_engine} 引擎建的，但目前設定為 {cur_engine}。"
            " embedding 不相容，請重新 register！"
        )
    for key in ("names", "encodings", "photo_paths"):
        if key not in data:
            data[key] = []
    n = len(data["encodings"])
    apps = data.get("appearances")
    if not isinstance(apps, list):
        apps = []
    while len(apps) < n:
        apps.append(None)
    data["appearances"] = apps[:n]
    return data


def save_registry(names, encodings, photo_paths, appearances=None):
    ensure_dirs()
    # encodings 可能是 list of lists（從 JSON 載入）或混有 numpy array，統一轉成 list
    def to_list(enc):
        if hasattr(enc, "tolist"):
            return enc.tolist()
        return enc

    n = len(names)
    if appearances is None:
        appearances = [None] * n
    while len(appearances) < n:
        appearances.append(None)
    appearances = appearances[:n]

    data = {
        "engine": _fe._current_engine(),
        "names": names,
        "encodings": [None if enc is None else to_list(enc) for enc in encodings],
        "photo_paths": photo_paths,
        "appearances": appearances,
    }
    with open(registry_file(), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def register_face(
    photo_path: str,
    name: str,
    *,
    prefer_cnn: bool = False,
    allow_no_face: bool = None,
) -> bool:
    """註冊一張照片對應的名字。同一名字可多次註冊以追加模版；超過上限時刪除最舊模版再追加。
    prefer_cnn=True 時優先用 CNN 偵測（較慢、較準，適合遠景或小臉）。
    allow_no_face：None 時依 config.REGISTER_WITHOUT_FACE_USE_APPEARANCE_ONLY；True 時無臉則存整圖外觀模版（encoding 為 null）。"""
    if not os.path.isfile(photo_path):
        raise FileNotFoundError(f"找不到照片: {photo_path}")
    name = name.strip()
    if not name:
        raise ValueError("名字不可為空")

    if allow_no_face is None:
        allow_no_face = REGISTER_WITHOUT_FACE_USE_APPEARANCE_ONLY

    img = face_recognition.load_image_file(photo_path)
    encoding = _fe.first_face_encoding(img, prefer_cnn=prefer_cnn)
    appearance_only_row = False

    if encoding is None:
        if not allow_no_face:
            raise ValueError(
                "照片中未偵測到臉部。請換「更近、更亮、更正面」的裁切，或執行："
                ' python main.py register --photo "..." --name "..." --cnn'
                "；若欲改存外觀模版可加 --allow-no-face"
            )
        h, w = img.shape[:2]
        app_sig = signature_from_person_crop(img, (0, 0, w, h))
        if app_sig is None:
            raise ValueError(
                "未偵測到臉部，且無法從整張照片擷取外觀（畫面可能過小或模糊）。"
                "請換含身體／服飾較清楚的照片，或改用能偵測到臉的圖。"
            )
        appearance_only_row = True
    else:
        app_sig = None

    registry = load_registry()
    names = list(registry["names"])
    encodings_list = list(registry["encodings"])
    photo_paths = list(registry["photo_paths"])
    appearances = list(registry.get("appearances", []))
    while len(appearances) < len(names):
        appearances.append(None)

    max_t = max(1, int(MAX_TEMPLATES_PER_PERSON))
    # 該名字已達上限時，刪除最舊的一筆（同名中 index 最小者）
    while sum(1 for n in names if n == name) >= max_t:
        try:
            i = names.index(name)
        except ValueError:
            break
        del names[i]
        del encodings_list[i]
        del photo_paths[i]
        del appearances[i]

    if not appearance_only_row:
        face_for_app = None
        if USE_APPEARANCE_AUXILIARY:
            app_locs, app_encs = _fe.detect_and_encode(img)
            if app_locs:
                if len(app_encs) == 1:
                    face_for_app = app_locs[0]
                elif len(app_encs) > 1:
                    dists = _fe.face_distance(app_encs, np.array(encoding))
                    face_for_app = app_locs[int(np.argmin(dists))]
        box_sig = signature_from_face_box(img, face_for_app) if face_for_app else None
        if box_sig is None and USE_APPEARANCE_AUXILIARY:
            h, w = img.shape[:2]
            box_sig = signature_from_person_crop(img, (0, 0, w, h))
        app_sig = box_sig

    enc_storage = None if appearance_only_row else encoding.tolist()

    store_path = os.path.normpath(os.path.abspath(photo_path))
    if active_scheme():
        ensure_dirs()
        rd = registered_dir()
        os.makedirs(rd, exist_ok=True)
        ext = os.path.splitext(photo_path)[1] or ".jpg"
        if not ext.startswith("."):
            ext = "." + ext
        safe_stub = "".join(
            c if (c.isalnum() or c in "._-") else "_" for c in name
        ).strip("_")[:36] or "photo"
        dest = os.path.join(rd, f"{safe_stub}_{uuid.uuid4().hex[:10]}{ext}")
        shutil.copy2(photo_path, dest)
        store_path = os.path.normpath(os.path.abspath(dest))

    names.append(name)
    encodings_list.append(enc_storage)
    photo_paths.append(store_path)
    appearances.append(app_sig)

    save_registry(names, encodings_list, photo_paths, appearances)
    return True


def get_registry_encodings():
    """回傳 (names, encodings)，每個模版一列；names 可重複（同一名字多模版）。"""
    names, encodings, _ = get_registry_bundle()
    return names, encodings


def get_registry_bundle():
    """回傳 (names, encodings, appearances)；appearances 與 encodings 對齊，元素可為 None。
    encodings[i] 可為 None（僅外觀模版，無臉向量）。"""
    registry = load_registry()
    names = registry["names"]
    encodings = []
    for e in registry["encodings"]:
        if e is None:
            encodings.append(None)
        else:
            encodings.append(np.array(e))
    appearances = list(registry.get("appearances", []))
    while len(appearances) < len(encodings):
        appearances.append(None)
    appearances = appearances[: len(encodings)]
    return names, encodings, appearances


def get_unique_registered_names():
    """已註冊的不重複名字列表（順序依首次出現）。"""
    names, _ = get_registry_encodings()
    seen = set()
    out = []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def min_distance_per_person(known_names, distances_1d):
    """
    將「每個模版一個距離」聚合為「每個人名一個距離」＝該人所有模版距離的最小值。
    known_names 與 distances_1d 長度須相同。
    """
    by_name = {}
    for i, n in enumerate(known_names):
        d = float(distances_1d[i])
        if n not in by_name or d < by_name[n]:
            by_name[n] = d
    return by_name


def _per_template_distances(
    known_names,
    known_encodings,
    known_appearances,
    face_encoding,
    query_appearance,
):
    """每個模版一個距離（愈小愈像）。face_encoding 可為 None（改純外觀比對）。"""
    from appearance_features import appearance_similarity

    n = len(known_names)
    dists = np.ones(n, dtype=float)
    for i in range(n):
        enc_i = known_encodings[i] if i < len(known_encodings) else None
        app_i = known_appearances[i] if known_appearances and i < len(known_appearances) else None

        if face_encoding is not None:
            if enc_i is not None:
                d = float(_fe.face_distance([enc_i], face_encoding)[0])
                if (
                    USE_APPEARANCE_AUXILIARY
                    and APPEARANCE_FACE_DISTANCE_BONUS > 0
                    and query_appearance is not None
                    and app_i
                ):
                    sim = appearance_similarity(query_appearance, app_i)
                    d = max(0.0, d - APPEARANCE_FACE_DISTANCE_BONUS * sim)
                dists[i] = d
            else:
                if query_appearance is not None and app_i:
                    sim = appearance_similarity(query_appearance, app_i)
                    dists[i] = max(0.0, 1.0 - APPEARANCE_FACE_QUERY_VS_APP_ONLY_TEMPLATE * sim)
                else:
                    dists[i] = 1.0
        else:
            if query_appearance is None or not app_i:
                dists[i] = 1.0
                continue
            sim = appearance_similarity(query_appearance, app_i)
            if enc_i is not None:
                dists[i] = max(0.0, 1.0 - APPEARANCE_ONLY_VS_FACE_TEMPLATE_WEIGHT * sim)
            else:
                dists[i] = max(0.0, 1.0 - sim)
    return dists


def match_face_encoding(
    known_names,
    known_encodings,
    face_encoding,
    tolerance,
    query_appearance=None,
    known_appearances=None,
    appearance_only_mode: bool = False,
    skip_min_margin: bool = False,
):
    """
    以多模版比對：每人取 min(距離)，再選全域最佳；若最佳距離 > tolerance 則無匹配。
    face_encoding=None：改以 query_appearance 與各模版外觀比對（須有外觀；tolerance 建議用 APPEARANCE_ONLY_TOLERANCE）。
    appearance_only_mode=True：強制用外觀門檻（與 tolerance 搭配，通常給人體框補判）。
    known_encodings 可含 None（僅外觀模版）。
    """
    if not known_names or len(known_names) != len(known_encodings):
        if not known_names:
            return None, None, None
        raise ValueError("names 與 encodings 長度不一致")
    if face_encoding is None and query_appearance is None:
        return None, None, None

    if face_encoding is not None and not appearance_only_mode:
        dists = _per_template_distances(
            known_names,
            known_encodings,
            known_appearances,
            face_encoding,
            query_appearance,
        )
    else:
        dists = _per_template_distances(
            known_names,
            known_encodings,
            known_appearances,
            None,
            query_appearance,
        )

    by_name = min_distance_per_person(known_names, dists)
    if not by_name:
        return None, None, None

    ordered = sorted(by_name.items(), key=lambda x: x[1])
    best_name, best_dist = ordered[0]
    second_dist = ordered[1][1] if len(ordered) > 1 else best_dist + 1.0
    margin = second_dist - best_dist

    if best_dist > tolerance:
        return None, None, None
    if not skip_min_margin:
        min_mar = float(FACE_MATCH_MIN_MARGIN)
        if min_mar > 0.0 and float(margin) < min_mar:
            return None, None, None
    return best_name, best_dist, margin
