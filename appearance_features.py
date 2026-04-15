# 臉部以外的輔助外觀特徵（色調／區域主色），用於註冊與辨識加權。
# 說明：髮型／首飾「款式」無法可靠辨識，僅能由髮區、頸部窄帶的顏色分佈做粗估；衣服以臉下方軀幹區主色與 H 直方圖表示。
from __future__ import annotations

import numpy as np
from typing import Any, Dict, Optional

try:
    import cv2
except ImportError:
    cv2 = None

SIG_VERSION = 1
_MIN_PIXELS = 80
_H_BINS = 18


def _clip_roi(img_h, img_w, x1, y1, x2, y2):
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(img_w, int(x2))
    y2 = min(img_h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _hist_h_norm(bgr_roi):
    if cv2 is None or bgr_roi is None or bgr_roi.size < _MIN_PIXELS * 3:
        return None
    hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    hist = cv2.calcHist([h], [0], None, [_H_BINS], [0, 180])
    s = float(hist.sum()) or 1.0
    hist = (hist.flatten() / s).astype(float)
    return hist.tolist()


def _mean_bgr(bgr_roi):
    if bgr_roi is None or bgr_roi.size < 12:
        return None
    m = bgr_roi.reshape(-1, 3).mean(axis=0)
    return [float(m[0]), float(m[1]), float(m[2])]


def signature_from_face_box(rgb_image: np.ndarray, face_trbl) -> Optional[Dict[str, Any]]:
    """
    依 face_recognition 臉框 (top, right, bottom, left) 切割：
    - hair：臉上方帶狀區（推估髮色／髮區色調）
    - neck_accent：臉下緣至下巴下細帶（首飾僅能做顏色粗估，無法辨識款式）
    - torso：臉下方較大區（上衣主視覺）
    若裁切過小或無 OpenCV，回傳 None。
    """
    if rgb_image is None or rgb_image.size == 0 or cv2 is None:
        return None
    if face_trbl is None or len(face_trbl) != 4:
        return None
    top, right, bottom, left = face_trbl
    h, w = rgb_image.shape[:2]
    face_h = max(1, bottom - top)
    face_w = max(1, right - left)
    bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    # 髮區：臉框上緣之上（y2=top 不含額頭皮膚為主）
    hair = _clip_roi(
        h, w,
        left - int(0.22 * face_w),
        top - int(0.9 * face_h),
        right + int(0.22 * face_w),
        top,
    )
    # 頸／首飾帶：下巴稍下
    neck = _clip_roi(
        h, w,
        left - int(0.12 * face_w),
        bottom,
        right + int(0.12 * face_w),
        bottom + int(0.28 * face_h),
    )
    # 上衣區：臉下方延伸至畫面
    torso = _clip_roi(
        h, w,
        left - int(0.45 * face_w),
        bottom + int(0.05 * face_h),
        right + int(0.45 * face_w),
        min(h, bottom + int(2.4 * face_h)),
    )

    out = {"version": SIG_VERSION}
    for key, roi_def in (("hair", hair), ("neck_accent", neck), ("torso", torso)):
        if roi_def is None:
            out[key] = {"h_hist": None, "mean_bgr": None}
            continue
        x1, y1, x2, y2 = roi_def
        crop = bgr[y1:y2, x1:x2]
        out[key] = {
            "h_hist": _hist_h_norm(crop),
            "mean_bgr": _mean_bgr(crop),
        }

    if all(
        out[k]["h_hist"] is None and out[k]["mean_bgr"] is None
        for k in ("hair", "neck_accent", "torso")
    ):
        return None
    return out


def signature_from_person_crop(rgb_image: np.ndarray, xyxy) -> Optional[Dict[str, Any]]:
    """以整個人體框 (x1,y1,x2,y2) 切成上／中／下三帶，擷取近似髮／上身／下身主色（適合 YOLO 人框）。"""
    if rgb_image is None or rgb_image.size == 0 or cv2 is None:
        return None
    if xyxy is None or len(xyxy) != 4:
        return None
    x1, y1, x2, y2 = map(int, xyxy)
    roi = _clip_roi(rgb_image.shape[0], rgb_image.shape[1], x1, y1, x2, y2)
    if roi is None:
        return None
    px1, py1, px2, py2 = roi
    crop_rgb = rgb_image[py1:py2, px1:px2]
    if crop_rgb.size < _MIN_PIXELS * 3:
        return None
    ch, cw = crop_rgb.shape[:2]
    bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
    bands = []
    for lo, hi in ((0.0, 0.32), (0.32, 0.68), (0.68, 1.0)):
        yy1 = int(ch * lo)
        yy2 = int(ch * hi)
        if yy2 <= yy1:
            continue
        bands.append(bgr[yy1:yy2, :])

    if not bands:
        return None
    keys = ("hair", "torso", "lower")
    out = {"version": SIG_VERSION, "from": "person_crop"}
    for i, crop in enumerate(bands[:3]):
        k = keys[i] if i < len(keys) else f"band_{i}"
        out[k] = {
            "h_hist": _hist_h_norm(crop),
            "mean_bgr": _mean_bgr(crop),
        }
    return out


def _hist_similarity(h1, h2):
    if not h1 or not h2 or len(h1) != len(h2):
        return 0.0
    a, b = np.array(h1, dtype=float), np.array(h2, dtype=float)
    return float(np.minimum(a, b).sum())


def _bgr_similarity(m1, m2):
    if not m1 or not m2:
        return 0.0
    a, b = np.array(m1, dtype=float), np.array(m2, dtype=float)
    d = np.linalg.norm(a - b)
    # BGR 距離粗略壓到 0~1：歐幾里得距離 0 → 1，>120 → 0
    return max(0.0, 1.0 - d / 120.0)


def _region_score(r_live, r_tpl, hist_w=0.55, bgr_w=0.45):
    if not r_live or not r_tpl:
        return 0.0
    hs = _hist_similarity(r_live.get("h_hist"), r_tpl.get("h_hist"))
    bs = _bgr_similarity(r_live.get("mean_bgr"), r_tpl.get("mean_bgr"))
    return hist_w * hs + bgr_w * bs


def appearance_similarity(live: Optional[Dict], template: Optional[Dict]) -> float:
    """0～1，愈高愈像；任一為 None 則 0。"""
    if not live or not template:
        return 0.0
    if live.get("from") == "person_crop" or template.get("from") == "person_crop":
        keys = ("hair", "torso", "lower")
        wts = (0.35, 0.45, 0.20)
    else:
        keys = ("hair", "neck_accent", "torso")
        wts = (0.35, 0.15, 0.50)

    parts = []
    ww = []
    for k, wt in zip(keys, wts):
        if k not in live or k not in template:
            continue
        s = _region_score(live.get(k), template.get(k))
        parts.append(s)
        ww.append(wt)
    if not parts:
        return 0.0
    ww = np.array(ww, dtype=float)
    ww = ww / ww.sum()
    return float(np.dot(parts, ww))


def signature_to_jsonable(sig: Optional[Dict]):
    """確保可 json.dump（已是純 list／dict）。"""
    return sig
