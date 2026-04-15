"""臉部偵測與編碼抽象層：依 config.FACE_ENGINE 分派至 dlib (face_recognition) 或 InsightFace (ArcFace)。"""
from __future__ import annotations

import numpy as np

try:
    from insightface.app import FaceAnalysis

    INSIGHTFACE_AVAILABLE = True
except ImportError:
    FaceAnalysis = None  # type: ignore
    INSIGHTFACE_AVAILABLE = False

_insightface_app = None


def _current_engine() -> str:
    from config import FACE_ENGINE

    return (FACE_ENGINE or "dlib").strip().lower()


def _get_insightface():
    global _insightface_app
    if _insightface_app is not None:
        return _insightface_app
    if not INSIGHTFACE_AVAILABLE:
        raise RuntimeError(
            "InsightFace 未安裝。請執行：pip install insightface onnxruntime\n"
            "或改用 dlib 引擎（FACE_ENGINE = \"dlib\"）。"
        )
    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    _insightface_app = app
    return app


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_and_encode(rgb: np.ndarray, *, upsample: int = 1):
    """
    偵測臉並產生 embedding。
    回傳 (face_locations, face_encodings)。
    face_locations: list of (top, right, bottom, left)。
    face_encodings: list of np.ndarray。
    """
    if _current_engine() == "insightface":
        return _if_detect_and_encode(rgb)
    return _dlib_detect_and_encode(rgb, upsample=upsample)


def first_face_encoding(rgb: np.ndarray, *, prefer_cnn: bool = False):
    """註冊用：多階段偵測取第一張臉的 encoding。回傳 np.ndarray 或 None。"""
    if _current_engine() == "insightface":
        return _if_first_encoding(rgb)
    from face_registry import _first_face_encoding as _dlib_first

    return _dlib_first(rgb, prefer_cnn=prefer_cnn)


def face_distance(known_encodings, face_encoding) -> np.ndarray:
    """計算 known_encodings 各項與 face_encoding 的距離（越小越像）。"""
    if _current_engine() == "insightface":
        return _if_distance(known_encodings, face_encoding)
    import face_recognition

    return face_recognition.face_distance(known_encodings, face_encoding)


# ---------------------------------------------------------------------------
# InsightFace 實作
# ---------------------------------------------------------------------------

def _if_detect_and_encode(rgb):
    import cv2

    app = _get_insightface()
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    faces = app.get(bgr)
    faces = sorted(faces, key=lambda f: f.det_score, reverse=True)
    locations = []
    encodings = []
    for f in faces:
        x1, y1, x2, y2 = f.bbox.astype(int)
        locations.append((int(y1), int(x2), int(y2), int(x1)))
        encodings.append(f.embedding.astype(np.float64))
    return locations, encodings


def _if_first_encoding(rgb):
    import cv2

    app = _get_insightface()
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    faces = app.get(bgr)
    if not faces:
        return None
    best = max(faces, key=lambda f: f.det_score)
    return best.embedding.astype(np.float64)


def _if_distance(known_encodings, face_encoding):
    """Cosine distance = 1 - cosine_similarity。"""
    if not known_encodings or len(known_encodings) == 0:
        return np.array([])
    known = np.array([np.asarray(e, dtype=np.float64) for e in known_encodings])
    query = np.asarray(face_encoding, dtype=np.float64)
    known_norms = np.linalg.norm(known, axis=1, keepdims=True)
    known_norms[known_norms == 0] = 1.0
    known_n = known / known_norms
    q_norm = np.linalg.norm(query)
    if q_norm == 0:
        return np.ones(len(known))
    query_n = query / q_norm
    sim = known_n @ query_n
    return 1.0 - sim


# ---------------------------------------------------------------------------
# dlib 實作
# ---------------------------------------------------------------------------

def _dlib_detect_and_encode(rgb, *, upsample=1):
    import face_recognition

    locations = face_recognition.face_locations(rgb, number_of_times_to_upsample=upsample)
    encodings = face_recognition.face_encodings(rgb, locations)
    return locations, encodings
