# 幼兒辨識與知識圖譜 - 設定
from __future__ import annotations

import json
import os
import sys

# 專案根目錄
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 資料目錄
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REGISTERED_DIR = os.path.join(DATA_DIR, "registered")   # 註冊照片
# audit_registry 預設匯出「符合」註冊照的目錄（僅複製，不改 registry）
COMPLIANT_REGISTERED_EXPORT_DIR = os.path.join(
    DATA_DIR, "registered_compliant"
)
# 下列三項為「未指定方案」時的預設路徑；指定 --scheme 後請用 output_dir() / graph_dir() / interactions_file()
OUTPUT_DIR = os.path.join(DATA_DIR, "output")            # 輸出影片
GRAPH_DIR = os.path.join(DATA_DIR, "graph")              # 圖譜輸出
INTERACTIONS_FILE = os.path.join(DATA_DIR, "interactions.json")  # 互動記錄

# --- 方案（不同班級／測次）：可完全隔離（註冊 + 互動 + 輸出 + 圖譜）到 data/schemes/<方案>/ ---
# 無 --scheme 時：仍使用 data/face_registry.json、data/registered/（舊行為）。
SCHEMES_PARENT = os.path.join(DATA_DIR, "schemes")
ACTIVE_SCHEME: str | None = None


def sanitize_scheme_id(raw: str) -> str:
    s = (raw or "").strip()
    if not s:
        raise ValueError("方案名稱不可為空白")
    out: list[str] = []
    for ch in s:
        if ch in '\\/:*?"<>|':
            out.append("_")
        elif ord(ch) < 32:
            continue
        else:
            out.append(ch)
    s2 = "".join(out).strip(" .")
    if not s2:
        raise ValueError("方案名稱僅含非法字元，請改用中英數、橫線、底線等")
    return s2[:64]


def set_active_scheme(name: str | None) -> None:
    """由 main / run_yolo_isolated 設定目前方案；None 表示沿用 data/ 根目錄（舊行為）。"""
    global ACTIVE_SCHEME
    if name is None or str(name).strip() == "":
        ACTIVE_SCHEME = None
    else:
        ACTIVE_SCHEME = sanitize_scheme_id(str(name))
    _apply_scheme_config_from_disk()


def active_scheme() -> str | None:
    return ACTIVE_SCHEME


def scheme_root() -> str:
    """目前方案根目錄；無方案時為 DATA_DIR。"""
    if not ACTIVE_SCHEME:
        return DATA_DIR
    return os.path.join(SCHEMES_PARENT, ACTIVE_SCHEME)


def interactions_file() -> str:
    if not ACTIVE_SCHEME:
        return INTERACTIONS_FILE
    return os.path.join(scheme_root(), "interactions.json")


def output_dir() -> str:
    if not ACTIVE_SCHEME:
        return OUTPUT_DIR
    return os.path.join(scheme_root(), "output")


def graph_dir() -> str:
    if not ACTIVE_SCHEME:
        return GRAPH_DIR
    return os.path.join(scheme_root(), "graph")


def registry_file() -> str:
    """face_registry.json：無方案時在 data/，有方案時在 data/schemes/<id>/。"""
    if not ACTIVE_SCHEME:
        return os.path.join(DATA_DIR, "face_registry.json")
    return os.path.join(scheme_root(), "face_registry.json")


def registered_dir() -> str:
    """註冊照片建議目錄；有方案時為該方案專用子資料夾。"""
    if not ACTIVE_SCHEME:
        return REGISTERED_DIR
    return os.path.join(scheme_root(), "registered")

# 辨識場景："default" | "classroom_masked"
# classroom_masked：口罩／低頭／小臉／黃背心教室；自動放寬臉距離與累積掛名門檻、啟用小外觀加權、追蹤改 more_coverage，並在臉數不足時加強偵測（較慢）。
RECOGNITION_PRESET = "classroom_masked"

# 臉部辨識：過嚴→大量未知；過鬆→相似臉誤掛。可搭配下方「臉—人框」門檻一起調。
FACE_MATCH_TOLERANCE = 0.58

# YOLO 全畫面臉比對：實際門檻 = min(上限, FACE_MATCH_TOLERANCE + EXTRA)
YOLO_FACE_TOLERANCE_CAP = 0.63
YOLO_FACE_TOLERANCE_EXTRA = 0.06
# 主門檻全滅時再加寬一點點；太大易認錯，太小又回到整片未知
YOLO_FACE_LOOSE_FALLBACK_EXTRA = 0.05
# 寬鬆後備：僅當「最佳人名與第二佳」人級距離差 ≥ 此值才接受（压低相似臉誤掛）
LOOSE_FALLBACK_MIN_MARGIN = 0.07

# 臉中心落在某人體框內時，臉框與人體框至少 IoU；太小表示幾何上不像同一人，易掛錯（并排幼兒常見）
# 設 0＝臉中心在人框內即允許配對（小臉對全身框 IoU 極小，勿在此用高 IoU）
FACE_PERSON_MIN_IOU_WHEN_CENTER_INSIDE = 0.0
# 臉中心不在該人體框內時，僅在 IoU 夠大才允許配對（減少臉飘到鄰居身上）
FACE_PERSON_MIN_IOU_WHEN_CENTER_OUTSIDE = 0.10

# 外觀輔助：常認錯時建議關 False（衣服同色會拉錯人）；要開請維持小 BONUS
USE_APPEARANCE_AUXILIARY = False
APPEARANCE_FACE_DISTANCE_BONUS = 0.04
# 註冊照偵測不到臉時，改存「整張圖外觀」（person_crop），無臉距離可比對（較不穩，適合背影／遠景）
REGISTER_WITHOUT_FACE_USE_APPEARANCE_ONLY = False
# 影片中人框無法對到臉時，用人體框做外觀比對；能接受的「偽距離」上限（愈大愈寬鬆，約對應 1-sim）
APPEARANCE_ONLY_TOLERANCE = 0.55
# 有臉的模版與「僅外觀」模版比對時：偽距離 = 1 − 此係數×sim（略降權，因無臉模版本較不可靠）
APPEARANCE_ONLY_VS_FACE_TEMPLATE_WEIGHT = 0.72
# 畫面有臉但比對「僅外觀模版」時：偽距離 = 1 − 此係數×sim
APPEARANCE_FACE_QUERY_VS_APP_ONLY_TEMPLATE = 0.70
# 同一名字可註冊多張臉（多模版）；超過時會刪除該名字最舊的一筆再追加
MAX_TEMPLATES_PER_PERSON = 5
# 一軌道一固定名字：True = 預設一軌道一名字，僅在「強匹配覆蓋」或修正時段才允許換人
FIXED_NAME_PER_TRACK = True
# 強匹配覆蓋：當某臉辨識距離 ≤ 此值視為「明顯更像」，允許整條軌道換成該名字（即使已掛別名）
STRONG_MATCH_OVERRIDE_DISTANCE = 0.44
# 強匹配覆蓋（進階）：避免單幀雜訊或小幅改善就換名
# - 需要「比目前軌道名字的最近距離」至少改善多少，才視為明顯更像
STRONG_OVERRIDE_MIN_IMPROVEMENT = 0.05
# - 需要連續幾次（辨識幀）都滿足「強匹配覆蓋」條件才真正換名
STRONG_OVERRIDE_CONSECUTIVE_HITS = 2
# - 換名後的冷卻幀數；冷卻期間不再接受強匹配覆蓋換名，避免來回跳
STRONG_OVERRIDE_COOLDOWN_FRAMES = 30
# - 是否讓強匹配覆蓋也遵守 NEVER_SWITCH_NAME（True=一軌道一名字且不換，最穩）
# 由於你希望「平常一軌道一名字，但若明顯更像可整軌換名」，因此預設要允許強匹配覆蓋修正。
STRONG_OVERRIDE_RESPECT_NEVER_SWITCH = False
NEVER_SWITCH_NAME = True   # 一般幀不換人
NAME_STICKY_THRESHOLD = 0.44  # 僅當 NEVER_SWITCH_NAME=False 時有效
TRACK_SMOOTHING_FRAMES = 12   # 追蹤平滑幀數（愈大愈平滑、愈不抖）
TRACK_USE_OPENCV_TRACKER = True   # 使用 OpenCV 追蹤器補幀，追蹤更穩
DETECT_EVERY_N_FRAMES = 1

# 未知掛名：連續幾幀都符合條件才掛名（加一點持續性，減少單幀誤匹配）
FIRST_ASSIGN_CONSECUTIVE_HITS = 2

# 跨幀累積命名（Belief / Score Accumulation）
# 透過累積證據分數，降低單幀誤判造成「記錯」或「未知」。
NAME_SCORE_DECAY_PER_FRAME = 0.95   # 每幀衰減；若辨識間隔越長，衰減越顯著
NAME_SCORE_INCREMENT_SCALE = 1.0    # 證據累積倍率
NAME_MIN_SCORE_TO_SHOW = 0.63
NAME_SWITCH_SCORE_DELTA = 0.38
NAME_SWITCH_COOLDOWN_FRAMES = 32
NAME_SWITCH_TOP2_DELTA = 0.32

# 追蹤模式：改這裡即可切換整組參數，不用再一個一個試
# - "stable_names" = 名字穩定優先（少跳、少同一人變別人），後半段可能較多未知
# - "more_coverage" = 綠框／後半優先（多一點名字、後半較少未知），可能名字偶爾跳
# - "balanced" = 折中
# 「很多未知 + 掛上的仍常錯」：stable 易把早期錯名黏死；balanced 較能靠後續幀修正
TRACKING_MODE = "balanced"
if (RECOGNITION_PRESET or "").strip().lower() == "classroom_masked":
    TRACKING_MODE = "more_coverage"

_tracking_mode = (TRACKING_MODE or "balanced").lower() if isinstance(TRACKING_MODE, str) else "balanced"
if _tracking_mode not in ("stable_names", "more_coverage", "balanced"):
    _tracking_mode = "balanced"

if _tracking_mode == "stable_names":
    FIRST_ASSIGN_MAX_DISTANCE = 0.55
    MIN_MARGIN_BEFORE_ASSIGN = 0.06
    CORRECTION_INTERVAL_FRAMES = 90
    CORRECTION_MAX_DISTANCE = 0.44
    TRACK_PERSISTENCE_FRAMES = 70
    TRACK_IOU_MIN, TRACK_AREA_MIN, TRACK_AREA_MAX, TRACK_CENTER_MAX, TRACK_IOU_FALLBACK = 0.20, 0.60, 1.75, 160, 0.08
elif _tracking_mode == "more_coverage":
    FIRST_ASSIGN_MAX_DISTANCE = 0.58
    MIN_MARGIN_BEFORE_ASSIGN = 0.05
    CORRECTION_INTERVAL_FRAMES = 60
    CORRECTION_MAX_DISTANCE = 0.46
    TRACK_PERSISTENCE_FRAMES = 85
    TRACK_IOU_MIN, TRACK_AREA_MIN, TRACK_AREA_MAX, TRACK_CENTER_MAX, TRACK_IOU_FALLBACK = 0.17, 0.52, 1.92, 195, 0.05
else:  # balanced
    FIRST_ASSIGN_MAX_DISTANCE = 0.57
    MIN_MARGIN_BEFORE_ASSIGN = 0.055
    CORRECTION_INTERVAL_FRAMES = 75
    CORRECTION_MAX_DISTANCE = 0.45
    TRACK_PERSISTENCE_FRAMES = 78
    TRACK_IOU_MIN, TRACK_AREA_MIN, TRACK_AREA_MAX, TRACK_CENTER_MAX, TRACK_IOU_FALLBACK = 0.18, 0.55, 1.85, 180, 0.06

# 未知掛名與強匹配覆蓋的 margin
FIRST_ASSIGN_MIN_MARGIN = MIN_MARGIN_BEFORE_ASSIGN
STRONG_OVERRIDE_MIN_MARGIN = MIN_MARGIN_BEFORE_ASSIGN
# 第一次掛名距離上限：使用 tracking mode 預設值，不額外夾斷

# 知識圖譜：僅高信心同框才計入（classroom_masked 會於下方略收紧）
GRAPH_CONFIRMED_MAX_DISTANCE = 0.50

# --- classroom_masked：在追蹤模式已定後，再放寬「比對／掛名」與外觀（圖譜仍略嚴）---
_rec_preset = (RECOGNITION_PRESET or "default").strip().lower()
if _rec_preset == "classroom_masked":
    FACE_MATCH_TOLERANCE = 0.60
    YOLO_FACE_TOLERANCE_CAP = 0.66
    YOLO_FACE_TOLERANCE_EXTRA = 0.07
    LOOSE_FALLBACK_MIN_MARGIN = 0.048
    NAME_MIN_SCORE_TO_SHOW = 0.50
    NAME_SWITCH_TOP2_DELTA = 0.20
    NAME_SWITCH_SCORE_DELTA = 0.28
    NAME_SCORE_INCREMENT_SCALE = 1.12
    USE_APPEARANCE_AUXILIARY = True
    APPEARANCE_FACE_DISTANCE_BONUS = 0.022
    APPEARANCE_ONLY_TOLERANCE = 0.58
    GRAPH_CONFIRMED_MAX_DISTANCE = 0.48

# 同框/互動判定（像素距離）
NEAR_DISTANCE_PX = 150        # 距離小於此視為「靠近」
SAME_FRAME_COOLDOWN = 0.5     # 同一對人在多少秒內不重複計一次同框

# YOLO 軌道互動統計：True＝每條軌道回溯名字規則：最後一幀若已掛名則用之；若最後是未知則用「未知前最後一次出現的註冊名」；從未掛名則不計。整段有框之幀皆歸該名。False＝僅本幀已掛名且高信心（舊行為）
# 注意：會多佔記憶體（每幀存框）；超長影片若 OOM 可改 False
TRACK_FINAL_NAME_RETROSPECTIVE_INTERACTIONS = True

# 追蹤方式（二擇一）
# 人框 ID 延續僅用「當前框與上一幀框」的 IoU/中心距離配對，不依賴 YOLO track ID
USE_YOLO_TRACKER = True   # True=YOLO 人體追蹤+臉部辨識（較穩），False=僅臉部偵測+平滑

# --- YOLO「人體框」偵測／追蹤（你說「人的判斷不準」多半在這一層：框漂、漏人、誤偵）---
# 模型：n 最快、s 通常比 n 準、m 更準但更慢（會自動下載對應 .pt）
YOLO_PERSON_MODEL_SIZE = "s"
# 推論解析度：640 常漏遠處小人；960～1280 較準但較慢
YOLO_IMGSZ = 640
YOLO_CONF = 0.28          # 偵測信心；太低易誤框，太高易漏人
YOLO_IOU = 0.50           # NMS IoU
YOLO_MAX_DET = 50         # 一幀最多幾個人（教室多人可加大）
# ultralytics 內建追蹤設定檔；可試 "botsort.yaml"
YOLO_TRACKER = "bytetrack.yaml"
# YOLO 推論裝置：None = 自動（通常用 GPU）；設 "cpu" 可減少與 dlib 同窗時閃退（較慢）。
# 若預設閃退，可改 "cpu" 或執行 process 時加 --yolo-cpu
YOLO_DEVICE = None
# PyTorch CPU 介入執行緒數；1 較利與 numpy/dlib 共存
YOLO_TORCH_INTRAOP_THREADS = 1
# process：YOLO 一律用獨立子行程跑（避免主程式已載入 dlib 再載 PyTorch 崩潰）
YOLO_USE_SUBPROCESS_IN_PROCESS_CMD = True
# 子行程若失敗（含無聲崩潰後非 0），自動改為純臉部追蹤以產生輸出；設 False 則直接報錯結束
YOLO_AUTO_FALLBACK_TO_FACE = True
# 過小框視為雜訊（面積占整張畫面比例下限）
YOLO_MIN_PERSON_AREA_FRAC = 0.0008
# 全畫面臉偵測：主階段 upsample；若臉數 < 當幀人數且人數 ≤ 上限，可再試更高 upsample（慢）
YOLO_FACE_UPSAMPLE_PRIMARY = 3
YOLO_FACE_UPSAMPLE_FALLBACK = 0
YOLO_FACE_UPSAMPLE_FALLBACK_MAX_PERSONS = 8
if _rec_preset == "classroom_masked":
    # 第二階 upsample 極耗 CPU，與 PyTorch 同窗時易觸發原生崩潰；改為 0，寧可稍漏臉
    YOLO_FACE_UPSAMPLE_FALLBACK = 0

# 圖譜（GRAPH_CONFIRMED_MAX_DISTANCE 見上方）
MIN_COOCCURRENCE_FOR_FRIEND = 3   # 至少同框幾次才畫「朋友」邊
PERSONALITY_WEIGHT = 0.3          # 個性在關係圖中的權重

# 以「每位幼兒為中心」的子圖（ego graph）：只顯示與該幼兒有達標邊的同伴
# 建議：EGO ≥ 全班門檻（預設與 MIN_COOCCURRENCE_FOR_FRIEND 相同）；資料吵可改 4～5
EGO_GRAPH_MIN_COOCCURRENCE = 3    # 與中心同框至少幾次才在「個人圖」出現該同伴
EGO_GRAPH_MIN_EDGE_WEIGHT = 0.0   # 綜合權重（同框+靠近）下限，0 表示不另設
EGO_GRAPH_MAX_NEIGHBORS = None    # 最多顯示幾位同伴；None = 不限，只受同框門檻篩選
EGO_GRAPH_SUBDIR = "ego"          # 個人圖 HTML 輸出於 data/graph/ego/

# 個性推論（personality.py）：以「全班互動指標」相對分位推標籤（代理指標，非問卷／生理）
# 靈感類似發展研究將氣質分為約 15% 高/低極端與中間多數（實際仍依你這批孩子的分佈）
PERSONALITY_EXTREME_GROUP_FRAC = 0.15   # 整合度最低/最高約各幾成的人標成偏內向/偏活躍
PERSONALITY_MIN_PEOPLE = 4              # 班上至少幾位有互動資料才用分位法；否則用簡化規則
PERSONALITY_NEAR_CO_RATIO_PCT = 70      # 靠近/同框 比率高於全班此百分位 → 易標「親密型」（身體距離近）
PERSONALITY_SOCIAL_PARTNER_PCT = 70     # 互動對象數高於此百分位 → 易標「社交型」（對象多元）

# 互動時間軸：回溯模式下寫入 interactions.json，關係圖 hover 節點時以折線圖顯示（秒為單位）
GRAPH_TIMELINE_BIN_SEC = 1.0

# 關係圖版面（pyvis / vis.js）：太壅擠時可把 SPRING_LENGTH 加大、REPULSION 加大（更負）
RELATIONSHIP_GRAPH_HEIGHT = "780px"   # 圖表高度
RELATIONSHIP_GRAPH_NODE_SIZE = 30   # 節點圓點大小（數值愈大圓愈大；可再微調 28～36）
GRAPH_PHYSICS_SPRING_LENGTH = 280     # 邊的理想長度（愈大節點愈散）
GRAPH_PHYSICS_REPULSION = -120        # forceAtlas2 斥力（愈負愈散開，約 -50～-200）
GRAPH_PHYSICS_CENTRAL_GRAVITY = 0.004 # 往中心拉的力（愈小愈不團成一坨）
GRAPH_PHYSICS_STABILIZATION_ITERS = 320  # 穩定化迭代，太少會還擠在一起

# --- 可選：提高掛名準確度（僅當 scheme_config.json 或此處設非預設時才與舊版不同）---
# 最佳人名與次佳人名之臉距離差（人級 min，見 face_registry.margin）至少此值才接受比對，否則視為太像、不掛名。0＝不檢查（預設，相容舊行為）。
FACE_MATCH_MIN_MARGIN = 0.0
# True：YOLO 路徑下臉—人配對僅在「臉中心落在該人體框內」時才成立；並排時較不易把臉掛到鄰居身上（可能略增未知）。False＝維持原本 IoU 規則。
REQUIRE_FACE_CENTER_IN_PERSON_TO_ASSIGN = False
# 已確認軌道在臉偵測失敗時延續名字的最大幀數。0＝關閉（舊行為：臉失敗就不再延續）。
# 例如設 150 且影片 25fps，相當於臉被遮住最多 6 秒仍保持原名。
# 僅在該軌道仍被 YOLO 持續追蹤（人框還在）時才延續，人框消失則不延續。
CONFIRMED_TRACK_NAME_CARRY_FRAMES = 0
# 臉部辨識引擎："dlib"（預設，face_recognition 套件）或 "insightface"（ArcFace，口罩／遮擋場景更準）。
# 切換引擎後須重新 register（embedding 格式不相容）。
FACE_ENGINE = "dlib"

# --- 方案專用參數（存在 data/schemes/<id>/scheme_config.json，JSON 物件鍵 = 下方 ALLOWLIST 中的變數名）---
SCHEME_CONFIG_FILENAME = "scheme_config.json"

SCHEME_CONFIG_ALLOWLIST = frozenset(
    {
        "RECOGNITION_PRESET",
        "TRACKING_MODE",
        "FACE_MATCH_TOLERANCE",
        "YOLO_FACE_TOLERANCE_CAP",
        "YOLO_FACE_TOLERANCE_EXTRA",
        "YOLO_FACE_LOOSE_FALLBACK_EXTRA",
        "LOOSE_FALLBACK_MIN_MARGIN",
        "FACE_PERSON_MIN_IOU_WHEN_CENTER_INSIDE",
        "FACE_PERSON_MIN_IOU_WHEN_CENTER_OUTSIDE",
        "FACE_MATCH_MIN_MARGIN",
        "REQUIRE_FACE_CENTER_IN_PERSON_TO_ASSIGN",
        "CONFIRMED_TRACK_NAME_CARRY_FRAMES",
        "FACE_ENGINE",
        "USE_APPEARANCE_AUXILIARY",
        "APPEARANCE_FACE_DISTANCE_BONUS",
        "REGISTER_WITHOUT_FACE_USE_APPEARANCE_ONLY",
        "APPEARANCE_ONLY_TOLERANCE",
        "APPEARANCE_ONLY_VS_FACE_TEMPLATE_WEIGHT",
        "APPEARANCE_FACE_QUERY_VS_APP_ONLY_TEMPLATE",
        "MAX_TEMPLATES_PER_PERSON",
        "FIXED_NAME_PER_TRACK",
        "STRONG_MATCH_OVERRIDE_DISTANCE",
        "STRONG_OVERRIDE_MIN_IMPROVEMENT",
        "STRONG_OVERRIDE_CONSECUTIVE_HITS",
        "STRONG_OVERRIDE_COOLDOWN_FRAMES",
        "STRONG_OVERRIDE_RESPECT_NEVER_SWITCH",
        "NEVER_SWITCH_NAME",
        "NAME_STICKY_THRESHOLD",
        "TRACK_SMOOTHING_FRAMES",
        "TRACK_USE_OPENCV_TRACKER",
        "DETECT_EVERY_N_FRAMES",
        "FIRST_ASSIGN_CONSECUTIVE_HITS",
        "NAME_SCORE_DECAY_PER_FRAME",
        "NAME_SCORE_INCREMENT_SCALE",
        "NAME_MIN_SCORE_TO_SHOW",
        "NAME_SWITCH_SCORE_DELTA",
        "NAME_SWITCH_COOLDOWN_FRAMES",
        "NAME_SWITCH_TOP2_DELTA",
        "FIRST_ASSIGN_MAX_DISTANCE",
        "MIN_MARGIN_BEFORE_ASSIGN",
        "CORRECTION_INTERVAL_FRAMES",
        "CORRECTION_MAX_DISTANCE",
        "TRACK_PERSISTENCE_FRAMES",
        "TRACK_IOU_MIN",
        "TRACK_AREA_MIN",
        "TRACK_AREA_MAX",
        "TRACK_CENTER_MAX",
        "TRACK_IOU_FALLBACK",
        "FIRST_ASSIGN_MIN_MARGIN",
        "STRONG_OVERRIDE_MIN_MARGIN",
        "GRAPH_CONFIRMED_MAX_DISTANCE",
        "NEAR_DISTANCE_PX",
        "SAME_FRAME_COOLDOWN",
        "TRACK_FINAL_NAME_RETROSPECTIVE_INTERACTIONS",
        "USE_YOLO_TRACKER",
        "YOLO_PERSON_MODEL_SIZE",
        "YOLO_IMGSZ",
        "YOLO_CONF",
        "YOLO_IOU",
        "YOLO_MAX_DET",
        "YOLO_TRACKER",
        "YOLO_DEVICE",
        "YOLO_TORCH_INTRAOP_THREADS",
        "YOLO_USE_SUBPROCESS_IN_PROCESS_CMD",
        "YOLO_AUTO_FALLBACK_TO_FACE",
        "YOLO_MIN_PERSON_AREA_FRAC",
        "YOLO_FACE_UPSAMPLE_PRIMARY",
        "YOLO_FACE_UPSAMPLE_FALLBACK",
        "YOLO_FACE_UPSAMPLE_FALLBACK_MAX_PERSONS",
        "MIN_COOCCURRENCE_FOR_FRIEND",
        "PERSONALITY_WEIGHT",
        "EGO_GRAPH_MIN_COOCCURRENCE",
        "EGO_GRAPH_MIN_EDGE_WEIGHT",
        "EGO_GRAPH_MAX_NEIGHBORS",
        "PERSONALITY_EXTREME_GROUP_FRAC",
        "PERSONALITY_MIN_PEOPLE",
        "PERSONALITY_NEAR_CO_RATIO_PCT",
        "PERSONALITY_SOCIAL_PARTNER_PCT",
        "GRAPH_TIMELINE_BIN_SEC",
        "RELATIONSHIP_GRAPH_HEIGHT",
        "RELATIONSHIP_GRAPH_NODE_SIZE",
        "GRAPH_PHYSICS_SPRING_LENGTH",
        "GRAPH_PHYSICS_REPULSION",
        "GRAPH_PHYSICS_CENTRAL_GRAVITY",
        "GRAPH_PHYSICS_STABILIZATION_ITERS",
    }
)


def scheme_config_path() -> str:
    if not ACTIVE_SCHEME:
        return ""
    return os.path.join(scheme_root(), SCHEME_CONFIG_FILENAME)


def _coerce_scheme_value(key: str, raw, current):
    """將 JSON 值轉成與目前變數相容的型別。"""
    if current is None:
        if raw is None or str(raw).strip().lower() in ("", "null", "none"):
            return None
        if isinstance(raw, (int, float, str, bool)):
            return raw
        raise TypeError(f"{key}: 無法推斷型別（目前為 null）")
    if isinstance(current, bool):
        if isinstance(raw, bool):
            return raw
        s = str(raw).strip().lower()
        if s in ("true", "1", "yes"):
            return True
        if s in ("false", "0", "no"):
            return False
        raise ValueError(f"{key}: 需為布林值")
    if isinstance(current, int) and type(current) is not bool:
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    if isinstance(current, str):
        return str(raw)
    return raw


def _sync_tracking_mode_derived():
    """依目前 TRACKING_MODE 寫入一組追蹤門檻（與本檔上半部邏輯一致）。"""
    m = sys.modules[__name__]
    tm = (m.TRACKING_MODE or "balanced").lower()
    if isinstance(tm, str) and tm not in ("stable_names", "more_coverage", "balanced"):
        tm = "balanced"
        m.TRACKING_MODE = tm
    if tm == "stable_names":
        m.FIRST_ASSIGN_MAX_DISTANCE = 0.55
        m.MIN_MARGIN_BEFORE_ASSIGN = 0.06
        m.CORRECTION_INTERVAL_FRAMES = 90
        m.CORRECTION_MAX_DISTANCE = 0.44
        m.TRACK_PERSISTENCE_FRAMES = 70
        m.TRACK_IOU_MIN, m.TRACK_AREA_MIN, m.TRACK_AREA_MAX, m.TRACK_CENTER_MAX, m.TRACK_IOU_FALLBACK = (
            0.20,
            0.60,
            1.75,
            160,
            0.08,
        )
    elif tm == "more_coverage":
        m.FIRST_ASSIGN_MAX_DISTANCE = 0.58
        m.MIN_MARGIN_BEFORE_ASSIGN = 0.05
        m.CORRECTION_INTERVAL_FRAMES = 60
        m.CORRECTION_MAX_DISTANCE = 0.46
        m.TRACK_PERSISTENCE_FRAMES = 85
        m.TRACK_IOU_MIN, m.TRACK_AREA_MIN, m.TRACK_AREA_MAX, m.TRACK_CENTER_MAX, m.TRACK_IOU_FALLBACK = (
            0.17,
            0.52,
            1.92,
            195,
            0.05,
        )
    else:
        m.FIRST_ASSIGN_MAX_DISTANCE = 0.57
        m.MIN_MARGIN_BEFORE_ASSIGN = 0.055
        m.CORRECTION_INTERVAL_FRAMES = 75
        m.CORRECTION_MAX_DISTANCE = 0.45
        m.TRACK_PERSISTENCE_FRAMES = 78
        m.TRACK_IOU_MIN, m.TRACK_AREA_MIN, m.TRACK_AREA_MAX, m.TRACK_CENTER_MAX, m.TRACK_IOU_FALLBACK = (
            0.18,
            0.55,
            1.85,
            180,
            0.06,
        )
    m.FIRST_ASSIGN_MIN_MARGIN = m.MIN_MARGIN_BEFORE_ASSIGN
    m.STRONG_OVERRIDE_MIN_MARGIN = m.MIN_MARGIN_BEFORE_ASSIGN


def _sync_recognition_preset_adjustments():
    """若 RECOGNITION_PRESET 為 classroom_masked，套用與本檔一致的加寬／圖譜門檻與 TRACKING_MODE。"""
    m = sys.modules[__name__]
    preset = (m.RECOGNITION_PRESET or "default").strip().lower()
    if preset != "classroom_masked":
        return
    m.TRACKING_MODE = "more_coverage"
    m.FACE_MATCH_TOLERANCE = 0.60
    m.YOLO_FACE_TOLERANCE_CAP = 0.66
    m.YOLO_FACE_TOLERANCE_EXTRA = 0.07
    m.LOOSE_FALLBACK_MIN_MARGIN = 0.048
    m.NAME_MIN_SCORE_TO_SHOW = 0.50
    m.NAME_SWITCH_TOP2_DELTA = 0.20
    m.NAME_SWITCH_SCORE_DELTA = 0.28
    m.NAME_SCORE_INCREMENT_SCALE = 1.12
    m.USE_APPEARANCE_AUXILIARY = True
    m.APPEARANCE_FACE_DISTANCE_BONUS = 0.022
    m.APPEARANCE_ONLY_TOLERANCE = 0.58
    m.GRAPH_CONFIRMED_MAX_DISTANCE = 0.48
    m.YOLO_FACE_UPSAMPLE_FALLBACK = 0


def _apply_scheme_config_from_disk() -> list[str]:
    """
    有 ACTIVE_SCHEME 且存在 scheme_config.json 時，覆寫允許清單內的變數。
    先套用 RECOGNITION_PRESET → 教室預設補齊；再 TRACKING_MODE → 衍生門檻；其餘鍵最後套用（可覆寫前面結果）。
    回傳已套用鍵名（供列印）。
    """
    if not ACTIVE_SCHEME:
        return []
    path = scheme_config_path()
    if not path or not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            blob = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(blob, dict):
        return []
    unknown = [k for k in blob if k not in SCHEME_CONFIG_ALLOWLIST]
    if unknown:
        print(
            "[方案參數] scheme_config.json 有下列鍵不在允許清單，已略過：",
            ", ".join(sorted(unknown)),
        )
    mod = sys.modules[__name__]
    applied_keys: list[str] = []
    remaining = dict(blob)

    def _apply_one(k: str, v):
        if k not in SCHEME_CONFIG_ALLOWLIST:
            return
        if not hasattr(mod, k):
            return
        cur = getattr(mod, k)
        try:
            newv = _coerce_scheme_value(k, v, cur)
        except (TypeError, ValueError) as e:
            print(f"[方案參數] 略過 {k}（無法轉換: {e}）")
            return
        setattr(mod, k, newv)
        applied_keys.append(k)

    if "RECOGNITION_PRESET" in remaining:
        _apply_one("RECOGNITION_PRESET", remaining.pop("RECOGNITION_PRESET"))
    _sync_recognition_preset_adjustments()

    if "TRACKING_MODE" in remaining:
        _apply_one("TRACKING_MODE", remaining.pop("TRACKING_MODE"))
    _sync_tracking_mode_derived()

    for ky in sorted(remaining.keys()):
        _apply_one(ky, remaining[ky])

    if applied_keys:
        preview = ", ".join(applied_keys[:20])
        more = " …" if len(applied_keys) > 20 else ""
        print(f"[方案參數] 已自 scheme_config.json 套用 {len(applied_keys)} 項：{preview}{more}")

    return applied_keys


def scheme_config_snapshot_for_template() -> dict:
    """目前記憶體中的允許參數快照（供 init 範本）；值須可 JSON 序列化。"""
    mod = sys.modules[__name__]
    out: dict = {}
    for k in sorted(SCHEME_CONFIG_ALLOWLIST):
        if not hasattr(mod, k):
            continue
        v = getattr(mod, k)
        if isinstance(v, frozenset):
            continue
        if v is not None and not isinstance(v, (bool, int, float, str)):
            continue
        out[k] = v
    return out


def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(REGISTERED_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(GRAPH_DIR, exist_ok=True)
    os.makedirs(SCHEMES_PARENT, exist_ok=True)
    if ACTIVE_SCHEME:
        root = scheme_root()
        os.makedirs(root, exist_ok=True)
        os.makedirs(registered_dir(), exist_ok=True)
        os.makedirs(output_dir(), exist_ok=True)
        os.makedirs(graph_dir(), exist_ok=True)
