# 繪製方法架構圖（依本專案實作整理，非外加功能）
#
# 【依據】數值與常數名稱：import config（config.py）。流程與模組：main.py、face_registry.py、
# yolo_tracker.py / video_tracker.py、extract_faces.py、knowledge_graph.py、personality.py、
# relationship_graph.py。圖上文字為上述程式行為之「摘要」，未發明額外演算法或檔名。
#
# 輸出：methodology_architecture.png（主圖）、methodology_config_table.png（參數表）
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config as cfg  # noqa: E402

for font in ("Microsoft JhengHei", "Microsoft YaHei", "SimHei", "PingFang TC"):
    plt.rcParams["font.sans-serif"] = [font, "DejaVu Sans"]
    break
plt.rcParams["axes.unicode_minus"] = False

# --- 參考圖風格配色 ---
C_INPUT = "#FFE0B2"
C_REGISTRY = "#BBDEFB"
C_JSON = "#ECEFF1"
C_TRACK = "#C8E6C9"
C_MATCH = "#E1BEE7"
C_INTERACT = "#B3E5FC"
C_KG = "#B2DFDB"
C_OUT = "#F8BBD9"
EC = "#37474F"


def draw_numbered_step_from_top(ax, y_top, w, left, step_no, title, lines, fc, ec=EC, title_fs=15, body_fs=12):
    """由上而下排版：y_top 為方塊上緣（axes 座標，1=圖上方）。回傳方塊下緣 y_bottom。"""
    line_h = 0.0152
    title_body_gap = 0.028
    pad_t, pad_b, pad_x = 0.012, 0.008, 0.026
    h = pad_t + pad_b + title_body_gap + len(lines) * line_h
    y_bottom = y_top - h
    x = left
    p = FancyBboxPatch(
        (x, y_bottom),
        w,
        h,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        transform=ax.transAxes,
        facecolor=fc,
        edgecolor=ec,
        linewidth=1.35,
        clip_on=False,
    )
    ax.add_patch(p)
    tx = x + pad_x
    ty = y_top - pad_t
    ax.text(
        tx,
        ty,
        f"步驟 {step_no}  {title}",
        transform=ax.transAxes,
        fontsize=title_fs,
        fontweight="bold",
        color="#1A237E",
        va="top",
    )
    ty -= title_body_gap
    for ln in lines:
        ax.text(
            tx,
            ty,
            f"• {ln}",
            transform=ax.transAxes,
            fontsize=body_fs,
            va="top",
            color="#263238",
            linespacing=1.05,
        )
        ty -= line_h
    return y_bottom


def v_arrow_down(ax, y_from, y_to, x_center):
    """由 y_from（上）指到 y_to（下），兩者皆為 axes fraction。"""
    ax.annotate(
        "",
        xy=(x_center, y_to),
        xytext=(x_center, y_from),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color="#546E7A", lw=1.8, shrinkA=2, shrinkB=2),
    )


def build_config_table_rows():
    c = cfg
    yolo_tol = min(0.72, float(c.FACE_MATCH_TOLERANCE) + 0.12)
    return [
        ("【臉部與註冊】", "", ""),
        ("FACE_MATCH_TOLERANCE", str(c.FACE_MATCH_TOLERANCE), "純臉追蹤匹配門檻"),
        ("YOLO 辨識幀有效門檻", f"min(0.72,TOL+0.12)={yolo_tol:.2f}", "YOLO 路徑臉比對上限"),
        ("MAX_TEMPLATES_PER_PERSON", str(c.MAX_TEMPLATES_PER_PERSON), "每人臉模版數上限"),
        ("【YOLO 人體】", "", ""),
        ("YOLO_PERSON_MODEL_SIZE", str(c.YOLO_PERSON_MODEL_SIZE), "yolov8s.pt 預訓練"),
        ("YOLO_IMGSZ", str(c.YOLO_IMGSZ), "推論縮放邊長"),
        ("YOLO_CONF", str(c.YOLO_CONF), "人體偵測信心"),
        ("YOLO_IOU", str(c.YOLO_IOU), "NMS IoU"),
        ("YOLO_MAX_DET", str(c.YOLO_MAX_DET), "單幀最多人數"),
        ("YOLO_TRACKER", str(c.YOLO_TRACKER), "追蹤器設定檔"),
        ("YOLO_MIN_PERSON_AREA_FRAC", str(c.YOLO_MIN_PERSON_AREA_FRAC), "過小人框丟棄"),
        ("【追蹤模式】TRACKING_MODE", str(c.TRACKING_MODE), "balanced / stable_names / more_coverage"),
        ("FIRST_ASSIGN_MAX_DISTANCE", str(c.FIRST_ASSIGN_MAX_DISTANCE), "首次掛名距離上限"),
        ("MIN_MARGIN_BEFORE_ASSIGN", str(c.MIN_MARGIN_BEFORE_ASSIGN), "掛名時與第二名最小差距"),
        ("TRACK_PERSISTENCE_FRAMES", str(c.TRACK_PERSISTENCE_FRAMES), "軌道名字保留幀數"),
        ("TRACK_IOU_MIN 等", f"{c.TRACK_IOU_MIN},{c.TRACK_AREA_MIN}…", "人框幾何配對"),
        ("NAME_MIN_SCORE_TO_SHOW", str(c.NAME_MIN_SCORE_TO_SHOW), "累積分數達此才顯示真名"),
        ("NAME_SWITCH_TOP2_DELTA", str(c.NAME_SWITCH_TOP2_DELTA), "第一名須領先第二名"),
        ("STRONG_MATCH_OVERRIDE_DISTANCE", str(c.STRONG_MATCH_OVERRIDE_DISTANCE), "強匹配覆蓋距離"),
        ("NEVER_SWITCH_NAME", str(c.NEVER_SWITCH_NAME), "一般幀是否禁止換名"),
        ("【互動與圖譜】", "", ""),
        ("NEAR_DISTANCE_PX", str(c.NEAR_DISTANCE_PX), "兩人框中心距離門檻(px)"),
        ("SAME_FRAME_COOLDOWN", str(c.SAME_FRAME_COOLDOWN), "同框計次時間間隔(秒)"),
        ("GRAPH_CONFIRMED_MAX_DISTANCE", str(c.GRAPH_CONFIRMED_MAX_DISTANCE), "嚴格模式單幀高信心"),
        ("MIN_COOCCURRENCE_FOR_FRIEND", str(c.MIN_COOCCURRENCE_FOR_FRIEND), "畫朋友邊最少同框次"),
        ("TRACK_FINAL_NAME_RETROSPECTIVE", str(c.TRACK_FINAL_NAME_RETROSPECTIVE_INTERACTIONS), "軌道回溯互動統計"),
        ("【個性 proxy】", "", ""),
        ("PERSONALITY_EXTREME_GROUP_FRAC", str(c.PERSONALITY_EXTREME_GROUP_FRAC), "整合度極端組比例"),
        ("PERSONALITY_MIN_PEOPLE", str(c.PERSONALITY_MIN_PEOPLE), "分位法最少人數"),
        ("USE_YOLO_TRACKER", str(c.USE_YOLO_TRACKER), "True=YOLO+臉 / False=純臉"),
        ("DETECT_EVERY_N_FRAMES", str(c.DETECT_EVERY_N_FRAMES), "每 N 幀臉辨識一次"),
        ("FIXED_NAME_PER_TRACK", str(c.FIXED_NAME_PER_TRACK), "一軌道一固定名字"),
    ]


def render_config_table_png(out_path: str):
    fig, ax = plt.subplots(figsize=(18, 24), dpi=150)
    ax.set_axis_off()
    ax.text(
        0.5,
        0.985,
        "config.py 參數完整一覽（由程式 import config 讀取）",
        ha="center",
        fontsize=17,
        fontweight="bold",
        transform=ax.transAxes,
    )
    rows = build_config_table_rows()
    table_data = [[a, b, c] for a, b, c in rows]
    tbl = ax.table(
        cellText=table_data,
        colLabels=["參數名稱", "目前數值", "說明"],
        loc="upper center",
        cellLoc="left",
        colColours=["#ECEFF1"] * 3,
        bbox=[0.02, 0.02, 0.96, 0.90],
        colWidths=[0.34, 0.22, 0.42],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1.0, 1.52)
    for k, cell in tbl.get_celld().items():
        ri, ci = k
        if ri == 0:
            cell.set_text_props(fontweight="bold", fontsize=11)
        elif table_data[ri - 1][0].startswith("【"):
            cell.set_facecolor("#E8EAF6")
            cell.set_text_props(fontweight="bold")
    fig.savefig(out_path, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "methodology_architecture.png")
    table_path = os.path.join(out_dir, "methodology_config_table.png")

    c = cfg
    yolo_tol = min(0.72, float(c.FACE_MATCH_TOLERANCE) + 0.12)

    fig = plt.figure(figsize=(16.5, 30), dpi=150)

    # ----- 頁首 -----
    ax_h = fig.add_axes([0.055, 0.928, 0.89, 0.068])
    ax_h.set_axis_off()
    ax_h.text(
        0.5,
        0.82,
        "幼兒追蹤與知識圖譜 — 方法架構總覽",
        ha="center",
        fontsize=24,
        fontweight="bold",
        color="#0D47A1",
        transform=ax_h.transAxes,
    )
    ax_h.text(
        0.5,
        0.44,
        "【無訓練】未對教室影片或幼兒臉部做深度學習 fine-tune。人體：YOLOv8 預訓練（COCO person）；臉部：dlib / face_recognition 128 維預訓練編碼。",
        ha="center",
        fontsize=14,
        color="#263238",
        transform=ax_h.transAxes,
        linespacing=1.12,
    )
    ax_h.text(
        0.5,
        0.05,
        "【註冊】將照片 encoding 寫入 face_registry.json（樣本庫擴充，非反向傳播）。【比對】face_distance；每人多模版取 min；YOLO 辨識幀門檻 min(0.72, TOL+0.12)（見 yolo_tracker.py）。",
        ha="center",
        fontsize=12.5,
        color="#37474F",
        transform=ax_h.transAxes,
        linespacing=1.12,
    )

    # ----- 主欄：步驟 1–8（由上而下：步驟 1 在上方）-----
    ax = fig.add_axes([0.055, 0.052, 0.89, 0.872])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    left, w = 0.02, 0.96
    xc = left + w / 2
    gap = 0.0045
    y_top = 0.996

    y_bot = draw_numbered_step_from_top(
        ax,
        y_top,
        w,
        left,
        1,
        "輸入資料",
        [
            f"註冊照片：每人最多 {c.MAX_TEMPLATES_PER_PERSON} 張臉模版（MAX_TEMPLATES_PER_PERSON）",
            "教室影片：MP4，作為追蹤、身分比對與互動統計來源",
            "指令：main.py register / process / build-graph / run-all",
        ],
        C_INPUT,
    )
    y_next_top = y_bot - gap
    v_arrow_down(ax, y_bot, y_next_top, xc)

    y_bot = draw_numbered_step_from_top(
        ax,
        y_next_top,
        w,
        left,
        2,
        "臉部註冊模組（face_registry.py）",
        [
            "多階段偵測臉部：HOG + upsample；必要時 CNN 或放大影像以提高偵測率",
            "face_encodings：輸出 128 維向量；註冊時可選 --cnn 較準較慢",
            "比對時：每人多模版逐一算距離，取該人之 min，再與其他人比較（非單一中心向量）",
        ],
        C_REGISTRY,
    )
    y_next_top = y_bot - gap
    v_arrow_down(ax, y_bot, y_next_top, xc)

    y_bot = draw_numbered_step_from_top(
        ax,
        y_next_top,
        w,
        left,
        3,
        "註冊庫 JSON",
        [
            "檔案：data/face_registry.json（姓名、encodings、照片路徑等）",
            "process 載入比對；build-graph 用於補齊節點（已註冊者即使未同框也會出現在圖譜）",
        ],
        C_JSON,
    )
    y_next_top = y_bot - gap
    v_arrow_down(ax, y_bot, y_next_top, xc)

    y_bot = draw_numbered_step_from_top(
        ax,
        y_next_top,
        w,
        left,
        4,
        "影片追蹤（config：USE_YOLO_TRACKER；CLI：--yolo / --no-yolo）",
        [
            f"模式 A【預設】：YOLOv8 人體 + 追蹤器（{c.YOLO_TRACKER}）+ 自建 our_id",
            "人體軌道延續：當前幀與上一幀人框做 IoU／面積／中心距離幾何配對（不依賴 YOLO 內建 track id）",
            "整張畫面偵測臉，再將臉配對到對應人體框內",
            "模式 B：video_tracker 純臉 + OpenCV 追蹤；無 our_id，不適用軌道回溯互動統計",
        ],
        C_TRACK,
    )
    y_next_top = y_bot - gap
    v_arrow_down(ax, y_bot, y_next_top, xc)

    y_bot = draw_numbered_step_from_top(
        ax,
        y_next_top,
        w,
        left,
        5,
        "身分匹配與掛名",
        [
            f"距離門檻（YOLO 辨識幀）：min(0.72, FACE_MATCH_TOLERANCE+0.12) = {yolo_tol:.2f}",
            f"純臉路徑：FACE_MATCH_TOLERANCE = {c.FACE_MATCH_TOLERANCE}",
            f"需累積分數達 NAME_MIN_SCORE_TO_SHOW = {c.NAME_MIN_SCORE_TO_SHOW}；第一與第二名須滿足 NAME_SWITCH_TOP2_DELTA = {c.NAME_SWITCH_TOP2_DELTA}",
            f"一般幀 NEVER_SWITCH_NAME = {c.NEVER_SWITCH_NAME}；極確定時可用強匹配覆蓋（距離 <= {c.STRONG_MATCH_OVERRIDE_DISTANCE} 等條件）",
            "回溯互動開啟時：片尾依每條軌道「最終歸屬名」重算 interactions.json（僅統計；不重編已輸出影片像素）",
        ],
        C_MATCH,
    )
    y_next_top = y_bot - gap
    v_arrow_down(ax, y_bot, y_next_top, xc)

    y_bot = draw_numbered_step_from_top(
        ax,
        y_next_top,
        w,
        left,
        6,
        "互動紀錄（interactions.json）",
        [
            f"靠近：兩人體框中心距離 < NEAR_DISTANCE_PX（{c.NEAR_DISTANCE_PX} px）",
            f"同框：同一幀內兩位不同姓名同時出現；SAME_FRAME_COOLDOWN = {c.SAME_FRAME_COOLDOWN} 秒內同一對不重複加計",
            f"TRACK_FINAL_NAME_RETROSPECTIVE_INTERACTIONS = {c.TRACK_FINAL_NAME_RETROSPECTIVE_INTERACTIONS}",
            "True：每軌道回溯最終名後重算整段同框／靠近；False：僅本幀距離 <= GRAPH_CONFIRMED_MAX_DISTANCE 計入",
        ],
        C_INTERACT,
    )
    y_next_top = y_bot - gap
    v_arrow_down(ax, y_bot, y_next_top, xc)

    y_bot = draw_numbered_step_from_top(
        ax,
        y_next_top,
        w,
        left,
        7,
        "知識圖譜與個性（knowledge_graph.py + personality.py）",
        [
            f"朋友邊：兩人同框次數 >= MIN_COOCCURRENCE_FOR_FRIEND（{c.MIN_COOCCURRENCE_FOR_FRIEND}）",
            f"個性標籤：全班互動指標相對分位（proxy），例如 PERSONALITY_EXTREME_GROUP_FRAC = {c.PERSONALITY_EXTREME_GROUP_FRAC}",
            "視覺化：relationship_graph.py 以 pyvis 輸出互動 HTML；節點顏色對應個性類型",
        ],
        C_KG,
    )
    y_next_top = y_bot - gap
    v_arrow_down(ax, y_bot, y_next_top, xc)

    draw_numbered_step_from_top(
        ax,
        y_next_top,
        w,
        left,
        8,
        "輸出產物",
        [
            "影片：data/output/*.mp4（人框／臉旁疊加繁體中文姓名或「未知」）",
            "關係圖：data/graph/relationship_graph.html（瀏覽器開啟）",
            "結構化：data/graph/knowledge_graph.json",
        ],
        C_OUT,
    )

    fig.savefig(out_path, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print("已輸出:", out_path)

    render_config_table_png(table_path)
    print("已輸出:", table_path)
    return out_path


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("繪圖失敗:", e, file=sys.stderr)
        sys.exit(1)
