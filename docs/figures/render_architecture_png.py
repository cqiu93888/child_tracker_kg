"""
輸出技術架構區塊圖與資料流示意圖（PNG）。
執行：python docs/figures/render_architecture_png.py

架構圖：強調 yolo_tracker 與 video_tracker 為「擇一」路徑，非串接。
時序圖：讀取 JSON 箭頭由檔案指向建圖模組。
"""
from __future__ import annotations

import pathlib

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent

# 印在圖角，方便確認已重新輸出（與 Word 內嵌快取無關）。
FIGURE_VERSION = "2026-04-18-v2"

for fn in ("Microsoft JhengHei", "Microsoft YaHei", "SimHei", "PingFang TC"):
    try:
        plt.rcParams["font.sans-serif"] = [fn]
        break
    except Exception:
        pass
plt.rcParams["axes.unicode_minus"] = False


def _box(ax, xy, w, h, text, fc="#FFFFFF", ec="#1565C0", fs=9):
    x, y = xy
    r = mpatches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.2,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(r)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        linespacing=1.12,
    )


def _arrow(ax, a, b, color="#424242"):
    ax.annotate(
        "",
        xy=b,
        xytext=a,
        arrowprops=dict(
            arrowstyle="->",
            color=color,
            lw=1.4,
            shrinkA=4,
            shrinkB=4,
        ),
    )


def _arrow_down(ax, x: float, y_hi: float, y_lo: float, color="#757575", lw: float = 1.15):
    """垂直向下箭頭（主流程連接用）。"""
    ax.annotate(
        "",
        xy=(x, y_lo),
        xytext=(x, y_hi),
        arrowprops=dict(arrowstyle="->", color=color, lw=lw, shrinkA=3, shrinkB=3),
    )


def _elbow_down(ax, x0: float, y_start: float, x1: float, y_end: float, color: str, lw: float = 1.1):
    """
    由上往下直角折線（y_start > y_end）：(x0,y_start)→垂直到轉折→水平→垂直到 (x1,y_end)。
    """
    y_mid = (y_start + y_end) / 2
    ax.plot([x0, x0, x1], [y_start, y_mid, y_mid], color=color, lw=lw, solid_capstyle="round")
    ax.annotate(
        "",
        xy=(x1, y_end),
        xytext=(x1, y_mid),
        arrowprops=dict(arrowstyle="->", color=color, lw=lw, shrinkA=2, shrinkB=4),
    )


def draw_architecture():
    """
    單欄區塊圖：①②③ 帶狀區 **垂直堆疊不重疊**，標題與內框留出帶寬；主流程加垂直箭頭；
    YOLO→yolo_tracker 用直角折線；備援用兩框之間的短虛線，不橫穿標題。
    """
    fig_h = 15.4
    fig, ax = plt.subplots(figsize=(10.6, fig_h), dpi=150)
    ax.set_xlim(0, 10.6)
    ax.set_ylim(0, fig_h)
    ax.axis("off")
    ax.set_title(
        f"child_tracker_kg · 技術架構總覽（區塊圖）　〔{FIGURE_VERSION}〕",
        fontsize=14,
        fontweight="bold",
        pad=10,
    )

    LM = 0.5
    W = 9.1
    cx = LM + W / 2
    FLOW_X = LM + 0.34
    SEP_ZONE = 0.48
    SEP_FLOW = 0.32
    BAND_SEP = 0.56

    def zone(y_top: float, h: float, title: str, fc: str, ec: str) -> tuple[float, float]:
        y_lo = y_top - h
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (LM, y_lo),
                W,
                h,
                boxstyle="round,pad=0.02,rounding_size=0.045",
                facecolor=fc,
                edgecolor=ec,
                linewidth=1.0,
                alpha=0.36,
            )
        )
        ax.text(LM + 0.12, y_top - 0.24, title, fontsize=10.5, fontweight="bold", color="#212121")
        return y_top, y_lo

    # —— ①②③：上緣依序下排，不與上一層重疊 ——
    y_hi1 = fig_h - 0.95
    h1 = 1.42
    y_hi1, y_lo1 = zone(y_hi1, h1, "輸入", "#FFE0B2", "#E65100")
    y_in = y_lo1 + 0.38
    _box(
        ax,
        (LM + 0.5, y_in),
        3.1,
        0.7,
        "註冊照片\n（data/schemes/<方案>/registered）",
        "#FFFFFF",
        "#BF360C",
        8.4,
    )
    _box(ax, (LM + 4.5, y_in), 3.1, 0.7, "影片檔\nMP4（--video）", "#FFFFFF", "#BF360C", 8.8)

    y_hi2 = y_lo1 - SEP_ZONE
    h2 = 1.52
    y_hi2, y_lo2 = zone(y_hi2, h2, "本機執行", "#C8E6C9", "#2E7D32")
    y_ex = y_lo2 + 0.44
    bh_ex = 0.78
    _box(ax, (LM + 0.48, y_ex), 3.25, bh_ex, "main.py\nregister / process / build-graph", "#FFFFFF", "#1B5E20", 8.6)
    sub_l = LM + 4.18
    sub_w = W - 4.58
    _box(
        ax,
        (sub_l, y_ex),
        sub_w,
        bh_ex,
        "YOLO：子行程 run_yolo_isolated.py\n（main 以 subprocess 啟動；與 dlib 分離）",
        "#FFFFFF",
        "#1B5E20",
        7.8,
    )
    _arrow(ax, (LM + 3.75, y_ex + bh_ex * 0.5), (sub_l - 0.02, y_ex + bh_ex * 0.5), "#2E7D32")
    ax.text(LM + 3.72, y_ex + bh_ex + 0.06, "subprocess", fontsize=7, color="#2E7D32", ha="right")

    y_hi3 = y_lo2 - SEP_ZONE
    h3 = 2.62
    y_hi3, y_lo3 = zone(
        y_hi3,
        h3,
        "核心處理（擇一 · 非先後串接）",
        "#E1BEE7",
        "#6A1B9A",
    )
    y_tr = y_lo3 + 0.5
    bh_tr = 1.08
    yolo_l = LM + 0.42
    yolo_w = 4.12
    vid_l = LM + 4.68
    vid_w = W - 5.1
    _box(
        ax,
        (yolo_l, y_tr),
        yolo_w,
        bh_tr,
        "yolo_tracker\n人體偵測＋臉部比對＋互動統計\n（YOLOv8；臉部：dlib／InsightFace）",
        "#FFFFFF",
        "#6A1B9A",
        8.0,
    )
    _box(
        ax,
        (vid_l, y_tr),
        vid_w,
        bh_tr,
        "video_tracker\n本機「純臉部」備援\n（YOLO 子行程失敗且自動 fallback）",
        "#FFFFFF",
        "#6A1B9A",
        7.85,
    )
    sub_cx = sub_l + sub_w / 2
    yolo_cx = yolo_l + yolo_w / 2
    y_sub_bottom = y_ex + 0.06
    y_yolo_top = y_tr + bh_tr - 0.02
    _elbow_down(ax, sub_cx, y_sub_bottom, yolo_cx, y_yolo_top, "#2E7D32", lw=1.12)

    gap_l = yolo_l + yolo_w
    gap_r = vid_l
    y_fb = y_tr + bh_tr * 0.5
    if gap_r > gap_l + 0.06:
        ax.annotate(
            "",
            xy=(gap_r - 0.04, y_fb),
            xytext=(gap_l + 0.04, y_fb),
            arrowprops=dict(
                arrowstyle="->",
                color="#7B1FA2",
                lw=1.0,
                linestyle=(0, (4, 3)),
            ),
        )
        ax.text(
            (gap_l + gap_r) / 2,
            y_fb - 0.24,
            "備援（fallback）",
            ha="center",
            fontsize=7.0,
            color="#6A1B9A",
        )

    # ①→②、②→③ 主流程（左側通道，避免與內框、YOLO 折線疊在一起）
    _arrow_down(ax, FLOW_X, y_lo1 + 0.06, y_hi2 - 0.02, "#9E9E9E", lw=1.0)
    _arrow_down(ax, FLOW_X, y_lo2 + 0.06, y_hi3 - 0.02, "#9E9E9E", lw=1.0)

    # 輸入 → 本機執行（對準兩框區）
    y_t1 = y_in + 0.7
    y_t2 = y_ex + bh_ex - 0.04
    _arrow(ax, (LM + 2.05, y_t1), (LM + 1.65, y_t2), "#757575")
    _arrow(ax, (LM + 6.05, y_t1), (LM + 7.25, y_t2), "#757575")

    # ④ interactions.json
    y_above_json = y_lo3 - SEP_FLOW
    box_h = 0.72
    y_box = y_above_json - box_h
    _arrow_down(ax, FLOW_X, y_lo3 + 0.08, y_box + box_h + 0.04, "#6D6D6D", lw=1.0)
    _box(
        ax,
        (LM + 1.15, y_box),
        W - 2.3,
        box_h,
        "寫入 data/schemes/<方案>/interactions.json\n（與 --scheme 一致；同框、靠近、可選 interaction_timeline）",
        "#E3F2FD",
        "#1565C0",
        8.3,
    )

    # ⑤ 資料儲存
    y_hi_st, y_lo_st = zone(y_box - BAND_SEP, 1.24, "資料儲存（檔案系統，非資料庫）", "#ECEFF1", "#455A64")
    _arrow_down(ax, FLOW_X, y_box - 0.06, y_hi_st + 0.02, "#6D6D6D", lw=1.0)
    ax.text(
        LM + 0.15,
        y_hi_st - 0.52,
        "data/schemes/<方案>/registered/、…/output/*.mp4、…/interactions.json、\n"
        "…/graph/knowledge_graph.json、…/graph/*.html（含 ego/）",
        fontsize=8.35,
        color="#37474F",
        va="top",
    )

    # ⑥ 圖譜與標籤
    y_hi_gr, y_lo_gr = zone(y_lo_st - BAND_SEP, 1.58, "圖譜與標籤（build-graph）", "#B3E5FC", "#0277BD")
    _arrow_down(ax, FLOW_X, y_lo_st - 0.05, y_hi_gr + 0.02, "#6D6D6D", lw=1.0)
    yr = y_lo_gr + 0.22
    _box(
        ax,
        (LM + 0.4, yr),
        2.5,
        0.88,
        "knowledge_graph\nNetworkX → knowledge_graph.json\n（data/schemes/<方案>/graph/）",
        "#FFFFFF",
        "#01579B",
        7.2,
    )
    _box(ax, (LM + 3.05, yr), 2.35, 0.88, "personality\n互動相對分位（proxy）", "#FFFFFF", "#01579B", 7.6)
    _box(
        ax,
        (LM + 5.55, yr),
        W - 5.9,
        0.88,
        "relationship_graph\npyvis → …/graph/*.html\n（同上方案目錄）",
        "#FFFFFF",
        "#01579B",
        7.0,
    )

    # ⑦ 視覺化
    y_hi_vz, y_lo_vz = zone(y_lo_gr - BAND_SEP, 1.28, "視覺化（靜態 HTML，瀏覽器離線開啟）", "#FFECB3", "#F57C00")
    _arrow_down(ax, FLOW_X, y_lo_gr - 0.05, y_hi_vz + 0.02, "#6D6D6D", lw=1.0)
    yr = y_lo_vz + 0.22
    _box(
        ax,
        (LM + 0.4, yr),
        2.65,
        0.74,
        "…/graph/\nrelationship_graph.html\nvis-network",
        "#FFFFFF",
        "#E65100",
        7.8,
    )
    _box(ax, (LM + 3.2, yr), 3.2, 0.74, "Chart.js（注入）\n節點 hover 互動時間軸", "#FFFFFF", "#E65100", 7.9)
    _box(ax, (LM + 6.55, yr), W - 6.9, 0.74, "無後端伺服器", "#FAFAFA", "#78909C", 9)

    foot_y = y_lo_vz - 0.38
    ax.text(
        cx,
        max(0.42, foot_y),
        "※ 入口：main.py。多方案時以 data/schemes/<方案>/ 為根；雲端可選 sync_to_cloud → Render（api_cloud／LINE）。",
        ha="center",
        fontsize=8,
        color="#616161",
    )
    ax.text(
        LM + W - 0.05,
        fig_h - 0.35,
        f"圖面版本\n{FIGURE_VERSION}",
        ha="right",
        va="top",
        fontsize=8.5,
        color="#B71C1C",
        fontweight="bold",
        linespacing=1.05,
    )

    out = HERE / "architecture_overview.png"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def draw_sequence():
    fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title(
        f"資料流與操作順序（示意）　〔{FIGURE_VERSION}〕",
        fontsize=15,
        fontweight="bold",
        pad=12,
    )

    cols = [
        (1.15, "使用者"),
        (2.75, "main.py"),
        (4.35, "註冊 / 追蹤\n模組"),
        (6.05, "interactions.json\n（各方案子目錄內）"),
        (7.85, "knowledge_graph\n+ pyvis"),
        (10.15, "HTML"),
    ]
    y_top = 8.35
    y_bot = 1.35
    for x, name in cols:
        ax.plot([x, x], [y_bot, y_top], color="#B0BEC5", lw=1.5, ls="--")
        ax.text(x, y_top + 0.4, name, ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # (from_col, to_col) — 步驟 8 改為 JSON(3) → 建圖(4)
    steps = [
        (0, 1),
        (1, 2),
        (0, 1),
        (1, 2),
        (2, 3),
        (0, 1),
        (1, 4),
        (3, 4),
        (1, 5),
        (0, 5),
    ]
    xs = [c[0] for c in cols]
    y = 7.85
    dy = 0.62
    labels = [
        "1 註冊",
        "2 寫入 registry",
        "3 處理影片",
        "4 process_video",
        "5 寫入 interactions",
        "6 build-graph",
        "7 呼叫建圖 / 繪圖",
        "8 讀取 interactions（JSON→建圖）",
        "9 輸出 HTML",
        "10 瀏覽器開啟",
    ]
    for i, ((a, b), lab) in enumerate(zip(steps, labels)):
        ya = y - i * dy
        # 步驟 8：箭頭由 interactions.json(3) 指向建圖(4)，與「讀取」語意一致
        col = "#1565C0" if i not in (4, 7) else "#6A1B9A"
        off = 0.0
        if i == 7:
            off = 0.04
        _arrow(ax, (xs[a], ya + 0.06 + off), (xs[b], ya + 0.06 + off), color=col)
        ax.text(
            (xs[a] + xs[b]) / 2,
            ya + 0.2 + off,
            lab,
            ha="center",
            fontsize=7.6 if i != 7 else 7.2,
            color="#37474F",
        )

    ax.text(
        6,
        0.55,
        "※ YOLO 預設：main.py 以 subprocess 執行 run_yolo_isolated.py。註冊用 face_registry；"
        "處理影片用 yolo_tracker 或 video_tracker（擇一）。互動／圖譜檔在 data/schemes/<方案>/ 下。",
        ha="center",
        fontsize=7.5,
        style="italic",
        color="#616161",
    )
    ax.text(
        11.85,
        9.75,
        f"圖面版本\n{FIGURE_VERSION}",
        ha="right",
        va="top",
        fontsize=8.5,
        color="#B71C1C",
        fontweight="bold",
        linespacing=1.05,
    )

    out = HERE / "dataflow_sequence.png"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


if __name__ == "__main__":
    a = draw_architecture()
    b = draw_sequence()
    print("已輸出:", a)
    print("已輸出:", b)
