# 在 OpenCV 畫面上繪製中文（或任何 Unicode）標籤
# cv2.putText 不支援中文，會變成空白或問號，改用 PIL 繪製後疊回
import os
import numpy as np
import cv2

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# 常見中文字體路徑（依序嘗試）
FONT_PATHS = [
    "C:/Windows/Fonts/msjh.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simsun.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/System/Library/Fonts/PingFang.ttc",
]

_font_cache = {}


def _get_font(size=24):
    if not PIL_AVAILABLE:
        return None
    key = size
    if key in _font_cache:
        return _font_cache[key]
    for path in FONT_PATHS:
        if os.path.isfile(path):
            try:
                _font_cache[key] = ImageFont.truetype(path, size, encoding="utf-8")
                return _font_cache[key]
            except Exception:
                continue
    try:
        _font_cache[key] = ImageFont.load_default()
        return _font_cache[key]
    except Exception:
        pass
    return None


def draw_label_cn(frame_bgr, box_xyxy, label, color_bgr, font_size=24):
    """
    在 frame 上於 box 上方繪製標籤（支援中文）。
    box_xyxy = (x1,y1,x2,y2), color_bgr = (B,G,R)。
    """
    x1, y1, x2, y2 = [int(x) for x in box_xyxy]
    top, left = y1, x1
    font = _get_font(font_size)
    if font is None:
        # 無 PIL 或無字體時退回 cv2.putText（中文會變問號）
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        y1_label = max(0, top - th - 12)
        y2_label = top - 4
        x2_label = min(frame_bgr.shape[1], left + tw + 16)
        cv2.rectangle(frame_bgr, (left, y1_label), (x2_label, y2_label), (40, 40, 40), -1)
        cv2.rectangle(frame_bgr, (left, y1_label), (x2_label, y2_label), color_bgr, 1)
        cv2.putText(frame_bgr, label, (left + 8, top - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_bgr, 2, cv2.LINE_AA)
        return

    # 用 PIL 畫文字到小圖再疊回
    padding = 8
    try:
        from PIL import ImageDraw as ID
        d = ID.Draw(Image.new("RGB", (1, 1)))
        if hasattr(d, "textbbox"):
            bbox = d.textbbox((0, 0), label, font=font)
        else:
            bbox = (0, 0, len(str(label)) * font_size, font_size)
    except Exception:
        bbox = (0, 0, max(80, len(str(label)) * font_size), font_size + 4)
    tw = (bbox[2] - bbox[0]) + padding * 2
    th = (bbox[3] - bbox[1]) + padding
    y1_label = max(0, top - th - 4)
    x2_label = min(frame_bgr.shape[1], left + tw)
    if x2_label <= left or y1_label + th > frame_bgr.shape[0]:
        return
    # 背景條
    cv2.rectangle(frame_bgr, (left, y1_label), (x2_label, top - 4), (40, 40, 40), -1)
    cv2.rectangle(frame_bgr, (left, y1_label), (x2_label, top - 4), color_bgr, 1)
    # PIL 畫文字
    patch = frame_bgr[y1_label : top - 4, left:x2_label]
    if patch.size == 0:
        return
    img_pil = Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text((padding, 2), label, font=font, fill=(color_bgr[2], color_bgr[1], color_bgr[0]))
    frame_bgr[y1_label : top - 4, left:x2_label] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
