# 僅執行 YOLO 影片處理。獨立新 Python 行程：不在此之前載入 dlib，避免與 PyTorch 載入順序衝突。
# 用法：python run_yolo_isolated.py --video data/input/test2.mp4 --output data/output/out.mp4 [--seconds 5] [--yolo-cpu] ...
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from runtime_hardening import apply as _rt_apply
from runtime_hardening import prepare_cuda_hidden_for_cpu_only_yolo

_rt_apply()


def main():
    import argparse

    p = argparse.ArgumentParser(description="獨立行程：YOLO + 臉部辨識處理影片")
    p.add_argument("--video", "-v", required=True)
    p.add_argument("--output", "-o", required=True)
    p.add_argument("--primary", default=None, help="主要追蹤對象（預設 registry 第一位）")
    p.add_argument("--seconds", "-s", type=float, default=None)
    p.add_argument("--start", type=float, default=None)
    p.add_argument("--yolo-cpu", action="store_true")
    p.add_argument(
        "--yolo-size",
        choices=("n", "s", "m", "l", "x"),
        default=None,
    )
    p.add_argument(
        "--scheme",
        default=None,
        help="與 main.py process 相同：寫入 data/schemes/<名稱>/（須早於載入 yolo_tracker）",
    )
    args = p.parse_args()
    if getattr(args, "scheme", None):
        from config import set_active_scheme

        try:
            set_active_scheme(str(args.scheme).strip())
        except ValueError as e:
            print(f"方案名稱無效：{e}", file=sys.stderr)
            sys.exit(1)
    if args.yolo_cpu:
        prepare_cuda_hidden_for_cpu_only_yolo()
    from yolo_tracker import process_video

    ydev = "cpu" if args.yolo_cpu else None
    primary = args.primary
    if not primary:
        from face_registry import get_registry_encodings

        names, _ = get_registry_encodings()
        if not names:
            print("錯誤：尚未註冊任何人臉。", file=sys.stderr)
            sys.exit(1)
        primary = names[0]
    try:
        r = process_video(
            args.video,
            output_path=args.output,
            primary_name=primary,
            collect_interactions=True,
            model_size=args.yolo_size,
            max_seconds=args.seconds,
            start_seconds=args.start,
            yolo_device=ydev,
        )
    except Exception as e:
        print("處理失敗:", e, file=sys.stderr)
        sys.exit(1)
    print(f"影片已輸出: {r['output_path']}")
    print(f"總幀數: {r['frames']}")
    if r.get("primary_visible_frames") is not None:
        print(f"主要追蹤對象可見幀數: {r['primary_visible_frames']}")
    if r.get("interactions_file"):
        print(f"互動記錄: {r['interactions_file']}")


if __name__ == "__main__":
    main()
