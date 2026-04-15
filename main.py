# 主流程：註冊、處理影片、建立知識圖譜與關係圖
# 避免 PyTorch/ultralytics 與 dlib 等同時使用時的 OpenMP 重複載入錯誤
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from runtime_hardening import apply as _runtime_hardening_apply
from runtime_hardening import prepare_cuda_hidden_for_cpu_only_yolo

_runtime_hardening_apply()

import argparse
import inspect
import json
import subprocess
import sys

from config import (
    ensure_dirs,
    EGO_GRAPH_SUBDIR,
    SCHEMES_PARENT,
    SCHEME_CONFIG_FILENAME,
    USE_YOLO_TRACKER,
    PROJECT_ROOT,
    YOLO_DEVICE,
    YOLO_USE_SUBPROCESS_IN_PROCESS_CMD,
    YOLO_AUTO_FALLBACK_TO_FACE,
    set_active_scheme,
    sanitize_scheme_id,
    output_dir,
    graph_dir,
    scheme_root,
    active_scheme,
    registry_file,
    registered_dir,
    scheme_config_snapshot_for_template,
)
from face_registry import register_face, get_registry_encodings
from relationship_graph import run_build_and_draw


def _maybe_set_scheme(args) -> None:
    sc = getattr(args, "scheme", None)
    if sc is None or str(sc).strip() == "":
        set_active_scheme(None)
        return
    try:
        set_active_scheme(str(sc))
    except ValueError as e:
        print(f"方案名稱無效：{e}")
        sys.exit(1)


def cmd_list_schemes(args):
    """列出 data/schemes/ 下已建立的方案目錄。"""
    if not os.path.isdir(SCHEMES_PARENT):
        print("尚無方案目錄。請用 process / build-graph 加 --scheme 名稱 後會建立 data/schemes/<名稱>/")
        return
    names = sorted(
        d for d in os.listdir(SCHEMES_PARENT) if os.path.isdir(os.path.join(SCHEMES_PARENT, d))
    )
    if not names:
        print("data/schemes/ 下尚無子目錄。")
        return
    mark = f" （目前未指定；請用 --scheme {names[0]} 接同一流程）" if not active_scheme() else ""
    print("已存在的方案（各行一個名稱，建 build-graph 時請用相同 --scheme）：")
    for n in names:
        sp = os.path.join(SCHEMES_PARENT, n)
        print(f"  · {n}")
        print(f"      註冊庫: {os.path.join(sp, 'face_registry.json')}")
        print(f"      註冊照片: {os.path.join(sp, 'registered')}")
        print(f"      interactions: {os.path.join(sp, 'interactions.json')}")
        print(f"      輸出影片: {os.path.join(sp, 'output')}")
        print(f"      圖譜: {os.path.join(sp, 'graph')}")
        scfg = os.path.join(sp, SCHEME_CONFIG_FILENAME)
        hit = "（有參數檔）" if os.path.isfile(scfg) else "（尚無，可 scheme-config init）"
        print(f"      方案參數: {scfg} {hit}")
    if mark:
        print(mark)


def cmd_scheme_config_init(args):
    """在 data/schemes/<名稱>/ 建立 scheme_config.json 範本（與目前專案 config 預設一致，可再手改）。"""
    try:
        name = sanitize_scheme_id(args.name)
    except ValueError as e:
        print(f"方案名稱無效：{e}")
        sys.exit(1)
    root = os.path.join(SCHEMES_PARENT, name)
    path = os.path.join(root, SCHEME_CONFIG_FILENAME)
    if os.path.isfile(path) and not getattr(args, "force", False):
        print(f"已存在: {path}")
        print("若要覆寫請加 --force")
        sys.exit(0)
    os.makedirs(root, exist_ok=True)
    snap = scheme_config_snapshot_for_template()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(snap, f, ensure_ascii=False, indent=2)
    print(f"已建立方案參數範本:\n  {path}")
    print("請刪掉不需覆寫的鍵；之後對該方案執行 register／process／build-graph 加相同 --scheme 即會自動載入。")


def _resolve_register_photo_path(photo: str) -> str:
    """--photo 為相對路徑時：先當前目錄，再專案根目錄（避免未 cd 到專案而找不到檔）。"""
    p = (photo or "").strip().strip('"').strip("'")
    if not p:
        return p
    if os.path.isfile(p):
        return os.path.normpath(os.path.abspath(p))
    rel = p.replace("/", os.sep)
    if rel.startswith(f".{os.sep}"):
        rel = rel[2:]
    cand = os.path.join(PROJECT_ROOT, rel)
    if os.path.isfile(cand):
        return os.path.normpath(os.path.abspath(cand))
    return p


def _prepare_yolo_import(args):
    """在 import yolo_tracker 之前呼叫；CPU 模式需隱藏 CUDA，否則部分環境仍載入 GPU 驅動而崩潰。"""
    _cpu = bool(getattr(args, "yolo_cpu", False)) or (
        YOLO_DEVICE is not None and str(YOLO_DEVICE).strip().lower() == "cpu"
    )
    if _cpu:
        prepare_cuda_hidden_for_cpu_only_yolo()


def _call_process_video(
    process_fn,
    video_path,
    *,
    output_path,
    primary_name,
    collect_interactions,
    max_seconds,
    start_seconds,
    model_size,
    yolo_device=None,
):
    """yolo_tracker.process_video 支援 model_size、yolo_device；video_tracker 不支援，勿傳以防 TypeError。"""
    kwargs = dict(
        output_path=output_path,
        primary_name=primary_name,
        collect_interactions=collect_interactions,
        max_seconds=max_seconds,
        start_seconds=start_seconds,
    )
    sig = inspect.signature(process_fn)
    if "model_size" in sig.parameters:
        kwargs["model_size"] = model_size
    if "yolo_device" in sig.parameters:
        kwargs["yolo_device"] = yolo_device
    return process_fn(video_path, **kwargs)


def _yolo_isolated_argv(
    args, *, output_path: str, primary_name: str, max_seconds, start_seconds
):
    cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "run_yolo_isolated.py"),
        "--video",
        args.video,
        "--output",
        output_path,
        "--primary",
        primary_name,
    ]
    if max_seconds is not None:
        cmd.extend(["--seconds", str(max_seconds)])
    if start_seconds is not None:
        cmd.extend(["--start", str(start_seconds)])
    ycpu = bool(getattr(args, "yolo_cpu", False)) or (
        YOLO_DEVICE is not None and str(YOLO_DEVICE).strip().lower() == "cpu"
    )
    if ycpu:
        cmd.append("--yolo-cpu")
    ys = getattr(args, "yolo_size", None)
    if ys:
        cmd.extend(["--yolo-size", ys])
    sch = getattr(args, "scheme", None)
    if sch:
        cmd.extend(["--scheme", str(sch).strip()])
    return cmd


def _register_cli(photo: str, name: str, *, prefer_cnn: bool, allow_no_face):
    photo_res = _resolve_register_photo_path(photo)
    register_face(
        photo_res,
        name.strip(),
        prefer_cnn=prefer_cnn,
        allow_no_face=allow_no_face,
    )
    return photo_res


def cmd_extract_faces(args):
    from extract_faces import extract_faces_from_video

    ensure_dirs()
    r = extract_faces_from_video(
        args.video,
        out_dir=getattr(args, "out_dir", None),
        every_n_frames=args.every_n,
        max_crops=args.max_crops,
        min_face_area_frac=args.min_face,
        padding=args.padding,
        dedup_distance=args.dedup,
        model=args.model,
        start_seconds=getattr(args, "start", None),
        max_seconds=getattr(args, "seconds", None),
    )
    print("擷取完成。")
    print(f"  輸出目錄: {r['out_dir']}")
    print(f"  已存臉圖: {r['saved']} 張")
    print(f"  採樣幀數: {r['frames_scanned']}")
    print(f"  略過（重複角度）: {r['skipped_duplicate']}  略過（臉太小）: {r['skipped_too_small']}")
    if r.get("write_failed"):
        print(f"  寫入失敗: {r['write_failed']}（若全失敗請改輸出到非 OneDrive 路徑，見 README）")
    print(f"  清單: {r['manifest']}")
    print('下一步：手動分類後執行 python main.py register --photo "圖.jpg" --name "名字"')


def cmd_register(args):
    _maybe_set_scheme(args)
    ensure_dirs()
    if active_scheme():
        print(f"[方案] {active_scheme()}  → 註冊庫: {registry_file()}")
        print(f"        註冊照片將複製至: {registered_dir()}")
    allow_nf = None
    if getattr(args, "strict_face", False):
        allow_nf = False
    elif getattr(args, "allow_no_face", False):
        allow_nf = True
    try:
        used = _register_cli(
            args.photo,
            args.name,
            prefer_cnn=getattr(args, "cnn", False),
            allow_no_face=allow_nf,
        )
    except FileNotFoundError as e:
        print("註冊失敗：找不到照片檔。")
        print(f"  詳情: {e}")
        print(f"  目前工作目錄: {os.getcwd()}")
        print(f"  專案根目錄: {PROJECT_ROOT}")
        print(
            "  請確認路徑正確，或先 cd 到專案資料夾后再下指令；"
            '也可用絕對路徑： --photo "C:\\...\\iid_4_3.png"'
        )
        sys.exit(1)
    except ValueError as e:
        print("註冊失敗：")
        print(e)
        print(
            "\n常見原因：① 戴口罩／側臉／低頭，dlib 偵測不到臉 → 加 --cnn 或換較正面照片；"
            "② 若堅持用此圖 → 加 --allow-no-face（僅外觀模版，辨識較弱）。"
        )
        sys.exit(1)
    except OSError as e:
        print("註冊失敗：無法讀寫檔案（例如 OneDrive 同步鎖定、權限）。")
        print(f"  詳情: {e}")
        sys.exit(1)
    print(f"已註冊: {args.name} <- {used}")


def cmd_process(args):
    _maybe_set_scheme(args)
    ensure_dirs()
    if active_scheme():
        print(f"[方案] {active_scheme()}  → 產物目錄: {scheme_root()}")
    out = args.output or os.path.join(output_dir(), "output.mp4")
    use_yolo = USE_YOLO_TRACKER
    if getattr(args, "no_yolo", False):
        use_yolo = False
    if getattr(args, "yolo", False):
        use_yolo = True

    names, _ = get_registry_encodings()
    if not names:
        print("錯誤：尚未註冊任何人臉，請先對「同一方案」執行 register（須與本指令相同的 --scheme）。")
        if active_scheme():
            print(f"  目前方案: {active_scheme()} ；註冊庫預期為: {registry_file()}")
        sys.exit(1)
    primary = args.primary or names[0]
    max_seconds = getattr(args, "seconds", None)
    start_seconds = getattr(args, "start", None)
    if max_seconds is not None:
        print(f"只處理前 {max_seconds} 秒（--seconds={max_seconds}）")
    if start_seconds is not None:
        print(f"從第 {start_seconds} 秒開始（--start={start_seconds}）")

    yolo_available = False
    if use_yolo:
        try:
            import ultralytics  # noqa: F401

            yolo_available = True
        except ImportError:
            print("未安裝 ultralytics，改用臉部追蹤。若要 YOLO 請執行: pip install ultralytics")

    if use_yolo and yolo_available:
        if YOLO_USE_SUBPROCESS_IN_PROCESS_CMD:
            cmd = _yolo_isolated_argv(
                args,
                output_path=out,
                primary_name=primary,
                max_seconds=max_seconds,
                start_seconds=start_seconds,
            )
            rc = subprocess.run(cmd).returncode
            if rc == 0:
                return
            no_fb = getattr(args, "no_yolo_fallback", False)
            if no_fb or not YOLO_AUTO_FALLBACK_TO_FACE:
                print(f"YOLO 子行程失敗（結束代碼 {rc}）。可加 --no-yolo 或檢查 PyTorch/ultralytics。")
                sys.exit(rc if rc is not None else 1)
            print(
                f"\n[提示] YOLO 子行程結束代碼 {rc}（此環境常見 PyTorch 與 dlib 不相容），"
                "已自動改為「純臉部追蹤」以輸出影片（無人體框／較易飄）。\n"
            )
        else:
            _prepare_yolo_import(args)
            from yolo_tracker import process_video as _yolo_pv

            yolo_dev = "cpu" if getattr(args, "yolo_cpu", False) else None
            result = _call_process_video(
                _yolo_pv,
                args.video,
                output_path=out,
                primary_name=primary,
                collect_interactions=True,
                max_seconds=max_seconds,
                start_seconds=start_seconds,
                model_size=getattr(args, "yolo_size", None),
                yolo_device=yolo_dev,
            )
            print(f"影片已輸出: {result['output_path']}")
            print(f"總幀數: {result['frames']}")
            if result.get("primary_visible_frames") is not None:
                print(f"主要追蹤對象可見幀數: {result['primary_visible_frames']}")
            if result.get("interactions_file"):
                print(f"互動記錄: {result['interactions_file']}")
            return

    from video_tracker import process_video

    result = _call_process_video(
        process_video,
        args.video,
        output_path=out,
        primary_name=primary,
        collect_interactions=True,
        max_seconds=max_seconds,
        start_seconds=start_seconds,
        model_size=None,
        yolo_device=None,
    )
    print(f"影片已輸出: {result['output_path']}")
    print(f"總幀數: {result['frames']}")
    if result.get("primary_visible_frames") is not None:
        print(f"主要追蹤對象可見幀數: {result['primary_visible_frames']}")
    if result.get("interactions_file"):
        print(f"互動記錄: {result['interactions_file']}")


def cmd_build_graph(args):
    _maybe_set_scheme(args)
    ensure_dirs()
    if active_scheme():
        print(f"[方案] {active_scheme()}  → 圖譜目錄: {graph_dir()}")
    out_dir = args.output_dir or graph_dir()
    kg_path, html_path, ego_paths = run_build_and_draw(
        output_dir=out_dir, build_ego=getattr(args, "ego", False)
    )
    print(f"知識圖譜: {kg_path}")
    print(f"關係圖: {html_path}")
    print("請用瀏覽器開啟 relationship_graph.html 查看誰跟誰是朋友與個性標籤。")
    if ego_paths:
        ego_dir = os.path.join(out_dir, EGO_GRAPH_SUBDIR)
        print(f"個人關係圖（每位幼兒一張）共 {len(ego_paths)} 份，目錄: {ego_dir}")


def cmd_run_all(args):
    _maybe_set_scheme(args)
    ensure_dirs()
    if active_scheme():
        print(f"[方案] {active_scheme()}  → 產物目錄: {scheme_root()}")
    allow_nf = None
    if getattr(args, "strict_face", False):
        allow_nf = False
    elif getattr(args, "allow_no_face", False):
        allow_nf = True
    try:
        _register_cli(
            args.photo,
            args.name,
            prefer_cnn=getattr(args, "cnn", False),
            allow_no_face=allow_nf,
        )
    except (FileNotFoundError, ValueError, OSError) as e:
        print("run-all 在註冊步驟失敗:", e)
        sys.exit(1)
    print(f"已註冊: {args.name}")
    out = args.output or os.path.join(output_dir(), "output.mp4")
    use_yolo = getattr(args, "yolo", False) or USE_YOLO_TRACKER
    if getattr(args, "no_yolo", False):
        use_yolo = False
    max_seconds = getattr(args, "seconds", None)
    start_seconds = getattr(args, "start", None)
    yolo_ok = False
    if use_yolo:
        try:
            import ultralytics  # noqa: F401

            yolo_ok = True
        except ImportError:
            print("未安裝 ultralytics，run-all 改用純臉部追蹤。")

    if use_yolo and yolo_ok:
        cmd = _yolo_isolated_argv(
            args,
            output_path=out,
            primary_name=args.name,
            max_seconds=max_seconds,
            start_seconds=start_seconds,
        )
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            if getattr(args, "no_yolo_fallback", False) or not YOLO_AUTO_FALLBACK_TO_FACE:
                print(
                    "run-all：YOLO 子行程失敗。可改用 --no-yolo 或執行: python run_yolo_isolated.py ..."
                )
                sys.exit(rc if rc is not None else 1)
            print(
                "\n[提示] YOLO 子行程失敗，已自動改為純臉部追蹤以完成 run-all。\n"
            )
            from video_tracker import process_video as _pv

            _call_process_video(
                _pv,
                args.video,
                output_path=out,
                primary_name=args.name,
                collect_interactions=True,
                max_seconds=max_seconds,
                start_seconds=start_seconds,
                model_size=None,
                yolo_device=None,
            )
    else:
        from video_tracker import process_video as _pv

        _call_process_video(
            _pv,
            args.video,
            output_path=out,
            primary_name=args.name,
            collect_interactions=True,
            max_seconds=max_seconds,
            start_seconds=start_seconds,
            model_size=None,
            yolo_device=None,
        )
    print("影片處理完成，正在建立圖譜...")
    gdir = args.graph_dir or graph_dir()
    _, _, ego_paths = run_build_and_draw(output_dir=gdir, build_ego=getattr(args, "ego", False))
    if active_scheme():
        print(f"完成。請查看方案目錄：{scheme_root()}（output/、graph/、interactions.json）")
    else:
        print("完成。請查看 data/output/ 與 data/graph/")
    if ego_paths:
        print(f"個人關係圖: {len(ego_paths)} 張於 {os.path.join(gdir, EGO_GRAPH_SUBDIR)}")


def main():
    ensure_dirs()
    parser = argparse.ArgumentParser(description="幼兒辨識、追蹤與關係圖專案")
    sub = parser.add_subparsers(dest="command", required=True)

    # extract-faces：從影片自動裁臉圖，供手動分類後 register
    p_ext = sub.add_parser("extract-faces", help="從影片採樣並自動擷取臉部小圖（存到 data/extracted/）")
    p_ext.add_argument("--video", "-v", required=True, help="輸入影片路徑")
    p_ext.add_argument("--out-dir", "-o", default=None, help="輸出資料夾（預設自動建立於 data/extracted/）")
    p_ext.add_argument("--every-n", type=int, default=20, help="每隔幾幀採樣（預設 20）")
    p_ext.add_argument("--max-crops", type=int, default=400, help="最多存幾張臉（預設 400）")
    p_ext.add_argument(
        "--min-face",
        type=float,
        default=0.0012,
        help="臉框至少占畫面比例（預設 0.0012）",
    )
    p_ext.add_argument("--padding", type=float, default=0.35, help="裁切外扩比例")
    p_ext.add_argument("--dedup", type=float, default=0.22, help="encoding 距離低於此視為重複、不存")
    p_ext.add_argument("--model", choices=["hog", "cnn"], default="hog", help="hog 較快 / cnn 較準較慢")
    p_ext.add_argument("--start", type=float, default=None, metavar="S", help="從第 S 秒開始")
    p_ext.add_argument("--seconds", type=float, default=None, metavar="N", help="只處理 N 秒")
    p_ext.set_defaults(func=cmd_extract_faces)

    # register
    p_register = sub.add_parser("register", help="註冊一張照片與名字")
    p_register.add_argument("--photo", "-p", required=True, help="照片路徑")
    p_register.add_argument("--name", "-n", required=True, help="幼兒名字")
    p_register.add_argument("--cnn", action="store_true", help="偵測臉時優先用 CNN（較慢、較適合小臉／遠景）")
    p_register.add_argument(
        "--allow-no-face",
        action="store_true",
        help="偵測不到臉時改存整張照片的外觀模版（無臉向量；見 config.REGISTER_WITHOUT_FACE_USE_APPEARANCE_ONLY）",
    )
    p_register.add_argument(
        "--strict-face",
        action="store_true",
        help="一定要偵測到臉，否則失敗（覆寫無臉外觀註冊）",
    )
    p_register.add_argument(
        "--scheme",
        default=None,
        metavar="名稱",
        help="方案代號：註冊寫入 data/schemes/<名稱>/face_registry.json，並將照片複製到該目錄下 registered/（與 process、build-graph 名稱須一致）",
    )
    p_register.set_defaults(func=cmd_register)

    # process
    p_process = sub.add_parser("process", help="處理影片、輸出帶名字的影片並收集互動")
    p_process.add_argument("--video", "-v", required=True, help="輸入影片路徑")
    p_process.add_argument("--output", "-o", help="輸出影片路徑")
    p_process.add_argument("--primary", help="主要追蹤對象名字（預設為第一位註冊者）")
    p_process.add_argument("--yolo", action="store_true", help="使用 YOLO 人體追蹤（較穩定，需 pip install ultralytics）")
    p_process.add_argument("--no-yolo", action="store_true", help="不使用 YOLO，改用純臉部追蹤")
    p_process.add_argument(
        "--yolo-cpu",
        action="store_true",
        help="YOLO 強制用 CPU（較慢，常能與 dlib 穩定同窗；亦可在 config 設 YOLO_DEVICE=\"cpu\"）",
    )
    p_process.add_argument(
        "--no-yolo-fallback",
        action="store_true",
        help="YOLO 子行程失敗時不要自動改純臉部追蹤（預設會自動 fallback）",
    )
    p_process.add_argument("--seconds", "-s", type=float, default=None, metavar="N", help="只處理前 N 秒（不指定則全跑）")
    p_process.add_argument("--start", type=float, default=None, metavar="S", help="從第 S 秒開始處理（可只跑後半部，不需 ffmpeg）")
    p_process.add_argument(
        "--yolo-size",
        choices=["n", "s", "m", "l", "x"],
        default=None,
        help="YOLO 人體模型大小（覆寫 config.YOLO_PERSON_MODEL_SIZE；s/m 通常比 n 準但較慢）",
    )
    p_process.add_argument(
        "--scheme",
        default=None,
        metavar="名稱",
        help="方案代號：使用該方案專用之 face_registry 處理影片；互動／輸出／建圖皆在 data/schemes/<名稱>/（須與 register 一致）",
    )
    p_process.set_defaults(func=cmd_process)

    # schemes：列出已有方案目錄
    p_schemes = sub.add_parser(
        "schemes",
        help="列出 data/schemes/ 下已建立的方案（不同班級／測次可各用一個方案資料夾）",
    )
    p_schemes.set_defaults(func=cmd_list_schemes)

    # scheme-config：每方案一份 scheme_config.json（辨識／YOLO／圖譜門檻等）
    p_scfg = sub.add_parser(
        "scheme-config",
        help="方案專用參數：data/schemes/<名稱>/scheme_config.json（鍵名見 config.SCHEME_CONFIG_ALLOWLIST）",
    )
    scfg_sub = p_scfg.add_subparsers(dest="scheme_config_action", required=True)
    p_scfg_init = scfg_sub.add_parser(
        "init",
        help="依目前 config.py 預設產生範本 JSON（再手動刪改要調的項）",
    )
    p_scfg_init.add_argument("name", metavar="名稱", help="與 --scheme 相同")
    p_scfg_init.add_argument(
        "--force",
        action="store_true",
        help="已存在 scheme_config.json 時仍覆寫",
    )
    p_scfg_init.set_defaults(func=cmd_scheme_config_init)

    # build-graph
    p_graph = sub.add_parser("build-graph", help="建立知識圖譜與關係圖")
    p_graph.add_argument("--output-dir", help="圖譜輸出目錄（覆寫方案預設之 graph/）")
    p_graph.add_argument(
        "--ego",
        action="store_true",
        help="另為每位幼兒輸出以該人為中心的子圖（data/graph/ego/ego_*.html，門檻見 config.EGO_GRAPH_*）",
    )
    p_graph.add_argument(
        "--scheme",
        default=None,
        metavar="名稱",
        help="與 process 相同名稱，讀寫該方案下的 interactions.json 與 graph/（請與處理影片時一致）",
    )
    p_graph.set_defaults(func=cmd_build_graph)

    # run-all
    p_all = sub.add_parser("run-all", help="一次執行：註冊 -> 處理影片 -> 建圖")
    p_all.add_argument("--photo", "-p", required=True, help="照片路徑")
    p_all.add_argument("--name", "-n", required=True, help="幼兒名字")
    p_all.add_argument("--video", "-v", required=True, help="輸入影片路徑")
    p_all.add_argument("--output", "-o", help="輸出影片路徑")
    p_all.add_argument("--graph-dir", help="圖譜輸出目錄")
    p_all.add_argument("--yolo", action="store_true", help="使用 YOLO 人體追蹤")
    p_all.add_argument("--no-yolo", action="store_true", help="不使用 YOLO")
    p_all.add_argument(
        "--yolo-cpu",
        action="store_true",
        help="同 process --yolo-cpu：YOLO 用 CPU 以穩定混合 dlib",
    )
    p_all.add_argument(
        "--no-yolo-fallback",
        action="store_true",
        help="同 process --no-yolo-fallback",
    )
    p_all.add_argument("--seconds", "-s", type=float, default=None, metavar="N", help="只處理前 N 秒（不指定則全跑）")
    p_all.add_argument("--start", type=float, default=None, metavar="S", help="從第 S 秒開始處理")
    p_all.add_argument("--cnn", action="store_true", help="註冊照片時優先用 CNN 偵測臉（較慢、較適合難偵測照片）")
    p_all.add_argument("--allow-no-face", action="store_true", help="同 register --allow-no-face")
    p_all.add_argument("--strict-face", action="store_true", help="同 register --strict-face")
    p_all.add_argument(
        "--yolo-size",
        choices=["n", "s", "m", "l", "x"],
        default=None,
        help="YOLO 人體模型大小（覆寫 config.YOLO_PERSON_MODEL_SIZE）",
    )
    p_all.add_argument(
        "--ego",
        action="store_true",
        help="建圖時一併輸出每位幼兒的個人子圖（同 build-graph --ego）",
    )
    p_all.add_argument(
        "--scheme",
        default=None,
        metavar="名稱",
        help="與 process 相同：註冊／影片／圖譜皆歸屬 data/schemes/<名稱>/（完全隔離）",
    )
    p_all.set_defaults(func=cmd_run_all)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
