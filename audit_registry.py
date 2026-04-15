# 稽核 face_registry：列出「符合」的有效模版；不以固定張數當標準
import argparse
import json
import os
import re
import shutil
import numpy as np
import face_recognition
from collections import defaultdict

from config import COMPLIANT_REGISTERED_EXPORT_DIR, registry_file, set_active_scheme
from face_registry import load_registry


def _enc_vec(enc):
    if enc is None:
        return None
    a = np.asarray(enc, dtype=np.float64).ravel()
    if a.size == 0 or np.all(a == 0):
        return None
    return a


def _face_distance(a, b):
    return float(np.linalg.norm(a - b))


_BAD_FS = re.compile(r'[\\/:*?"<>|]')


def _safe_fs_segment(s: str) -> str:
    return _BAD_FS.sub("_", s).strip() or "unnamed"


def export_compliant_photos(compliant, export_dir: str):
    """將符合條件的註冊照複製到 export_dir，並寫入 manifest.json。"""
    os.makedirs(export_dir, exist_ok=True)
    tag_short = {"臉+外觀": "both", "僅臉": "face", "僅外觀": "app"}
    manifest = []
    copied = 0
    for i, name, path, tag in compliant:
        if not path or not os.path.isfile(path):
            continue
        short = tag_short.get(tag, "x")
        base = os.path.basename(path)
        dest_name = _safe_fs_segment(
            f"{i:03d}_{name}_{short}_{base}"
        )
        dest_path = os.path.join(export_dir, dest_name)
        shutil.copy2(path, dest_path)
        copied += 1
        manifest.append(
            {
                "registry_index": i,
                "name": name,
                "tag": tag,
                "source_path": os.path.normpath(path),
                "exported_as": os.path.normpath(dest_path),
            }
        )
    man_path = os.path.join(export_dir, "manifest.json")
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return copied, export_dir, man_path


def main():
    ap = argparse.ArgumentParser(description="稽核 face_registry，可匯出符合的註冊照")
    ap.add_argument(
        "--no-copy",
        action="store_true",
        help="只列印報告，不複製圖片",
    )
    ap.add_argument(
        "--export-dir",
        default=COMPLIANT_REGISTERED_EXPORT_DIR,
        help="符合註冊照匯出目錄（預設：config.COMPLIANT_REGISTERED_EXPORT_DIR）",
    )
    ap.add_argument(
        "--scheme",
        default=None,
        metavar="名稱",
        help="與 main.py 相同：稽核 data/schemes/<名稱>/face_registry.json",
    )
    args = ap.parse_args()
    if getattr(args, "scheme", None):
        try:
            set_active_scheme(str(args.scheme).strip())
        except ValueError as e:
            print(f"方案名稱無效：{e}")
            raise SystemExit(1) from e

    data = load_registry()
    names = data["names"]
    encs = data["encodings"]
    paths = data.get("photo_paths", [])
    apps = data.get("appearances", [])
    n = len(names)
    while len(paths) < n:
        paths.append("")
    while len(apps) < n:
        apps.append(None)

    by_name = defaultdict(list)
    for i in range(n):
        by_name[names[i]].append(i)

    print(f"註冊檔: {registry_file()}")
    print(f"總筆數: {n}，唯一姓名: {len(by_name)}")
    print(
        "「符合」＝檔案存在且可讀，且至少有臉向量或外觀向量其一（不依固定幾張判斷）。\n"
    )

    hard_fail = []
    soft_notes = []

    for i in range(n):
        name = names[i]
        path = paths[i]
        enc = encs[i] if i < len(encs) else None
        app = apps[i]
        vec = _enc_vec(enc)
        problems = []
        notes = []

        if not path:
            problems.append("無 photo_path")
        elif not os.path.isfile(path):
            problems.append("檔案不存在")
        else:
            try:
                img = face_recognition.load_image_file(path)
                h, w = img.shape[:2]
                if min(h, w) < 80:
                    notes.append(f"解析度較小 ({w}x{h})")
                locs = face_recognition.face_locations(img, model="hog")
                has_face_hog = len(locs) > 0
                if vec is None:
                    if app is None:
                        problems.append("無臉向量且無外觀")
                    elif not has_face_hog:
                        notes.append("HOG 未偵測臉（僅外觀模版，可接受）")
                else:
                    if not has_face_hog:
                        notes.append("HOG 未偵測臉（但有註冊臉向量，仍視為有效模版）")
            except Exception as e:
                problems.append(f"讀圖失敗: {e}")

        if problems:
            hard_fail.append((i, name, path, problems))
        elif notes:
            soft_notes.append((i, name, path, notes))

    compliant = []
    for i in range(n):
        if any(h[0] == i for h in hard_fail):
            continue
        vec = _enc_vec(encs[i] if i < len(encs) else None)
        has_app = apps[i] is not None
        if vec is not None and has_app:
            tag = "臉+外觀"
        elif vec is not None:
            tag = "僅臉"
        else:
            tag = "僅外觀"
        compliant.append((i, names[i], paths[i], tag))

    print(f"=== 符合的註冊筆（共 {len(compliant)}）===")
    for i, name, path, tag in compliant:
        tail = os.path.basename(path) if path else ""
        print(f"  #{i:2d} {name:8s} [{tag:6s}] {tail}")

    if args.no_copy:
        print("\n（已指定 --no-copy，未複製圖片）")
    elif compliant:
        n, out_dir, man = export_compliant_photos(compliant, args.export_dir)
        print(f"\n已複製 {n} 張符合註冊照至：\n  {out_dir}\n對照清單：\n  {man}")
    else:
        print("\n無符合筆數，未複製圖片。")

    if hard_fail:
        print(f"\n=== 不符合（硬錯誤，共 {len(hard_fail)}）===")
        for i, name, path, probs in hard_fail:
            tail = os.path.basename(path) if path else "(無路徑)"
            print(f"  #{i:2d} {name:8s} | {'；'.join(probs)}  [{tail}]")
    else:
        print("\n不符合（硬錯誤）：無")

    if soft_notes:
        print(f"\n=== 提醒（仍算符合，共 {len(soft_notes)}）===")
        for i, name, path, notes in soft_notes:
            tail = os.path.basename(path) if path else ""
            print(f"  #{i:2d} {name:8s} | {'；'.join(notes)}  [{tail}]")

    print("\n=== 每人模版筆數（僅統計）===")
    for name in sorted(by_name.keys()):
        c = len(by_name[name])
        ok = sum(1 for ii, nm, _, _ in compliant if nm == name)
        print(f"  {name}: 總 {c} 筆，其中符合 {ok} 筆")

    print("\n=== 同一檔案路徑對應不同姓名 ===", flush=True)
    path_names = defaultdict(set)
    for i in range(n):
        p = paths[i]
        if p:
            path_names[p].add(names[i])
    multi = [(p, s) for p, s in path_names.items() if len(s) > 1]
    if multi:
        for p, s in multi:
            print(f"  {p}")
            print(f"    姓名: {s}")
    else:
        print("  無")

    uniq = list(by_name.keys())
    thresh = 0.42
    print(f"\n=== 不同孩童間臉距離 < {thresh}（易互相誤認）===")
    confuse = []
    for ia in range(len(uniq)):
        for ib in range(ia + 1, len(uniq)):
            na, nb = uniq[ia], uniq[ib]
            best = None
            pair_ij = None
            for ii in by_name[na]:
                va = _enc_vec(encs[ii] if ii < len(encs) else None)
                if va is None:
                    continue
                for jj in by_name[nb]:
                    vb = _enc_vec(encs[jj] if jj < len(encs) else None)
                    if vb is None:
                        continue
                    d = _face_distance(va, vb)
                    if best is None or d < best:
                        best = d
                        pair_ij = (ii, jj)
            if best is not None and best < thresh:
                confuse.append((na, nb, best, pair_ij))
    if confuse:
        for na, nb, d, (ii, jj) in sorted(confuse, key=lambda x: x[2]):
            print(f"  {na} vs {nb}: 距離 {d:.4f}（registry #{ii} vs #{jj}）")
    else:
        print("  無（或僅單人有有效臉向量）")

    print("\n=== 外觀為 null 但有臉向量（可補註冊以啟用外觀輔助）===")
    missing_app = [
        ii
        for ii in range(n)
        if _enc_vec(encs[ii] if ii < len(encs) else None) is not None
        and apps[ii] is None
    ]
    if missing_app:
        for ii in missing_app:
            print(
                f"  #{ii:2d} {names[ii]:8s} "
                f"{os.path.basename(paths[ii]) if paths[ii] else ''}"
            )
    else:
        print("  無")


if __name__ == "__main__":
    main()
