# 方法二：批次註冊多張照片（一次註冊 iid_1.png ~ iid_16.png）
# 請依實際幼兒名字修改底下 names 串列，然後執行此腳本
# 若某張照片偵測不到臉部，會跳過該張並繼續，最後列出失敗項目
# 同一名字可再執行 main.py register 追加第二、第三張臉（一人多模版，上限見 config.MAX_TEMPLATES_PER_PERSON）

import os
import sys

# 對應 iid_1.png, iid_2.png, ... iid_16.png 的名字（請改成真實姓名）
names = [
    "小孩1", "小孩2", "小孩3", "小孩4", "小孩5", "小孩6",
    "小孩7", "小孩8", "小孩9", "小孩10", "小孩11", "小孩12",
    "小孩13", "小孩14", "小孩15", "小孩16",
]


def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)
    # 直接呼叫註冊模組，方便捕捉單張錯誤
    from face_registry import register_face

    base = os.path.join("data", "registered")
    succeeded = []
    failed = []
    for i, name in enumerate(names, start=1):
        photo = os.path.join(base, f"iid_{i}.png")
        if not os.path.isfile(photo):
            failed.append((photo, "找不到檔案"))
            print(f"跳過（找不到）: {photo}")
            continue
        try:
            register_face(photo, name)
            succeeded.append((name, photo))
            print(f"已註冊: {name} <- iid_{i}.png")
        except ValueError as e:
            failed.append((photo, str(e)))
            print(f"跳過（偵測不到臉部）: iid_{i}.png — {name} — {e}")
        except FileNotFoundError as e:
            failed.append((photo, str(e)))
            print(f"跳過: {photo} — {e}")
    print("---")
    if failed:
        print(f"成功 {len(succeeded)} 筆，失敗 {len(failed)} 筆。失敗項目請換成清晰正面照後，單獨執行：")
        for path, reason in failed:
            print(f"  python main.py register --photo \"{path}\" --name \"<名字>\"")
    else:
        print("全部註冊完成。")
    print("接著執行:")
    print("  python main.py process --video data/input/test2.mp4")
    print("  python main.py build-graph --output-dir data/graph")


if __name__ == "__main__":
    main()
