# 批次註冊 data/registered 裡的照片（iid_1.png ~ iid_16.png）
# 請依實際幼兒名字修改底下 $names 陣列
$names = @(
    "小孩1", "小孩2", "小孩3", "小孩4", "小孩5", "小孩6",
    "小孩7", "小孩8", "小孩9", "小孩10", "小孩11", "小孩12",
    "小孩13", "小孩14", "小孩15", "小孩16"
)
$base = "data/registered"
for ($i = 1; $i -le 16; $i++) {
    $photo = "$base/iid_$i.png"
    if (Test-Path $photo) {
        $name = $names[$i - 1]
        python main.py register --photo $photo --name $name
    }
}
Write-Host "註冊完成。接著執行: python main.py process --video data/input/test2.mp4"
