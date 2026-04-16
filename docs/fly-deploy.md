# Fly.io 部署（api_cloud + 持久 Volume）

## 帳務（重要）

Fly.io 目前多數帳號在**建立 App、使用 Volume 前**，必須在後台綁定**信用卡或購買額度**（防濫用；實際用量仍可能落在免費額度內，以官網為準）。若出現：

`We need your payment information to continue!`

請到 Fly Dashboard → **Billing** 完成設定後，再執行 `fly apps create`。

若無法或不願綁卡，可改：**Render 付費 + Persistent Disk**、**Oracle Cloud 免費 VPS**、**院方／自架主機**。

---

## 你遇到的錯誤代表什麼

- `fly apps list` 是空的 → 還沒建立 **App**，要先有 app 才能建 volume。
- `missing an app name` → 專案目錄裡要有 **`fly.toml`**（本 repo 已附範本），或指令加 `-a 應用名`。

## 事前準備

1. 已安裝 `fly` 並執行過 `fly auth login`。
2. 在專案根目錄（有 `fly.toml`、`Dockerfile.cloud`）操作。

## 步驟一：建立 App（擇一）

**做法 A（建議）**：改 `fly.toml` 第一行 `app = "..."` 為你要的名稱（小寫、英數與連字號），然後：

```bash
fly apps create child-tracker-kg
```

（名稱須與 `fly.toml` 的 `app` **完全一致**；若已被占用請換名。）

**做法 B**：在專案目錄執行 `fly launch`，依提示建立 app（可再對照本 repo 的 `fly.toml` 調整）。

## 步驟二：建立 Volume（與 app 同區）

`primary_region` 在 `fly.toml` 預設為 `nrt`（東京），volume 必須同區：

```bash
fly volumes create kg_data --app child-tracker-kg --region nrt --size 1
```

`kg_data` 須與 `fly.toml` 裡 `[[mounts]]` 的 `source` 相同。

## 步驟三：環境變數（Secrets）

```bash
fly secrets set CLOUD_DATA_DIR=/data -a child-tracker-kg
fly secrets set LINE_CHANNEL_SECRET=xxx LINE_CHANNEL_ACCESS_TOKEN=xxx SYNC_SECRET=xxx -a child-tracker-kg
fly secrets set LINE_TEACHER_PASSWORD=teacher123 LINE_DEFAULT_SCHEME=甲班 -a child-tracker-kg
```

（依實際替換；可分批 `fly secrets set`。）

## 步驟四：部署

```bash
fly deploy -a child-tracker-kg
```

成功後網址約為：`https://child-tracker-kg.fly.dev`（依實際 app 名為準）。

## 步驟五：本機同步圖譜

```bash
python sync_to_cloud.py --scheme 甲班 --url https://child-tracker-kg.fly.dev --secret 你的SYNC_SECRET
```

## 注意

- 若 `fly deploy` 報 volume 相關錯誤，請確認 **volume 已建立** 且 **region 與 primary_region 一致**。
- 修改 `fly.toml` 的 `app` 後，請同步使用 `-a 新名稱` 或先 `fly apps create 新名稱`。
