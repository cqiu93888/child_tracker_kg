# 幼兒辨識與關係圖專案 (Child Tracker & Knowledge Graph)

從**一張照片 + 名字**開始，在影片中穩定追蹤該幼兒、在輸出影片上顯示名字，並建立**知識圖譜**與**關係圖**，呈現幼兒之間的友誼與個性推論。

---

## 給組員的快速上手

```bash
# 1. 複製專案
git clone https://github.com/<你的帳號>/child_tracker_kg.git
cd child_tracker_kg

# 2. 建議先建虛擬環境
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # macOS / Linux

# 3. 安裝依賴（face-recognition 需 dlib，Windows 建議用 conda 或預編譯 wheel）
pip install -r requirements.txt

# 4. 啟動網頁介面
python app_gradio.py
```

開啟瀏覽器前往 **http://localhost:7860** 即可使用。
不需要記任何 CLI 指令，所有功能都在網頁介面上操作。

> 若想產生公開連結分享給他人：`python app_gradio.py --share`

---

## 功能

- **臉部註冊**：上傳一張照片並輸入名字，註冊要追蹤的幼兒
- **從影片擷取臉圖**：從影片採樣裁出臉部小圖到 `data/extracted/`，再手動分類後 `register`
- **影片追蹤**：處理影片，穩定追蹤已註冊的幼兒，輸出畫面上顯示名字
- **互動記錄**：自動記錄誰與誰常同框、距離接近（推論互動）
- **知識圖譜**：節點為幼兒與屬性（個性等），邊為關係（朋友、常一起出現）
- **關係圖**：視覺化誰跟誰是朋友、關係強度
- **個性推論**：根據同框對象多樣性、與他人相處時間、是否常獨處等推論個性標籤

---

## 兩種使用方式

### 方式一：Gradio 網頁介面（推薦）

```bash
python app_gradio.py
```

瀏覽器開啟 http://localhost:7860，包含四個頁面：

| 頁面 | 說明 |
|------|------|
| **臉部註冊** | 上傳照片、輸入名字、選方案，一鍵註冊 |
| **影片處理** | 上傳影片、設定秒數與追蹤模式，處理後預覽結果 |
| **建構圖譜** | 選方案後一鍵產生關係圖與知識圖譜 |
| **查看結果** | 瀏覽輸出影片與關係圖 |

### 方式二：命令列 (CLI)

```bash
# 註冊
python main.py register --photo data/registered/小明.jpg --name "小明" --scheme 甲班

# 處理影片
python main.py process --video data/input/教室.mp4 --output data/output/output.mp4 --yolo --yolo-cpu --scheme 甲班

# 建圖
python main.py build-graph --scheme 甲班

# 一次跑完
python main.py run-all --photo data/registered/小明.jpg --name "小明" --video data/input/教室.mp4 --scheme 甲班
```

---

## 方案 (Scheme) 系統

不同班級或測試可以各建一個**方案**，資料完全隔離：

```
data/schemes/
├── 甲班/
│   ├── scheme_config.json   ← 該方案專用參數
│   ├── face_registry.json   ← 註冊庫
│   ├── interactions.json    ← 互動記錄
│   ├── registered/          ← 註冊照片副本
│   ├── output/              ← 輸出影片
│   └── graph/               ← 關係圖 HTML
└── 乙班/
    └── ...
```

使用 `--scheme 甲班` 指定方案（CLI），或在網頁介面的下拉選單選擇。

### 方案參數 (scheme_config.json)

每個方案可以有自己的 `scheme_config.json`，覆寫 `config.py` 中的預設值。範例：

```json
{
  "RECOGNITION_PRESET": "classroom_masked",
  "FACE_ENGINE": "insightface",
  "FACE_MATCH_TOLERANCE": 0.38,
  "FACE_MATCH_MIN_MARGIN": 0.05
}
```

建立範本：`python main.py scheme-config init 甲班`

---

## 臉部辨識引擎

支援兩種引擎，可在 `scheme_config.json` 中透過 `FACE_ENGINE` 切換：

| 引擎 | 設定值 | 特點 |
|------|--------|------|
| **dlib** (預設) | `"dlib"` | 安裝較簡單，128 維特徵向量 |
| **InsightFace (ArcFace)** | `"insightface"` | 512 維特徵向量，口罩/遮擋/側臉場景表現更佳 |

> 切換引擎後，需要重新註冊所有照片（編碼維度不同）。

---

## 目錄結構

```
child_tracker_kg/
├── app_gradio.py         # Gradio 網頁介面
├── main.py               # CLI 主程式
├── config.py             # 設定（含方案系統）
├── face_engine.py        # 臉部辨識引擎抽象層
├── face_registry.py      # 臉部註冊
├── video_tracker.py      # 純臉部追蹤
├── yolo_tracker.py       # YOLO 人體追蹤 + 臉部辨識
├── run_yolo_isolated.py  # YOLO 子行程
├── knowledge_graph.py    # 知識圖譜建構
├── relationship_graph.py # 關係圖視覺化
├── personality.py        # 個性推論
├── appearance_features.py# 外觀特徵輔助
├── extract_faces.py      # 從影片擷取臉圖
├── diagnose_tracking.py  # 追蹤診斷工具
├── api.py                # REST API（本機版）
├── api_cloud.py          # REST API（Render 雲端版，不需 ML 套件）
├── sync_to_cloud.py      # 本機→雲端資料同步腳本
├── requirements.txt
├── requirements-cloud.txt# 雲端版輕量依賴
├── render.yaml           # Render 部署設定
├── README.md
├── docs/                 # 技術文件
├── lib/                  # 前端視覺化庫 (vis.js 等)
└── data/                 # 資料目錄
    ├── input/            # 輸入影片（放在這裡）
    ├── output/           # 輸出影片
    ├── registered/       # 預設註冊照片
    ├── graph/            # 預設圖譜輸出
    └── schemes/          # 各方案獨立資料夾
```

---

## 安裝

### 基本依賴

```bash
pip install -r requirements.txt
```

`face-recognition` 需搭配 dlib，Windows 常見安裝方式：
- 用 conda：`conda install -c conda-forge dlib`
- 或下載預編譯 wheel

### InsightFace（選用）

若方案使用 `FACE_ENGINE: "insightface"`：

```bash
pip install insightface onnxruntime
```

首次使用時會自動下載 ArcFace 模型（約 300 MB）。

---

## 追蹤方式：YOLO vs 純臉部

- **YOLO 追蹤（建議）**：先用 YOLOv8 偵測「人」並用 ByteTrack 維持追蹤 ID，再在每人框內做臉部辨識。追蹤較穩、不易抖動。
- **純臉部追蹤**：每幀只做臉部偵測 + 辨識。

預設使用 YOLO。可在 CLI 用 `--no-yolo` 或在網頁介面取消勾選切換。

---

## 追蹤很差時：先跑診斷

```bash
python diagnose_tracking.py --video data/input/test.mp4 --frame 30 --save data/output/diagnose.png
```

會顯示 YOLO 人框數、臉部偵測數、與註冊名單的匹配距離。依結果調整 `scheme_config.json` 中的門檻。

---

## 從影片擷取臉圖

適合沒有現成照片，想從影片「挖」臉圖再分類的情況：

```bash
python main.py extract-faces --video data/input/教室.mp4
```

擷取後手動分類，再對每張 `register`。

---

## 個性推論

依行為資料做簡單推論（僅供參考）：

- **社交型**：常與多人同框、同框對象多樣性高
- **內向型**：獨處時間較多、同框對象少
- **親密型**：與特定少數人同框時間長、關係邊權重高
- **活躍型**：在畫面中移動或出現的次數多

標籤寫入知識圖譜，並在關係圖中一併顯示。

---

## LINE Bot 雲端部署（Render）

不需要一直開電腦，把 LINE Bot 部署到免費雲端即可 24/7 運作。

### 架構

```
本機（處理影片） ──sync──→ Render 雲端（LINE Bot API）←── LINE 使用者
```

### 一次性設定

1. **註冊 [Render](https://render.com)**（用 GitHub 帳號登入）

2. **在 Render 建立 Web Service**（擇一）：
   - **Blueprint**：在 Render 選「New Blueprint」連結 repo，會讀取根目錄 `render.yaml`（內含 build／start）。
   - **手動 Web Service**：選「New Web Service」連結 repo 時，**不會**自動套用 `render.yaml`；請在服務 **Settings** 手動填：
     - Build Command：`pip install -r requirements-cloud.txt`
     - Start Command：`uvicorn api_cloud:app --host 0.0.0.0 --port $PORT`  
     專案根目錄已加 **`Procfile`**（`web: uvicorn api_cloud:app ...`），可與上述設定雙重保險，避免誤設成 `api:app`（那是另一支 `api.py`，沒有雲端 LINE／分塊同步）。

3. **在 Render 設定環境變數**（Environment → Environment Variables）：

   | 變數名稱 | 值 |
   |----------|----|
   | `LINE_CHANNEL_SECRET` | 你的 LINE Channel Secret |
   | `LINE_CHANNEL_ACCESS_TOKEN` | 你的 LINE Channel Access Token |
   | `LINE_TEACHER_PASSWORD` | 老師密碼（預設 teacher123） |
   | `LINE_PROFESSOR_PASSWORD`（選用） | 另設一組密碼給「教授」身分；未設定則無法以教授登入 |
   | `SYNC_SECRET` | 自訂同步密碼（本機同步時要用） |
   | `LINE_DEFAULT_SCHEME` | 預設方案名稱（如：甲班） |
   | `CLOUD_DATA_DIR`（選用） | 見下方「持久碟」，可避免休眠後資料被清空 |

4. **在 LINE Developers 設定 Webhook URL**：
   ```
   https://your-app.onrender.com/webhook
   ```
   若曾改為其他主機（例如 Fly），請改回上述 **Render 網址** 並重新驗證。

### 分塊 API 仍 404／影片同步失敗？徹底檢查

1. **本機與 GitHub 一致**  
   `git pull` 後，`git log -1 --oneline` 應能看到含 `video-chunk` 或 `deploy` 相關的 commit。

2. **Render 真的在部署這個 repo 的 `main`**  
   Settings → **Build & Deploy**：Repository、Branch（建議 `main`）、**Root Directory**（多數情況留空）。

3. **Start Command 必須是 `api_cloud`**  
   必須為：`uvicorn api_cloud:app --host 0.0.0.0 --port $PORT`  
   若誤設為 `uvicorn api:app`，健康檢查可能仍像雲端 API，但**不會**有分塊上傳等 `api_cloud` 專用路由。

4. **強制重編**  
   **Manual Deploy → Clear build cache & deploy**，等狀態 **Live** 且 Build 無錯。

5. **用瀏覽器或腳本驗證線上版本**  
   - 開 `https://你的服務.onrender.com/`：JSON 裡應有 **`deploy_mark`**（會隨版本變更）與 **`video_chunk_paths`**。若沒有，代表線上仍舊版。  
   - 任一探測網址應回 **`"video_chunk": true`**（GET）：`/api/sync/video-chunk` 或 `/api/sync-video-chunk`。  
   - 本機執行：  
     `python scripts/check_render_deploy.py https://你的服務.onrender.com`  
     若顯示 `[FAIL]`，依腳本提示對照 Render 設定。

6. **暫時無法部署分塊 API 時（不建議，僅過渡）**  
   `sync_to_cloud.py` 在雲端 **404** 分塊路徑時，若影片總大小 ≤ `SYNC_LEGACY_MULTIPART_MAX_MB`（預設 48），會自動改試單次 `POST /api/sync`（雲端須已支援 multipart）。**超過預設大小會直接中止**，請先部署新版；若仍要賭單次上傳可設 `SYNC_FORCE_MULTIPART=1`（大檔極易 **502**）。

### 平時使用

在本機處理完影片、建好圖譜後，執行一行指令把資料推到雲端：

```bash
python sync_to_cloud.py --scheme 甲班 --url https://your-app.onrender.com --secret 你的同步密碼
```

若要讓老師／教授在 LINE 用「**影片**」取得**追蹤輸出影片**連結，需一併上傳本機輸出影片（預設掃描 `data/schemes/<方案>/output`；若為空會改掃 `data/output`，因未加 `--scheme` 時輸出會在那裡）。檔案可能很大，上傳較久：

```bash
python sync_to_cloud.py --scheme 甲班 --url https://your-app.onrender.com --secret 你的同步密碼 --with-videos
```

只上傳單一輸出檔（例如 `data/output/output3.mp4`）：

```bash
python sync_to_cloud.py --scheme 甲班 --url https://your-app.onrender.com --secret 你的同步密碼 --video output3.mp4
```

（影片預設以 **`POST /api/sync/video-chunk`** 分塊上傳；部署的 `api_cloud` 須為最新版。若本機出現 **404**，代表 Render 尚未部署到含此 API 的 commit，請 **Manual Deploy**。部署後可用瀏覽器開 `https://你的網址/api/sync/video-chunk`，應看到 JSON 含 `video_chunk: true`。若網路慢可設 `SYNC_UPLOAD_CHUNK_MB=4`；進階可設 `CLOUD_SYNC_VIDEO_CHUNK_URL` 覆寫端點。）

之後 LINE Bot 就能用最新的圖譜資料回覆使用者了。

### 不想休眠後一直重跑 sync？用 Render 持久碟

免費 Web Service 的檔案系統在**休眠／重啟**後可能清空，`cloud_data` 會不見，LINE 就讀不到圖譜。做法：

1. 在 Render 服務上新增 **Persistent Disk**（持久碟；多為付費方案才有），掛載路徑例如：`/var/data`
2. 環境變數新增 **`CLOUD_DATA_DIR=/var/data`**（與掛載路徑一致）
3. 重新部署後再執行 **一次** `sync_to_cloud.py` 寫入資料；之後休眠醒來，檔案仍會在碟上，**不必因為主機重啟而反覆 sync**

> 本機若**重新建圖、換資料**，仍要再 sync 一次才會更新雲端內容；持久碟只解決「空機重啟資料不見」。

其他做法：改用有內建硬碟的 VPS、或之後把圖譜改存 S3／物件儲存（需改程式）。

### LINE Bot 身份驗證

- 老師輸入密碼 → 可查看完整圖譜、方案列表、追蹤輸出影片連結（須已 `--with-videos` 同步）
- 教授輸入 `LINE_PROFESSOR_PASSWORD`（須先在 Render 設定）→ 權限與老師相同（含「影片」）
- 學生／家長輸入幼兒名字 → 只能查看該幼兒的個人圖譜
- 輸入「登出」可切換身份

> 圖譜與影片連結為**知道網址即可開啟**（與多數輕量部署相同）。若需更嚴格存取控制，需另行加驗證（例如簽名網址）。

---

## 注意事項

- 幼兒資料（照片、影片）屬隱私，**請勿上傳至公開場所**
- `data/` 下的影片和照片已加入 `.gitignore`，不會被 Git 追蹤
- 組員 clone 後需自行準備影片和照片
- 本專案為學術專題用途

---

## 授權與免責

本專案為獨立專案，僅供學術研究使用。幼兒資料請勿外流，僅在本地處理。
