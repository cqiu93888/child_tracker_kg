# 追蹤與掛名流程 - 改進建議（資深工程師視角）

## 一、正確性與穩定性

### 1. 人框 ID 指派改為「全域最佳配對」（已實作）
- **現狀**：逐框貪婪選「當前框與上一幀框」最大 IoU，兩人交叉時容易 A↔B 互換。
- **改法**：用**指派問題**（cost = 1 - IoU）做一對一配對，最大化總 IoU，減少交叉時互換。
- **實作**：`yolo_tracker._assign_curr_to_prev_global` 使用 `scipy.optimize.linear_sum_assignment`；**scipy 為可選**，未安裝則自動退回貪婪邏輯。建議安裝：`pip install scipy`

### 2. 臉↔人框指派也可改為全域最佳
- **現狀**：依 score 排序後貪婪指派，多人並排時仍有機會左右對調。
- **改法**：臉與人框的 score 矩陣做一次 linear_sum_assignment，再寫回 id_to_name / id_to_best_dist / id_to_best_margin。

### 3. 長時間未辨識到臉的軌道
- **現狀**：一旦掛名就一路沿用，即使後續很多幀都偵測不到該軌道的臉。
- **改法**：可選「連續 N 幀未出現臉部匹配則將該軌道名字清回未知」，避免錯名黏太久（需與「一軌道一名字」策略取捨）。

---

## 二、可觀測性與除錯

### 4. 除錯 / 診斷模式
- **現狀**：出錯時難以還原「是哪一幀、哪個 track、距離與 margin 多少」。
- **改法**：`config.DEBUG_TRACKING = True` 時，每 N 幀輸出一筆 log：`frame, our_id, current_name, new_candidate, dist, margin, action`（action 如 first_assign / same / skip），或寫入 JSONL 供事後分析。

### 5. 註冊品質檢查
- **現狀**：註冊照若兩人很像，編碼接近，容易互掛。
- **改法**：`register` 或獨立腳本在寫入前計算所有註冊臉的兩兩 `face_distance`，若有任一对 < 閾值（如 0.5）則警告並列出名字，建議重拍或換照。

---

## 三、效能與參數

### 6. 臉部辨識頻率
- **現狀**：`recognize_every_n = 1` 每幀都做臉部辨識，成本高。
- **改法**：設為 2 或 3，其餘幀沿用上一幀名字；名字會多延續 1～2 幀，換取約 1/2～1/3 的辨識量。

### 7. 設定收斂為少數情境
- **現狀**：門檻與開關很多，不易調。
- **改法**：收斂為 2～3 種情境，例如：
  - **stable**：一軌道一名字、不換、margin 較嚴、適合正式分析；
  - **coverage**：允許修正與強匹配覆蓋、margin 較鬆、適合先求掛上名字再手動修。

---

## 四、程式結構

### 8. 掛名邏輯集中為「狀態機」
- **現狀**：主迴圈內一長串 if/elif，不易單測與說明。
- **改法**：抽出函式 `decide_name_update(current_name, new_name, new_dist, new_margin, frame_index, state) -> (new_name_or_keep, updated_state)`，主迴圈只呼叫並寫回 `track_id_to_name` 與 state，邏輯與註解集中一處。

### 9. 軌道狀態一併清理
- **現狀**：已改為軌道逾時時一併 pop 各類 cache（last_best_dist、override_candidate 等），避免髒狀態。
- **建議**：若日後新增其他 per-track 狀態，一律在「軌道逾時清理」處一併移除。

---

## 五、建議實作優先順序

1. **高**：人框 ID 指派改為 Hungarian（本輪已實作），立即減少交叉互換。
2. **中**：可選的 `DEBUG_TRACKING` 日誌或 JSONL，方便複現與調參。
3. **中**：註冊時或獨立腳本做「兩兩距離警告」。
4. **低**：臉↔人框改 Hungarian、長時間未辨識清回未知、掛名狀態機重構，視需求再做。
