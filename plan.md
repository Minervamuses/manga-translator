# manga-translator v0.3.2 精簡收尾計畫

本檔取代舊 P0～P5／G0～G5 計畫。目標只剩三件事：修正目前排版退步、完成必要 OCR／翻譯收尾、以既有五頁完成最終驗收。不得再擴張成研究評測、工作流平台或發行供應鏈專案。

## 1. 進度

狀態只用 `done`、`in_progress`、`pending`、`blocked`、`cancelled`。每完成一項直接更新本表，不另建進度系統。

| ID | 狀態 | 工作 | 完成 commit／備註 |
|---|---|---|---|
| F0 | done | exact-ID mapping、safe-no-erase、五頁真實 baseline | 凍結，不再擴充 gate |
| F1 | done | PageDocument／resume／restart 基礎 | 凍結，不再擴充 durable framework |
| T0 | done | RAQM、safe region、ROI atomic renderer 接線 | `cfa96a0` |
| T1 | done | 修正 38 組候選排版的視覺退步 | 使用者於 2026-07-30 核准 38/38 `new_better`，`critical_regression = 0` |
| T2 | in_progress | 新排版設為預設，保留明確 rollback |  |
| O1 | pending | 只修真實頁面可證明的 OCR 問題 |  |
| X1 | pending | 只修 exact mapping 與實際翻譯問題 |  |
| E1 | pending | 環境政策收斂為 Conda＋Poetry |  |
| V1 | pending | 一次性最終驗收 |  |
| OLD | cancelled | 舊 P3/P4/P5、G0～G5、release infrastructure | 不再執行 |

## 2. 硬限制

### 環境

- 只用既有 Conda environment `manga`＋Poetry。
- Conda 管 Python／原生函式庫；Poetry 管 Python dependency 與 `poetry.lock`。
- 保留 `poetry.toml`：`virtualenvs.create=false`、`virtualenvs.in-project=false`。
- 禁止 `venv`、`.venv`、uv、PDM、Pipenv、mamba、micromamba、conda-lock。
- 禁止 `pip install`、`poetry run`、刪除／重建／主動更新 `manga`。
- 禁止自行新增 dependency；確有必要時先停止並回報。
- 命令一律從 `conda run -n manga ...` 進入。

### Git 與產物

- 保留目前 branch、HEAD、未提交 P2-C2 變更與 `benchmarks/visual_v1/` 產物。
- 禁止 `git reset --hard`、`git clean`、丟失現有工作。
- 不建新 branch/worktree，不 merge、push、建 PR／release，除非使用者批准。
- 一個里程碑一個 commit；禁止大量微型 evidence/fix commits。
- 不造 reviewer 資料、不把 mock 當 real、不為過 gate 降門檻。
- 既有大型歷史 evidence 暫留；不得再新增重複 multi-MB profiler trace。

### 範圍

只有能直接修正五頁或日常使用中可重現問題的工作才可做。看到可「順便重構」的地方預設不做；先重用現有函式、套件與已完成元件。

## 3. 測試政策

### 開發中

只跑受影響的 targeted tests，不機械式跑完整 suite。例如排版修改：

```bash
PYTHONPATH=src conda run -n manga python -m pytest -q \
  tests/test_layout_solver.py \
  tests/test_roi_render.py \
  tests/test_safe_region.py \
  tests/test_shaping.py
```

每個里程碑另跑：

```bash
PYTHONPATH=src conda run -n manga python -m compileall -q src tests
conda run -n manga ruff check <本里程碑修改檔案>
git diff --check
```

完整 `pytest` 只跑兩次：T2 正式切換時、V1 最終驗收時。

### 真實模型／provider

- 品質驗收：既有五頁各跑一次。
- 不再使用每頁 `1 cold + 2 warmup + 5 measured` 作品質 gate。
- 只有聲稱改善效能時，才對最多兩頁各做 `1 cold + 1 warm`。
- 未受修改影響的 provider response／人工結果可重用，不因 SHA 改變重跑付費 API。
- API key 可從既有 shell 環境讀取，但不得出現在 log、JSON 或命令輸出。

## 4. T1：修正 38 組候選排版

### 起點

目前 P2-C2 未通過。禁止把 30 組全部填成 A、B 各 4 分。

現有 hard metrics 只證明無缺字、CLREQ hard violation、collision、ROI 外修改及 containment failure；它沒有驗證方向、讀序、文字塊大小、中心、欄數與風格。盲評圖已有直排改橫排、過度壓縮、不自然分欄及低可讀性描邊。

### 行動

1. 先確認實際根因，重點檢查：
   - primary／alternate direction 同時進入搜尋；
   - score 缺少方向與欄數偏離懲罰；
   - `max_lines` 依譯文字數放大；
   - whitespace 使用 glyph alpha pixels 而非 text-block bbox；
   - 白底框 stroke fallback 過度積極。

2. 方向：
   - 已判定直排的 group 預設只搜尋直排；橫排同理。
   - 只有明確標記 ambiguous 且設定允許時才可搜尋 alternate direction。
   - 非 ambiguous group 方向一致率必須 100%。

3. 欄／行與讀序：
   - 從原 mask／既有 accepted layout 推估來源欄／行數。
   - candidate 預設只搜尋來源數量 `-1、0、+1`；必要時最多再放寬一級。
   - 禁止以 `len(text)` 當欄／行上限。
   - 中文直排固定右至左、欄內上至下；橫排固定左至右、行內上至下。

4. 字級與位置：
   - 正常 candidate 字級不得低於可靠來源估計的 90%；否則 overflow／保留原文。
   - 新增 rendered text-block bbox 指標，檢查主軸長度、次軸長度、面積及中心。
   - candidate 明顯縮在角落、比來源／舊版更集中或讀序異常時拒絕。
   - 不再以 glyph alpha pixel 比例單獨代表「填滿空間」。

5. 風格：
   - 白色／淺色對話框預設黑字、無描邊。
   - 只有 fill、stroke、stroke width 都有高信心且對比足夠時才沿用描邊。
   - 深色背景可用白字／描邊，但必須通過 contrast 檢查。
   - 無法可靠判定時使用保守預設，不猜 shadow／特殊效果。

6. 安全：
   - 無可接受 layout 時保留原文，不擦除。
   - T1 不刪 legacy engine。
   - 不為了 38/38 強行接受劣質 candidate。

7. 驗證：
   - 只使用現有五頁、38 groups；不新增 corpus。
   - 開發中只重產受影響 group；收斂後完整重產一次。
   - 保留 exact mapping、missing glyph、clipping、collision、ROI diff、containment、no-erase checks。
   - 新增 orientation、reading order、font floor、text-block bbox、center、contrast checks。
   - 人工覆核每組只記 `new_better`、`tie`、`legacy_better`、`critical_regression` 與可選備註；不用 30×7×2 分數，不建新 gate framework。

### 驗收

- 38/38 安全 checks 通過。
- 非 ambiguous group 方向一致率 100%。
- 無重疊、亂序、明顯過小、角落聚集或低對比描邊。
- `critical_regression = 0`。
- 所有 `legacy_better` 項目均修正、改為安全拒絕，或由使用者明確接受。
- 使用者批准最終 review sheets 後才標 `done`。

Commit：

```text
fix(typography): preserve source orientation and readable text blocks
```

## 5. T2：正式接線

只有 T1 經使用者批准後開始。

### 行動

- 新 RAQM engine 設為預設 production path。
- 舊 engine 本版保留為明確 rollback 選項，不做大規模刪除。
- 舊 engine 不得靜默自動選中。
- 只移除直接相關、已證明不可達的分支與無效 config key。
- 五頁各跑一次真實 end-to-end，結果須與 T1 批准版本一致。
- 執行第一次完整 test suite。

### 驗收

- CLI 預設走新 engine。
- 無 critical regression、空白框或先擦後失敗。
- rollback 可明確啟用且不影響預設路徑。
- compileall、完整 pytest、Ruff、`git diff --check` 通過。

Commit：

```text
feat(typesetting): enable approved RAQM renderer by default
```

## 6. O1：OCR 實證收尾

先用五頁確認：模型是否只初始化一次、完整框與碎片是否重複 OCR、batch 是否真為一次 forward 且順序正確、no-text 是否造成實際錯誤、現有 resume 是否已足夠。

若沒有可重現問題，直接標 `done`，不得為舊計畫的 300＋300 corpus 繼續開發。

確有問題時：

- 使用現有 38 groups，加最多 20～50 個針對失敗類型的 crop。
- 保留 exact ID、batch order、單次初始化、safe-no-erase tests。
- 簡單 heuristic 足夠時，不新增 classifier、概率校準、Brier score 或 title split。
- 只有改 batching 時才做最多兩頁的 `1 cold + 1 warm`；品質不得退、速度至少不變差。
- 不新增 durable store、cache graph 或 stage framework。

有實際修改才 commit：

```text
fix(ocr): resolve measured batching or duplicate-recognition issue
```

## 7. X1：翻譯實證收尾

### 必須保留

- 固定 region/group ID；response 必須完整且無 duplicate／missing／unknown ID。
- mapping 失敗整批拒絕，不得先擦除。
- OpenRouter 只接收 OCR 文字，不傳圖片。
- 來源感知標點、線條、省略號與重複內容清理。
- raw response 可 debug，但不得含 secret。

### 不再擴充

- 不建 200-unit MQM、3-title split。
- 不新增／擴充 entity ledger、translation memory、visual escalation、圖片上傳、targeted repair orchestration。
- 已存在但非必要的元件保持關閉，不為舊 gate 強行接線。

### 驗證

- 使用五頁及最多 30～50 個代表 units。
- prompt／parser 未改時優先重用既有 provider response。
- 真 provider 最多每頁一次。
- 驗收：mapping 100%、無漏譯／重複、台灣繁中品質不退、譯文不再導致嚴重排版退化。

有實際修改才 commit：

```text
fix(translation): keep strict mapping and remove measured output defects
```

## 8. E1：只保留 Conda＋Poetry

功能收尾後執行：

1. 搜尋 `conda-lock` 使用點。
2. 若只由舊計畫／benchmark metadata 引入：
   - 從 `environment.yml` 移除 `conda-lock`；
   - 刪除 repo 的 `conda-lock.yml`；
   - 讓 metadata 對該檔不存在正常處理；
   - 不重寫既有歷史 benchmark JSON。
3. 不從現有 Conda 環境 uninstall；只是不再列為專案要求。
4. 保留 Python、Poetry、FreeType、HarfBuzz、FriBiDi、libraqm。
5. 保留 `pyproject.toml`、`poetry.lock`、`poetry.toml`。

驗收：

```bash
conda run -n manga python -c "import sys; print(sys.executable)"
conda run -n manga poetry env info --executable
conda run -n manga poetry check --lock
conda run -n manga python -m pip check
test ! -d .venv
```

兩個 Python 路徑必須屬於同一個 `manga` 環境。

Commit：

```text
build(env): keep Conda and Poetry as the only project managers
```

## 9. V1：最終驗收

### 完整檢查

```bash
set -euo pipefail
unset VIRTUAL_ENV
export POETRY_VIRTUALENVS_CREATE=false

PYTHONPATH=src conda run -n manga python -m compileall -q src tests
PYTHONPATH=src conda run -n manga python -m pytest -q -p no:cacheprovider
conda run -n manga ruff check .
conda run -n manga poetry check --lock
conda run -n manga python -m pip check
git diff --check
```

### 真實 smoke

- `manga-translate --help`。
- 五個問題頁各跑一次真 detector＋OCR＋provider＋新 renderer。
- 檢查 mapping、no-erase、空白框、字級、方向、位置、欄數、風格、secret。
- 再用一張非 benchmark 自有圖片作 CLI smoke。

### 報告

更新 `VALIDATION_REPORT.md`，只記：final commit、實際命令與結果、六頁簡短結果、已知限制、rollback 方法、取消項目。不要再產生 G0～G5、SBOM、release ZIP、fresh CI environment 或新 evidence schema。

Commit：

```text
test(release): validate v0.3.2 on real manga pages
```

完成後停止並回報 final SHA、測試結果、真實頁面結果及限制。不得自行 merge、push 或建立 release。

## 10. 已取消項目

除非使用者另開任務批准，以下不是本版完成條件：

- G0～G5 gate／validator／全樹 fingerprint refresh。
- 每個小 commit 跑完整 500+ tests。
- 五頁 × 8 次 real benchmark。
- 30 pages／3 titles、300 text＋300 no-text、200 translation units。
- OCR calibration、Brier score、no-text classifier、title-disjoint split。
- MQM、entity ledger、translation memory、visual escalation 擴充。
- 未經 profiler 證明的 ModelRegistry、bounded queues、NMS、adaptive detector。
- conda-lock、fresh CI、wheel smoke、SBOM、license/release archive。
- 多 branch/worktree 的 P0～P5 合併流程。
- 因無關 SHA 變動重跑付費 API 或人工覆核。
