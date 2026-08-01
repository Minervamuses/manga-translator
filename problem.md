# `manga-translator` 兩個分支的方法、workflow 與問題說明

> 分析快照：2026-08-01。這份文件的目標讀者是**看不到 repository**、但需要評估
> 架構、演算法、工程風險與後續優先順序的第三方。文中的檔名、函式名、設定值與
> commit SHA 都刻意保留，讓建議可以落到具體實作，而不只停在概念層。

## 0. 分析範圍、分支身分與證據等級

本文件比較 Git 裡實際存在的兩個 branch：

| 本文件名稱 | 實際 Git branch | 分析 commit | 定位 |
|---|---|---|---|
| `main` | `main` | `ba0e3a46d5d864cca12166dbbbbd674e1c767154` | 原始、單體式 production pipeline |
| `repair` | `repair` | `2953d0401319cb905bee78e3c597b05deeeda43f` | 加入 durable stage、RAQM 排版與可重播狀態的重構版 |

兩者的 merge base 是 `eb606f3aed1f0bc235c0c0d2d426c4a87930c7ae`。相對 merge base，
`main` 只有 1 個獨有 commit，而 `repair` 有 142 個；兩邊最後都加入了內容相同的
`AGENTS.md`，但 Git graph 上 `repair` 不是目前 `main` 的直接後裔。`main..repair` 約有
654 個檔案差異，其中大量是 benchmark 圖片、JSON 與歷史 evidence；若只看 source、
tests、config 與 docs，仍是約 154 個檔案的實質重構。

`repair` branch 的 `VALIDATION_REPORT.md` 內寫的 branch `repair_p0_p4_completion` 與 implementation
HEAD `1e58b46` 是歷史驗證快照，**不是本文件所稱的 branch 身分**。本文件中的
`repair` 一律指 `repair@2953d04`。

問題標示採以下語意：

- **已證實**：可由目前 commit 的 production call path、設定或測試直接確認。
- **驗證缺口**：不能斷言結果一定錯，但現有證據不足以支持 README 或設計宣稱。
- **風險推論**：由資料流或演算法可合理推導，仍需要專門測試或真實樣本重現。

目前工作樹另有未追蹤的 `REVIEW_POSTFIX_LAYOUT_REPORT.log`、
`REVIEW_RENDER_MASK_FIX_REPORT.log`、`build/` 與 `dist/`。兩份 log 沒有 branch、SHA、
完整命令或 committed/rejected group 數；因此本文件沒有把它們當作可追溯的成功證據。

本次分析以兩個 Git refs 的 source/tests/config 與已提交 validation artifacts 為證據；沒有呼叫
付費 OpenRouter API，也沒有把歷史報告改寫成「本次重新執行」的測試結果。

## 1. 專案目的與共同不變條件

專案版本是 0.3.2，目的是把日本漫畫頁面的日文文字轉為台灣繁體中文。輸入是本機
圖片，文字偵測與 OCR 在本機進行；翻譯把 OCR 後的純文字送到 OpenRouter，圖片本身
不應送給翻譯 provider。兩個分支都遵循同一個最重要的產品不變條件：

> OCR、翻譯、遮罩或排版只要不夠可靠，就保留該區域的來源像素；不可先擦除原文，
> 再因為後段失敗而留下空白對話框。

兩者的概念流程相同：

```text
圖片解碼
  → comic-text-detector 找文字區域與 pixel mask
  → manga-ocr 辨識日文
  → OpenRouter 轉成台灣繁中
  → 決定譯文能否在原位置安全排版
  → OpenCV inpaint 擦除「確定會寫回」的原字
  → Pillow 寫入新字
  → 編碼輸出與 debug evidence
```

真正差異在於：`main` 把這些動作串在一個記憶體內頁面流程；`repair` 把它們拆成有
持久狀態、指紋、快取、重播與 typed issue 的 10-stage DAG，並以 RAQM 取代預設的
逐字手排引擎。

## 2. 共通開發、環境與交付規範（只列一次）

以下規範來自兩個 branch 內容相同、且被指定為權威的 `AGENTS.md`。

### 2.1 作業環境與依賴所有權

- 正式 target 是 Linux。
- 在 Windows 工作時必須進 WSL Bash；**所有 Git 指令**，包含 status、branch、commit、
  push，都必須使用 WSL 內的 Git，不能使用 Windows Git。
- Python 目標版本是 3.11。
- Conda 擁有 Python interpreter 與 native runtime；Poetry 擁有 Python packages 與 lock。
- 禁止以 `pip`、`venv` 或 `uv` 取代上述組合，也不得在 repository 建 `.venv`。

全新環境：

```bash
conda env create -f environment.yml
conda activate manga
poetry install --with dev
```

既有環境：

```bash
conda env update -f environment.yml --prune
conda activate manga
poetry install --with dev
```

若較舊 branch 的 README 或 setup 文件與這裡矛盾，以此 Conda + Poetry 規則為準。

### 2.2 專案結構

- application code：`src/manga_translator/`
- 主流程：`src/manga_translator/pipeline.py`
- detection、OCR、translation、inpainting、typesetting：各自模組
- vendored detector：`src/manga_translator/ctd/`
- tests：`tests/test_*.py`，檔名與 source module 對應
- runtime assets：`models/`、`fonts/`
- regression evidence：`samples/`、`validation_samples/`
- local/generated data：`input/`、`output/`、`build/`、`dist/`

### 2.3 標準命令與測試政策

所有命令都應在 `conda activate manga` 後執行：

```bash
poetry run manga-translate doctor --config config.yaml --strict-api-key
poetry run manga-translate run --config config.yaml
poetry run pytest -q
poetry run ruff check .
poetry build
```

單頁 regression 使用：

```bash
poetry run manga-translate test --image input/page.jpg --dump-json
```

不需要翻譯時使用 `detect-only`。測試採 pytest，檔名為 `test_<module>.py`、函式為
`test_<behavior>`；先跑 targeted tests，再跑全套。任何改變 rendered output 的修改都要
提供 `validation_samples/` 的代表性前後證據。未經明確批准，不能執行會付費的 API
integration test。

### 2.4 程式與交付規範

- 四空格縮排、type hints、100 字元 line limit、Ruff defaults。
- module/function 用 `snake_case`，class 用 `PascalCase`，constant 用 `UPPER_SNAKE_CASE`。
- 完成的修改必須 commit，不可只留在 working tree。
- commit 使用 Conventional Commits，例如
  `fix(typesetting): preserve vertical spacing`。
- PR 要說明 problem、approach、tests、configuration impact；視覺變更要附 before/after。
- 不可提交 API key、私人漫畫頁、model cache 或 generated output。

### 2.5 共同 Python/native 套件角色

兩個分支都使用：

| 套件 | 角色 |
|---|---|
| PyTorch | CTD detector 與 manga-ocr inference runtime |
| Transformers | `ViTImageProcessor`、Japanese BERT tokenizer、`VisionEncoderDecoderModel` |
| fugashi / unidic-lite | 日文 tokenizer runtime |
| OpenCV / NumPy | 圖片 I/O、mask morphology、threshold、edge、inpaint、幾何運算 |
| Pillow | 字型載入、glyph raster、文字 layer 合成 |
| Shapely / pyclipper | polygon、IoU、幾何裁切相關操作 |
| httpx | 非同步呼叫 OpenRouter |
| Pydantic / PyYAML | `config.yaml` schema、驗證與載入 |
| Click / Rich | CLI、進度與診斷輸出 |

兩個 branch 都宣告 `torchvision`，但目前 production source 沒有直接 import；vendored NMS 反而
刻意採 pure PyTorch 避免 `torchvision.ops.nms`。因此它是 declared dependency，不宜視為已證實
的直接 runtime call。

`repair` 額外使用 `fontTools`、`regex`、`uniseg`，並依賴 FreeType、HarfBuzz、FriBiDi、
libraqm，讓 Pillow RAQM 能做 Unicode shaping、bidi 與直排；SQLite 使用 Python stdlib。

### 2.6 分析時的關鍵設定快照

以下是兩個 branch 的 checked-in `config.yaml` 與 production constants；未列出的細項仍應以
分析 commit 為準。

| 區塊 | 共同/`main` 值 | `repair` 差異 |
|---|---|---|
| OpenRouter | model `x-ai/grok-4.5`；endpoint `/api/v1/chat/completions`；batch 20；temperature .2；retry temperature 0；content retries 2；timeout 90 s；output/source length ratio 4 | schema 另有 `data_collection=deny`、`zdr=true` defaults，但 production payload 未使用 |
| translation context | `translation_mode=context`、context size 5、`page_context_mode=page`；一般頁 <=6000 chars 且 <=120 items 時整頁送出 | mapping/response 改為 strict IDs + source hashes，dispatch thresholds 相同 |
| detector | `comictextdetector.pt`、CUDA、1024 + 1536、FP16 off、NMS .35、confidence .30、mask .30 | 相同；另有 durable detector identity/issue |
| mask safety | raw support threshold 30、dilate 2；segmentation fallback off；bbox fallback off | 相同 CTD 原則，再新增 safe-region stage |
| grouping | min area 36、thin ratio .015、same-text IoM .55、containment .82、fuzzy similarity .78、group IoM .15、center ratio 1.2 | 大致相同，結果另做 durable reconciliation |
| OCR | adaptive raw/mask + contrast/threshold/region views；一般 .46、短文 .66、fallback .74、agreement .70 | config 新增 pinned model revision、batch 4、max length 300，但目前 `_get_model()` 未讀這些欄位 |
| layout | auto direction；10–180 px；preserve floor .92、normal min .85、hard min .62；bubble expand .72、max 720 px | 預設 engine 從 legacy 改 RAQM；safe containment 與 production candidate policy更嚴 |
| inpaint | hybrid、mask dilation 1、Telea radius 2、只處理 translated groups | RAQM path 改為 per-ROI atomic transaction |
| fonts/assets | `Iansui-Regular.ttf` + `NotoSansCJKtc-Regular.otf`；CTD `.pt`；manga-ocr weights | 新 font metadata/role 系統存在，但 production layout 固定 neutral sans |

## 3. 高階比較

| 面向 | `main@ba0e3a4` | `repair@2953d04` |
|---|---|---|
| production orchestration | 單體 `process_single_page()` | 固定 10-stage DAG + `StageRunner` |
| 中間狀態 | process memory；debug JSON/圖為輔 | SQLite + content-addressed artifacts + `PageDocument` checkpoint |
| resume/replay | 無 | stage cache、resume、force-stage、provider raw replay、offline encode replay |
| region identity | 排序後的短 ID，重跑可能改變 | UUID identity + revision hash + merge/split lineage |
| OCR | 多視圖 heuristic quality，逐 group | production 仍沿用多視圖 group OCR；runtime 增加 token metrics/batching，但未完整接線 |
| reading order | 全頁中心座標排序 | production 仍是全頁中心排序；panel-aware 模組尚未接線 |
| translation mapping | 寬鬆 JSON/編號 fallback，位置對應 | deterministic request/item IDs、source SHA、strict exact mapping、raw response artifact |
| typesetting | Pillow 逐 glyph 手排 | 預設 Pillow RAQM + Unicode/CLREQ-like breaker + raster verification |
| safe area/style | 亮色近純色矩形擴張、固定字體風格 | edge barrier/flood-fill safe region + 原圖 style fingerprint |
| render transaction | 全部 active groups 先 inpaint，再逐 group render | 每個 ROI copy→inpaint→render→驗證，最後才 commit |
| page failure | 原圖 decode 後重編碼到正常檔名；CLI 可仍為 0 | 原始 bytes exact-copy 到 `output/failed/`；batch status/exit 可阻擋 |
| 主要優勢 | 簡單、容易追 call stack；固定 visual fixture 的 writeback 高 | 可追溯、可續跑、mapping 與 rollback 安全性高 |
| 目前最大問題 | 設定/文件矛盾、弱 provenance、多個 correctness/config 缺口 | 最後一份可追溯的歷史 live smoke 只有 2/76 groups 寫回，current-head 比率未知；不少新增子系統並未接入 production |

---

## 4. Branch：`main`（`ba0e3a4`）

### 4.1 分支定位與安裝現況

`main` 是較小、較直接的版本。`pyproject.toml` 使用
`setuptools.build_meta`，沒有 `poetry.lock` 或 `poetry.toml`。主要版本約束是
`torch==2.7.1`、`torchvision==0.22.1`、`transformers>=4.45,<6`、
`numpy>=1.26,<2`、`opencv-python-headless==4.9.0.80`、`Pillow>=10.2,<12`。

這個 branch 的 `environment.yml` 把多數 Python 套件交給 Conda，並在 `pip:` subsection
安裝 CUDA 12.8 的 torch/torchvision、OpenCV 與 Transformers。README 又明寫「不需要
Poetry」並要求 `python -m pip install -e .`。因此實際 setup 文件與第 2 節的權威規範
互相衝突；這不是風格問題，而是乾淨環境能否重建的直接問題。

### 4.2 入口、設定與資料結構

console script 在 `pyproject.toml` 將 `manga-translate` 指到
`manga_translator.cli:main`。`cli.run()` 以 Click 解析參數，呼叫
`AppConfig.from_yaml()`；Pydantic v2 驗證設定，所有相對路徑以 YAML 所在目錄為基準，
Rich 顯示進度。

核心資料結構：

- `detector.TextRegion`：axis-aligned bbox、直/橫方向、來源 pass、字級提示、bbox-local mask。
- `detector.TextGroup`：member region IDs、union mask、OCR candidates、heuristic confidence、
  translation、status、skip reason 與 layout debug 欄位。
- `detector.DetectionResult`：raw/post regions、groups、raw/refined masks、CTD blocks。
- `ocr.OCRCandidate` / `OCRResult`：raw/normalized text、view source、quality。
- `typesetter.OriginalTextGeometry`：來源字形 bbox、估計字級、欄/行數、gap、字距。
- `typesetter.TextLayoutPlan`：方向、bbox、font size、分欄/分行、步距、理論 block bbox、fits。
- `translator.TranslationValidation`：`valid` 與 issue strings。

`process_single_page()` 最後只回傳
`tuple[np.ndarray, list[TextRegion], list[str], list[str]]`；它不是一份能完整重建決策歷程的
domain document。

### 4.3 實際 production call chain

```text
manga_translator.cli:main
  → cli.run()
  → AppConfig.from_yaml()
  → run_pipeline()
     → initialize_ocr_model()              # batch preflight，位於 page try/except 外
     → process_single_page()               # 每一頁
        → read_image()
        → detect_text_regions()
        → initialize_ocr_model()           # 有 groups 時的 singleton/no-op preflight
        → ocr_group_detailed()             # 每個 group
        → _merge_duplicate_groups()
        → _refresh_group_order()
        → _translate_groups()
        → _merge_translation_duplicates()
        → _resolve_render_collisions()
        → _preflight_layout_plans()
        → inpaint_regions()
        → render_text_into_group()          # 每個通過的 group
        → dump_debug_artifacts()
     → write_image()                       # 每頁回到 run_pipeline 後輸出
```

production 中「先排版、後擦除」的核心順序可濃縮成：

```python
layout_plans = _preflight_layout_plans(original, groups, regions_by_id, config)
detection.groups = groups
inpainted = inpaint_regions(original, detection, config.inpainting)

renderable = [
    group for group in groups
    if group.translation_valid
    and group.translation.strip()
    and group.id in layout_plans
]
for group in renderable:
    result = render_text_into_group(..., layout_plan=layout_plans[group.id])
```

這個順序是 `main` 最重要、也確實有接入 production 的安全措施。

### 4.4 Detection：CTD、多解析度與 conservative mask

`detect_text_regions()` 使用 vendored comic-text-detector。底層是 PyTorch + OpenCV：

1. `ctd.inference.TextDetector.__call__()` 對頁面 letterbox/normalize。
2. YOLO-like head 經 `non_max_suppression()` 找 text block。
3. DB line map 經 `SegDetectorRepresenter` 轉成文字線 polygon。
4. `group_output()` 把 line/block 組合，再以 `REFINEMASK_ANNOTATION` refine mask。
5. 預設跑 1024 primary pass，加 1536 extra pass；bbox/mask 跨 pass 合併。
6. `postprocess_regions()` 以面積、thin ratio 過濾；`_build_groups()` 用 union-find 分組。

為避免 refinement 擴到臉或線稿，`_conservative_text_mask()` 不直接相信 refined mask，
而是要求像素仍在 thresholded raw segmentation 的鄰域內：

```python
_, support = cv2.threshold(raw_mask, cfg.raw_support_threshold, 255, cv2.THRESH_BINARY)
support = cv2.dilate(support, kernel, iterations=1)
refined_binary = (refined_mask > 0).astype(np.uint8) * 255
return cv2.bitwise_and(refined_binary, support)
```

預設 `raw_support_threshold=30`、support dilation 2 px；segmentation component fallback 有
面積、density、component 數等 gate，但預設關閉。換言之，此 branch 偏向漏翻也不願把
不確定背景擦掉。

grouping 的 `_should_group()` 主要使用 bbox intersection-over-minimum、bbox containment、中心距離、
主方向與 cross-axis 距離，只在特定 nested case 使用 mask containment。OCR 後
`_merge_duplicate_groups()` 再以 bbox IoM/containment、mask IoU/containment、substring、coverage
與 `difflib.SequenceMatcher` 合併多尺度重複框。

### 4.5 OCR：manga-ocr 多視圖 ensemble

`manga_ocr_runtime.MangaOcrRuntime` 使用：

- `transformers.ViTImageProcessor`
- Japanese BERT slow tokenizer（依賴 fugashi/unidic-lite）
- `VisionEncoderDecoderModel`
- Torch CUDA/MPS/CPU 自動選擇
- PIL image 作為模型輸入

模型 ID 固定為 `kha-white/manga-ocr-base`。`ocr_group_detailed()` 的做法是：

1. bbox 依 `crop_padding_ratio=.08`、至少 4 px 外擴。
2. 小 crop 以 Lanczos 放大，短邊目標 160、上限 3 倍。
3. 先辨識 raw crop 與 mask-isolated crop。
4. 若候選低分、互相分歧或 group 含多 region，再加 CLAHE + unsharp、Otsu threshold，
   並逐一辨識 constituent regions。
5. `_combine_region_candidates()` 把 whole-group 與 leaf columns 視為替代假設，避免同一句
   被重複串接。
6. 內容 hash 的 process-global bounded insertion-order cache 最多 4096 筆；hit 不更新順序，
   滿額時刪除最早四分之一，因此不是嚴格 LRU。

`OCRCandidate.quality` 不是模型 probability；它只由日文/CJK/字母/數字/符號比例、長度、
useful ratio 與重複模式組成。不同 view 的 agreement 是 `_select_best_candidate()` 的額外 ranking
因子，並另用於 fallback acceptance gate。`assess_ocr_result()` 的一般門檻為 .46、極短文字 .66；
只來自 mask fallback 的候選要 .74、至少 2 個 kana/CJK 字元，且 view agreement 至少 .70。
即使 OCR 通過，group 沒有非空 pixel mask 仍會以 `missing_text_mask` 拒絕。

### 4.6 Reading order、翻譯與 mapping

`_refresh_group_order()` 的日文直排順序只是全頁 `(-center_x, center_y)`；auto 模式先看
全頁 vertical group 比例是否至少 0.5，再在直排/橫排兩種中心排序中二選一。沒有 panel、
bubble graph 或人工 order override。

只把 accepted OCR 純文字與 glossary 放進 prompt，不上傳圖片。一般頁面若不超過 6000
characters/120 groups，`_request_translations()` 優先呼叫 `translate_page()`；否則才使用
context-window 或 batch。HTTP 層是 `httpx.AsyncClient`，對 408、425、429、5xx 與 network
error 最多 5 次 exponential backoff。

OpenRouter payload 實際只有：

```python
return {
    "model": cfg.model,
    "messages": [{"role": "user", "content": prompt}],
    "temperature": cfg.retry_temperature if retry else cfg.temperature,
    "max_tokens": max_tokens,
}
```

prompt 要求台灣繁中、固定 `T0000` 類 ID 與 JSON；parser 同時接受 dict、list、純字串、
編號行及位置 fallback。`sanitize_translation_text()` 做 NFKC、控制字/markdown/mojibake 清理，
移除原文沒有的省略號、線條型字元與長音誤譯破折號，並折疊明顯的長重複。
`validate_translation()` 檢查 empty、mojibake、長度、標點比例、mostly-kana、來源原樣回傳。
`content_retries=2` 的名稱看似兩次，但 single-item repair loop 使用
`range(content_retries + 1)`；初次 batch/page response 失敗後，production 最多可再送出 3 次
single-item repair request。

### 4.7 排版、inpaint 與 render

`typesetter.py` 是約 1,500 行的 Pillow 手排引擎。它從原圖與 local mask 推估
`OriginalTextGeometry`，包含字形 bbox、原字級、欄/行數與間距；亮且近純色的 bubble 才由
`_safe_background_bbox()` 向外擴張。

直排 `_plan_vertical()` 窮舉字級與欄數，以 `_balanced_chunks()` 將 Unicode codepoints
近似等長分欄，再調 character/column spacing。橫排 `_plan_horizontal()` 做 wrap、tracking、
line spacing。`_choose_layout_candidate()` 只要有任何候選達原估計字級的 92%，就不會選更小
候選；正常 lower floor 設為 85%，hard floor 62%。`_preflight_layout_plans()` 使用理論 block
bbox 檢查互撞，必要時再嘗試不擴 bubble 的 compact plan，仍不安全就保留日文。

`inpaint_regions()` 預設 hybrid：只處理 `translation_valid` groups，以 group mask dilation 1，
平坦背景依 ring median/std/dominant color 決定是否吸收最多 3 px 的 anti-aliased 邊緣，再用
`cv2.inpaint(..., INPAINT_TELEA)` 半徑 2；也支援 Navier-Stokes。bbox fallback 預設關閉。

`render_text_into_group()` 用 Pillow `ImageFont`/`ImageDraw` 在 local RGBA patch 畫字，再 alpha
compose 回 OpenCV BGR。缺字判斷靠 `.notdef` bitmap signature，先嘗試主字體，再用繁中 Noto
fallback，最後才以方框/問號/中點代替。

### 4.8 Batch failure 與輸出

`run_pipeline()` 在 batch 前先初始化一次 OCR model，而且這個 preflight 位於逐頁 try/except
之外；model/dependency/weights 初始化失敗會直接中止整批，不會觸發逐頁原圖 fallback。進入
page loop 後，每頁例外才會讀回原圖並以相同正常檔名寫到 output，避免整本缺頁；但函式只
累計 `failed_pages` 與印摘要，不 raise。圖片 I/O 用 `np.fromfile + cv2.imdecode` 與
`cv2.imencode + tofile` 支援 Unicode 路徑。

### 4.9 `main` 的問題清單

#### P0/P1：可直接影響正確性、可操作性或安全契約

1. **[已證實] 安裝規範互相矛盾。** `AGENTS.md` 要 Conda + Poetry、禁止 pip；README、
   guide 與 `environment.yml` 卻採 pip，build backend 是 setuptools，沒有 Poetry lock。
   `[project.optional-dependencies].dev` 也不是 Poetry dependency group，
   `poetry install --with dev` 未必能按文件語意工作。第三方無法只照 repo 文件重建唯一環境。

2. **[已證實] CLI exit code 會把失敗表示成成功。** 無輸入、`test` 找不到圖、
   `detect-only` 解碼失敗與 batch 中有 failed pages，都可能只印訊息後正常 return。CI 或批次
   orchestrator 看到 exit 0，無法知道頁面其實只是保留原圖。

3. **[已證實] 失敗保留不是 byte-exact。** 失敗頁先 decode 再 encode 到正常輸出檔名；JPEG
   會再次有損壓縮，且檔名本身不能區分成功翻譯與失敗 fallback。

4. **[已證實] CTD confidence 被丟失。** `group_output()` 取得 `conf` 卻沒存入
   `TextBlock`，而 `TextBlock.__init__` 固定 `self.prob = 1`；後續 `TextRegion.confidence` 與
   manifest 幾乎永遠是 1.0，不能用來排序或校準 detector reliability。

5. **[已證實] `detection.half=true` 的 `.pt` path 有 dtype 缺口。** wrapper 會把 input 轉
   FP16，但建立 `TextDetBase` 時未把 `half` 傳給 model；可能形成 FP16 input + FP32 weight。
   預設 false 因而沒有觸發，但公開設定缺少測試與可靠實作。

6. **[已證實] 多個公開設定無效或語意不符。** `balance_columns`、`min_gap_px`、
   `use_font_metrics` 沒有 production read；直排永遠 balanced。`clip_render=true` 建的是全白
   clip mask，不是 group mask；`render_scope=group_mask` 也沒有真正 glyph clipping。
   `text_color: white` 等非 `auto` 字串會被當黑色，只有 RGB tuple 才真正指定色彩。

7. **[已證實] 85% 字級 floor 可被 absolute max 繞過。** `_font_bounds()` 先以
   `font_size_max` 截斷。如果來源估計 240 px、max 180，候選 180 px（75%）仍可被接受，而
   不是依 `min_font_scale=.85` 拒絕。

8. **[風險推論] plan 與實際 glyph 的幾何不一致。** 直排用主字體「國」估所有 glyph
   cell；實畫可切 fallback font。橫排 wrap 先用主字體，後段才量 fallback；outline 也未納入
   fit/collision。可能出現 plan 判定 fits，實畫卻越界或碰撞。

9. **[已證實/風險推論] reading order 不理解漫畫分鏡。** code 可證實只有全頁中心座標、
   沒有 panel model；跨 panel 可能錯序並污染 page-context 翻譯，是尚待專門 corpus 量測的風險。

10. **[風險推論] union-find 有 transitive bridge。** grouping、OCR dedup 都是 pairwise
    union；A-B、B-C 分別通過就會把其實分離的 A/C 合成一群，缺少 component-level bubble
    或 panel 約束。錯誤 accepted group 的 union mask 仍可能一起被擦，fail-safe 只保護明確
    rejection，不能保護「錯誤但通過」的 hypothesis。

11. **[已證實/風險推論] OCR confidence 名稱誤導。** 分數是字元腳本 heuristic，不是
    token/model probability；依公式，純 Latin 候選的最高值約 .38，低於一般 acceptance .46，
    所以英文 SFX 有系統性被拒風險；看似 CJK 的線稿 hallucination 可能高分則仍需實測。
    模型 revision 未鎖定，也不能由 config 控制 model/device/offline mode。

12. **[已證實] OCR 沒有真正 batch。** `ocr_regions_batch()` 仍逐張 loop；raw、mask、
    contrast、threshold 與 leaf-region views 依序呼叫 generate。雙尺度 detector 與大量 SFX
    會放大延遲，也沒有 OOM 後縮 batch/降 CPU 的 production fallback。

13. **[已證實/風險推論] 翻譯驗證不是語意或繁中驗證。** code 可證實沒有繁簡、角色一致性
    或一般語意/hallucination gate；因此漢字很多的未翻日文、簡體或語意錯誤存在通過風險，但
    需要標註語料測量實際比率。整頁 API 失敗不降級逐句，而是整頁留日文；沒有 durable
    response cache，重跑可能再次付費且產生不同結果。

    同一條資料路徑還有 mapping 風險：response parser 接受 dict/list/純字串/編號行與位置
    fallback，duplicate ID 可被後值覆寫，missing item 可落成空字串；request 沒有 durable
    source hash，group ID 又會在重新 grouping/sorting 後改變。它比 `repair` 的 exact item/source
    mapping 更容易把 provider 格式錯誤誤當成合法結果。

14. **[已證實] 設定選項優先順序容易誤導。** 一般頁面會先走 page mode，所以使用者把
    `translation_mode=batch`、`batch_size` 或 `context_size` 改掉，對 <=6000 chars/120 groups
    的頁面仍不生效。

#### P1/P2：視覺品質、可重現性與維護性

15. **[已證實] CTD 的豐富結構在 adapter 被丟棄。** vendored `TextBlock` 原有 polygon
    lines、angle、language、fg/bg/style；`_blocks_to_regions()` 只保留 axis-aligned bbox、
    vertical 與 font hint。後段因此無法重現旋轉/曲線 SFX、斜排、原字色、字重或描邊。

16. **[已證實] 直排不是完整 CJK typography。** 它按 codepoint 等長切欄，只對少數 ASCII
    括號做 vertical map；沒有 CLREQ 禁則、標點懸掛、grapheme/單詞保護、tate-chu-yoko 或
    glyph rotation。一般對白可用，美術字與英數混排是明確品質上限。

17. **[風險推論] safe-region 與 collision heuristic 有雙向誤差。** 只有亮、近純色 bubble
    能向外擴張；深色、漸層、網點背景常被拒。collision 用 axis-aligned 理論 block 矩形，
    可能因透明留白 false reject，也可能漏掉 fallback bearing/outline 的真碰撞。

18. **[已證實/風險推論] mask fallback 預設 off 且 refined mask 必須有 raw support。** 這有助於
    不擦線稿；code 可證實 raw segmentation 漏掉的像素不能由 refinement 補回，因此可能漏翻或
    保留日文，並在其他 groups 成功寫回時形成中日混排風險。

19. **[風險推論] multi-pass mask provenance 不夠精確。** 各 pass safe mask 先 OR 成全頁
    union，再裁給每個 region；一個 bbox 內屬於另一 pass/鄰近候選的像素可能混入 local mask。

20. **[已證實] 缺 mask group 仍先花 OCR 成本。** `ocr_group_detailed()` 完成後才檢查
    `missing_text_mask`；batch 甚至在任何 detection 前先載整個 HF OCR model。全白書頁也依賴
    OCR weights 可用，而且 batch preflight 在逐頁 try/except 外，初始化失敗會中止整批而不是
    逐頁寫出原圖 fallback output。

21. **[已證實] I/O 只保留像素，不保留媒體語意。** `IMREAD_COLOR` 會丟 PNG/WebP alpha；
    EXIF orientation、ICC profile 與 metadata 不保留，JPEG 一律重新壓縮。

22. **[已證實] debug evidence 不足以精確 replay。** duplicate loser 不在 final manifest；
    manifest 沒有完整 config snapshot、model/package revision、input checksum、safe/inpaint mask。
    `prep_manual` 的 `*_inpainted.png` 與 `*_blanked.png` 實際寫同一陣列；compact fallback 的
    layout mode 也可能被 render 覆寫而失去 provenance。整個流程也沒有 SQLite job state、
    stage fingerprint、resume/force-stage 或 raw provider response artifact；process 中斷只能重跑。

23. **[已證實] Pydantic extra 預設忽略。** YAML key 拼錯不報錯，與 dead config field 疊加
    後很難辨別調參是否真的生效。`api_key` 又是 required field；若整個 YAML key 刪除，即使
    `OPENROUTER_API_KEY` 已存在仍先 validation fail。

24. **[已證實] build artifact 不自含。** setuptools 只 package `src/manga_translator*`；根目錄
    CTD model、fonts、config、glossary 與 scripts 不在 wheel，換目錄安裝會找不到 default assets。

25. **[已證實/治理問題] 供應鏈與授權記錄不足。** detector 以 `torch.load()` 讀 pickle，
    download script 不驗 SHA-256/signature；repo 沒有完整 LICENSE/third-party notices，卻含
    vendored CTD、模型與兩套字體。vendored code 還有 import-time 全域 suppression warnings。

26. **[已證實] repository hygiene 與規範不一致。** `.gitignore` 未忽略 `build/`、`dist/`；
    tracked `config.yaml` 接受 inline API key，若直接編輯會有誤提交秘密的風險，雖然 guide 已建議
    使用環境變數。glossary 註解說會在翻譯前替換，實際只把命中項放 prompt，並非硬性術語保證。

### 4.10 `main` 的測試與現有證據

`VALIDATION_REPORT.md` 記錄 compileall 與 67 tests passed。Git tree 的 67 個 pytest test
functions 主要是 unit/synthetic，覆蓋 mask、bbox fallback opt-in、OCR 初始化、whole/leaf
替代、去重、標點清理、字級保留、layout rejection 與 block collision。

報告另用 5 張實際問題頁、固定譯文做視覺檢查，共 38 groups：38/38 產生 layout plan、字級
比例 0.987–1.029、沒有 collision 或 rectangle fallback。這證明 legacy layout 在那組固定
boundaries/text 上可工作，但有三個重要限制：

- 翻譯文字是固定的，沒有測 OpenRouter 語意與不穩定性。
- 報告明載沒有完整 Hugging Face weights；OCR runtime 用 fake backend，不是真模型推論。
- 原始 input 被 Git ignore，生成 evidence 的完整 script 也不在 tree；第三方不能只靠版本庫
  重跑同一份結果。

67-test 自動測試集沒有真 CTD forward；另有上述歷史 validation report 聲稱使用真
`comictextdetector.pt` 與真 input，但缺完整 script/ignored inputs，第三方無法由 Git tree 獨立
重跑。自動測試同樣沒有真 manga-ocr、OpenRouter transport/backoff、完整
`process_single_page()` E2E、panel order、FP16、alpha/metadata 或 fallback glyph 越界案例；
也沒有 CI workflow 可以證明 `main@ba0e3a4` 在乾淨 Conda + Poetry 環境持續通過。

---

## 5. Branch：`repair`（`2953d04`）

### 5.1 分支定位與安裝現況

`repair` 保留 `main` 的 CTD、group OCR、翻譯與部分 legacy typesetter 邏輯，但在外層建立
durable pipeline，並新增 RAQM typography、safe-region、style、identity/mapping contracts、
benchmark 與 inspection CLI。

此 branch 使用 `poetry-core` build backend，有 `poetry.lock`、`poetry.toml`，且關閉 Poetry
virtualenv。`environment.yml` 只由 Conda 安裝 Python 3.11、Poetry 2.2.1、FreeType、HarfBuzz、
FriBiDi、libraqm；Python packages 交給 Poetry。這一點與共同規範相符。

Python dependencies 進一步固定 `torch==2.7.1+cu128`、`torchvision==0.22.1+cu128`、
`transformers==5.14.1`，並加入 `fontTools`、`regex`、`uniseg`；optional eval group 使用
`scikit-learn==1.9.0`。pytest 預設排除 `model_integration`、`api_integration`、`gpu` markers。

`repair` 的 `doctor` 也比 `main` 更接近部署 preflight：它檢查 OpenCV、OCR package/version、
Shapely/pyclipper、uniseg、Pillow RAQM 與 FreeType/HarfBuzz/FriBiDi/libraqm、durable state 目錄、
detector model、fonts、glossary 及 API key；`--strict-api-key` 可把 placeholder/missing key 當失敗。

### 5.2 10-stage durable DAG

固定 DAG 定義在 `src/manga_translator/stages/runner.py`：

```python
STAGE_DAG = {
    StageName.SOURCE: (),
    StageName.DETECT: (StageName.SOURCE,),
    StageName.STYLE: (StageName.SOURCE, StageName.DETECT),
    StageName.SAFE_REGION: (StageName.SOURCE, StageName.DETECT),
    StageName.OCR: (StageName.SOURCE, StageName.DETECT),
    StageName.ORDER: (StageName.SOURCE, StageName.DETECT),
    StageName.TRANSLATE: (StageName.OCR, StageName.ORDER),
    StageName.LAYOUT: (StageName.TRANSLATE, StageName.STYLE, StageName.SAFE_REGION),
    StageName.INPAINT_RENDER: (
        StageName.SOURCE, StageName.DETECT, StageName.TRANSLATE, StageName.LAYOUT
    ),
    StageName.ENCODE: (StageName.INPAINT_RENDER,),
}
```

依賴關係可讀成：

```text
SOURCE ──→ DETECT ─┬─→ STYLE ───────┐
                   ├─→ SAFE_REGION ─┼─→ LAYOUT ─────────────┐
                   ├─→ OCR ─────┐   │                       │
                   └─→ ORDER ───┴─→ TRANSLATE ──────────────┼─→ INPAINT_RENDER → ENCODE
SOURCE ──────────────────────────────────────────────────────┘
DETECT ──────────────────────────────────────────────────────┘
```

`pipeline._build_pipeline_stage_runners()` 建立每個 stage 的 production function；
`process_single_page_staged()` 交給 `StageRunner` 依 topological order 執行。stage contract 以
typed media type 與 artifact 數量檢查輸入/輸出。

### 5.3 Durable state、cache、resume 與 replay

預設 state 在 `output/.manga-translator/`：

- `storage.JobStore`：SQLite `jobs.sqlite3`，記 job/page/stage run、fingerprint、artifact refs、
  page document、issues 與 provider lease。
- `storage.ArtifactStore`：SHA-256 content-addressed 檔案；temp write、`fsync`、atomic replace；
  read 時再驗 hash。
- `domain.PageDocument`：Pydantic strict/frozen schema，保存 source hash、region identity/revision、
  group OCR、translation、layout records、stage records、issues、entity/override slots。

stage fingerprint 包含 stage code identity、指定 config keys、upstream artifact hashes、模型、字型
與部分 package version。相同 fingerprint 可直接取 cached outputs；`--resume` 接續未完成頁，
`--force-stage` 使指定 stage 與 downstream 失效。每個 stage 完成後 checkpoint `PageDocument`。

`StageRunner` 用 page-run lease 與 heartbeat 避免同頁併發重做，也用 provider-response lease
避免兩個 worker 對同一 translation request 重複付費。stage 失敗時不登記該 attempt outputs，
`StageFailureContext` 保留已完成 ancestors。CLI `inspect` 可查看 document/stages/cache/issues，
`replay` 可在無 model/network 下由 encoded artifact 重建 output，`cache gc` 手動清除孤兒 artifacts。

### 5.4 Durable identity 與 geometry reconciliation

`PageDocument` 的 source identity 來自原始 bytes SHA，不只檔名。region 有持久 UUID identity 與
revision SHA；偵測重跑時 reconciliation 使用 Shapely polygon IoU、mask IoU、bbox/center 距離、
crop perceptual hash 等加權分數。明確 match 沿用 identity；模糊 match 產生新 identity 與 issue；
merge/split 記 lineage。這讓 audit/document 層能追蹤 region，不再只剩 `g000` 類位置 ID；但它
沒有完全讓 production group/request identity 對排序或 detector geometry 變動保持穩定。

原因是 `_build_translation_request()` 仍在 reconciliation-aware `PageDocument` 之外，以當次 legacy
`group.region_ids` + bbox 建 `_mapping_region_key()`；ORDER 也仍把 group 重編成 `gNNN`。request
hash 又包含 units 順序。因此 durable region reconciliation 已接線，但 translation/group identity
仍可能隨 grouping、geometry 或 order 改變。

translation 又建立七段 mapping chain：region/revision → group OCR → request item → raw response →
validated translation → layout plan → render outcome。若 item、source SHA 或 group identity 對不上，
系統 fail closed，而不是依 list position 猜測。

### 5.5 各 stage 的實際方法

#### SOURCE

讀取原始 bytes、計 SHA/media type、OpenCV decode，並把原始檔本身存入 artifact store。失敗頁
可直接 exact-copy source bytes，而不是重新壓縮。

#### DETECT

沿用 vendored CTD + PyTorch/OpenCV/NumPy：1024 primary + 1536 extra、YOLO NMS、DB line
representer、refine mask 與 raw-supported conservative mask。預設 FP32、mask fallback off。
設定指定的 CUDA/MPS 不可用時會選 CPU並產生 typed issue；CUDA runtime error 有本頁 CPU retry，
MPS runtime error 沒有同等 retry path。grouping 仍是以 IoM/containment/center/direction 做
union-find。

#### STYLE

`style.extract.extract_style_fingerprint()` 從原圖與 source mask 的 LAB/灰階統計推估 fill、
stroke、shadow、background、ink density、roundness、stroke variation 與 angle 等 evidence；它本身
不輸出 font role。深色背景只有在 confidence 足夠時保留裝飾，否則採高對比保守樣式。render
目標對比約以 WCAG 4.5 為安全門檻。

#### SAFE_REGION

`typography.safe_region.build_safe_region()` 對 ROI 做 grayscale blur、Sobel/Canny edge barrier，
把 detector text seed 與 line polygon 當起點，保護鄰近其他文字，再以 flood-fill 找可用背景。
confidence 結合 mask support、色彩、edge、geometry；低於 .48 時 builder 雖會縮回來源附近，
production `_preflight_raqm_layout_plans()` 隨後仍會直接拒絕 `safe.confidence < .48`，所以這個
fallback 只留下診斷 artifact，不會進 render。最後 erosion 2 px，但保留 detector seed，輸出
binary safe mask、distance/protected mask 與 confidence artifact。

#### OCR

`manga_ocr_runtime.MangaOcrRuntime` 固定模型
`kha-white/manga-ocr-base@aa6573bd10b0d446cbf622e29c3e084914df9741`，使用 Transformers
5.14.1、greedy generation、`max_length=300`。runtime 能計 token log probability、entropy、
margin、truncation，也會在 CUDA OOM 時把 micro-batch 減半。

但 production stage 實際仍逐 group 呼叫 `ocr_group_detailed()`，使用 raw、mask isolation、
CLAHE/unsharp、Otsu、constituent-region views 與 heuristic acceptance；不是把整頁所有 groups
交給 `stages.ocr.PageOCRStager`。換言之，runtime 有 batch primitive，但 production 只在單一
group 的 views 內做有限 micro-batch。

#### ORDER

production 仍呼叫 `_refresh_group_order()`，本質是 `main` 的全頁中心座標排序。repository 雖有
`reading_order.resolve_reading_order`、`order.panels.detect_panel_candidates` 與 override model，
目前 pipeline 沒有呼叫它們。

#### TRANSLATE

`pipeline._build_translation_request()` 以 page ID、sorted region keys/bbox/source hashes 產生
deterministic request ID；每個 item 是 request-scoped 的 `request-id:T0000`。units 順序是 request
hash 與 positional suffix 的一部分，因此重新排序會產生不同 item IDs，不能稱為跨 reorder
不變。response envelope 必須
只有預期欄位，item IDs 必須 exact、不可 missing/duplicate/unknown，source SHA 也必須相同。

小頁仍走 page mode，否則 context/batch。production 由 `translator.py` 的
`translate_page_mapped()`、`translate_with_context_mapped()`、`translate_batch_mapped()` 透過
`httpx.AsyncClient` 呼叫 OpenRouter；transport 最多重試 5 次。每次 raw HTTP response 先成為
content-addressed artifact，再解析與驗證，因此能 audit/replay provider 結果。

#### LAYOUT

預設 `typesetting.engine=raqm`。production preflight 建立嚴格 `LayoutRequest`：只允許來源方向、
不允許 alternate direction、font 候選目前只有 `FontRole.NEUTRAL_SANS`、最小字級比例 .85、
最多來源 line count + 1、tracking 固定 0、safe-mask containment 預設 .995，並把其他 group
視為 occupied mask。

`typography.solver.solve_layout()` 使用 `regex` grapheme、`uniseg`/CLREQ-like legal break、
Pillow `ImageFont.Layout.RAQM`（底層 FreeType/HarfBuzz/FriBiDi/libraqm），以 deterministic beam
search（上限 4096）枚舉字級、行/欄與 gap。每個 candidate 都實際 shape/raster，再檢查 missing
glyph、clipping、safe containment、neighbor collision、原幾何比例、center/angle、line count；
通過才保存 plan hash 與 shaped runs。

#### INPAINT_RENDER

`stages.render.render_page_atomic()` 只複製整頁一次，但每個 ROI 是獨立 transaction：

```python
original_roi = page_bgr[y1:y2, x1:x2].copy()
try:
    inpainted = inpaint_roi(original_roi, inpaint_mask, ...)
    # conditional residual-source check → render shaped layer → post-containment validation
except Exception as exc:
    return AtomicRenderOutcome(committed=False, reason=str(exc))
page_bgr[y1:y2, x1:x2] = committed_roi  # 最後一步才改 working page
```

任何 shape mismatch、空 mask、plan hash 不一致、glyph overflow 或 post-render containment 失敗，
都 rollback 該 ROI。high-contrast residual gate 只有在 `source_text_bbox` 與 `background_rgb` 都可用
時執行；背景估計不足時不能宣稱已檢查 residual source。這是 per-ROI atomic，不是「整頁所有
ROI 全成或全敗」。

#### ENCODE

成功頁以 OpenCV encode 寫 artifact，再由 output materialization 落地。真正 blocking failure 則把
原始 bytes exact-copy 到 `output/failed/<stem>.source-preserved.<ext>`，並移除容易誤認為成功的
正常 output。

### 5.6 Production 已接線與尚未接線的邊界

這個區分對第三方分析非常重要。repository 內「有 class / 有 tests」不等於 run command 會用。

| 子系統 | repository 有實作 | `manga-translate run` 目前狀態 |
|---|---|---|
| durable 10-stage runner / store / replay | 有 | **已接線** |
| region revision / strict mapping / raw response artifact | 有 | **已接線** |
| RAQM solver / safe region / atomic ROI render | 有 | **已接線，且為預設** |
| `stages.ocr.PageOCRStager` 頁級 batch、view cache | 有 | **未接線**；production 用 legacy group OCR |
| OCR calibration framework `ocr_confidence.py` | 有 | **未接線，且尚無 fitted artifact**；production gate 仍用 heuristic quality |
| panel detection / panel-aware reading order / manual order override | 有 | **未接線** |
| `translation.client.StructuredTranslationClient` | 有 | **未接線**；production 用 `translator.py` |
| response JSON schema / job-level HTTP connection reuse | 新 client 有 | **未接線** |
| EntityLedger / translation memory / visual escalation / repair coordinator | 有或有設計 | **關閉/未接線** |
| 動態 font-role selection `resolve_role()`（serif/rounded/handwritten 等） | 有 | **未接線**；production 有用 `FontResolver.from_paths()` 做 neutral/fallback 字型解析，但候選 role 固定 neutral sans |

`git grep` 的具體結果也能證明邊界：`StructuredTranslationClient` 只由其模組與 tests 使用；
`PageOCRStager` 只由 `stages/ocr.py` 與 tests 使用，`pipeline.py` 都沒有 import/call。

### 5.7 Failure/status 語意

`repair` 會把 OCR/API/schema 等 blocking failure 反映在 `PageResult`/`BatchResult`，run command 在
未使用 `--allow-partial` 時可 exit 1。可是 `layout_rejected` 與
`layout_collision_rejected` 被明確列為 non-blocking；一頁即使沒有任何 ROI commit，也可能是
`status="succeeded"`。因此「batch succeeded / failed pages 0」只表示沒有 blocking exception，
不能當成「翻譯已寫回」證據。可靠報表至少還要分開列 detected、OCR accepted、translated、
layout accepted、committed、rolled back、source-preserved 數量。

### 5.8 `repair` 的問題清單

#### P0：目前產品效果與 production/architecture divergence

1. **[歷史證據/驗證缺口] 最後一份可追溯的 live production smoke 只有 2/76 groups
   （約 2.6%）。** 這是歷史 implementation `1e58b46` 的結果：前五頁共有 59 個 live groups，
   只有 `0188:g017` 與 `caption4:g001` commit；第六頁 17 groups 全部 no-op。多數被 layout、
   OCR、collision 或 safe-confidence gate 拒絕。其後三個 commits 直接修改 RAQM request、safe
   seed 與 solver，所以 `repair@2953d04` 的實際比率未知；能確定的是 current HEAD 缺少等價重跑，
   不能把 2/76 當作 current 數字，也不能假定問題已消失。

2. **[歷史證據/驗證缺口] fixed visual benchmark 與 live detector distribution 脫節。** 同一份
   歷史驗證中，固定五頁/38 groups 的 visual corpus 是 38/38 `new_better`，但 fixed boundaries/text
   不等同 production detector 的 group size、OCR 長度與 safe region。不能把 38/38 外推為
   current live pipeline 可用率。

3. **[已證實] 多個高階模組存在但 production 沒用。** panel order、PageOCRStager、OCR
   calibration framework、StructuredTranslationClient、EntityLedger、memory、visual escalation與動態
   `resolve_role()` 都是 dormant。production 仍有用 `FontResolver.from_paths()` 處理 neutral/fallback，
   未接線的是 style-driven role 選擇。這些邊界造成 architecture 文件、tests 與真正 call path
   分裂，維護者容易在錯的 subsystem 修 bug，或誤認設定已生效。

#### P1：設定、provider contract、cache 與診斷正確性

4. **[已證實] OCR config 不控制 OCR runtime。** `OCRConfig` 公開 `model_id`、`revision`、
   `batch_size`、`max_length`，但 production `_get_model()` 是：

   ```python
   def _get_model() -> MangaOcrRuntime:
       ...
       model = MangaOcrRuntime()
   ```

   沒有傳任何上述欄位。改 config 會改 stage fingerprint，實際 model/runtime 卻仍是 defaults；
   這同時是 no-op 設定與 cache identity 語意錯誤。

5. **[已證實] runtime 產生的 token evidence 被 production document 丟掉。** runtime 能提供
   token IDs/logprobs/entropy/margin/truncation，但 `_document_from_page_result()` 只保存最後文字與
   heuristic score，confidence calibration 模組又沒接線。OCR benchmark manifest 目前是
   0/300 crops、0/300 no-text、0/3 titles，沒有可支持 threshold 的人工 calibration corpus。

6. **[已證實] production OpenRouter payload 沒有落實公開隱私/schema 設定。** 實際 payload：

   ```python
   return {
       "model": cfg.model,
       "messages": [{"role": "user", "content": prompt}],
       "temperature": cfg.retry_temperature if retry else cfg.temperature,
       "max_tokens": max_tokens,
   }
   ```

   `OpenRouterConfig.data_collection="deny"`、`zdr=True` 沒送出，也沒有
   `response_format` JSON schema。這些能力存在於未接線的 `translation/client.py`，不能用來
   描述 production 的 privacy guarantee。production 是「prompt 要求 JSON + 收到後 strict parse」，
   不是 provider-side schema enforcement。

7. **[已證實] strict mapping 的 fail-closed 粒度較大。** exact IDs/source SHA 是重要進步，
   但 malformed root、missing/duplicate/unknown ID 在單句 repair 前就可能讓整批 translation stage
   失敗。應由第三方評估要維持 whole-request atomic，或隔離合法 items 後只重試壞 item。

8. **[已證實] 每頁反覆建立 event loop/client。** pipeline 每頁以 `asyncio.run` 進 translation，
   call 內新建 `AsyncClient`；未接線的 structured client 才有較長生命週期的 connection reuse。

9. **[已證實] sanitizer 行為改變，但 README/test 名稱仍描述舊行為。** `repair` 的 production
   code 是：

   ```python
   def sanitize_translation_text(text: str, source: str | None = None) -> str:
       del source
       return normalize_display_text(text)
   ```

   它會保留 `||你好丨`、`你好...`、`謝謝指導——！` 與長重複句；tests 也明確期待保留，
   但部分 test 名稱與 README 仍宣稱會移除 separators、額外省略號/破折號與重複。這是確定的
   code/doc mismatch；是否造成 0211 類「多線條寫回」是尚未被 live render 驗證的 regression
   風險，不能直接宣稱已重現。

10. **[已證實] production reading order 仍不是 panel-aware。** 新 panel/order modules 未接線，
    因而 durable pipeline 只是把舊的全頁中心排序結果保存得更可靠，沒有修正語境順序本身。

11. **[已證實] style evidence 沒有驅動字體家族。** layout request 永遠只傳
    `FontChoice(FontRole.NEUTRAL_SANS)`；`FontResolver.from_paths()` 確實會解析 neutral/fallback
    字型，但 resolver 雖支援多種 roles，production 沒有 classifier/selection 把 roundness/stroke
    variation 轉成 handwritten/serif/rounded role。STYLE 是 per-region 抽樣，group render 又只取
    `group.region_ids` 中第一個有 style 的 region，沒有 group-level 聚合或衝突處理。style 目前
    主要影響 fill/stroke/shadow/angle，而非完整原字體重現。

12. **[已證實] stage fingerprint 依賴不完整。** adapters 查 package 名
    `opencv-python`，實際安裝是 `opencv-python-headless`，可能永久記為 `missing`；STYLE config keys
    為空且 dependency mapping 未列其 cv2/numpy，LAYOUT 只 fingerprint Pillow，卻也使用
    NumPy/OpenCV/uniseg/regex/native RAQM stack。升級依賴後可能錯誤命中舊 cache。

13. **[已證實] `LayoutOverflow:shaping_failed` 原因過載。** validation 明載這個 reason 可能
    同時代表 tracking policy、safe mask、font floor、geometry constraint 或真 shaping error。
    它無法指出歷史 66 個 layout rejects 的細部 bottleneck；其餘 8 個 no-op 已可分成 4 collision、
    3 OCR、1 safe-confidence，不能把全部 74 個都歸因 shaping。直接調 RAQM 很可能治錯問題。

#### P1/P2：驗證、儲存、媒體 fidelity 與效能

14. **[驗證缺口] current HEAD 沒有完整綠燈證據。** 歷史 report 在 `1e58b46` 跑出
    516 passed、2 failed、1 skipped、2 deselected；兩個 fail 是舊 environment fingerprint assertion，
    後續改成 skip 並有 targeted tests。其後還有 `4b78fcd`、`a1938c6`、`d59a3de` 等 layout/
    safe-region/performance 修改，但沒有找到 `repair@2953d04` 的完整 suite + live visual 重跑。
    不能斷言 current suite 一定失敗，也不能宣稱 current HEAD 已全綠。

15. **[已證實] README 與 validation 敘事落後。** README 仍稱 67 tests、固定五頁 38 groups
    全部排版成功；真正 repair suite 已 500+ tests，而最後一份歷史 live production smoke 只有
    2/59 writeback。`VALIDATION_REPORT.md` 的 branch/SHA 也不是目前 branch/head。

16. **[已證實] OCR/translation 人工品質 gates 沒有語料。** 歷史 gate artifacts 的 OCR
    crops/no-text 與 human translation corpus 是 0/300、0/300、0/200，titles 0/3；舊 G2/G3/G4
    JSON 顯示 blocked，後續 plan 選擇取消這些擴充 gate，而不是補完語料。因此 unit tests 多不等於
    模型品質已校準。

17. **[已證實] durable replay 會額外保存完整漫畫與 provider response。** SOURCE bytes、stage
    states、masks、PNG、raw responses 與 SQLite 都落在 `output/.manga-translator/`。這是可重播的必要
    代價，但沒有自動 retention policy，只有手動 `cache gc`。「圖片不送 provider」不等於
    「本機不額外保存圖片」；磁碟成長、備份、敏感頁刪除與存取權限都需要政策。

18. **[已證實] evidence 體積使 repository 很重。** `repair` 約 705 tracked files，包含大量
    benchmark 圖片與可達數萬/十多萬行的 profiler JSON。這有助 audit，但 source review、clone、
    diff 與長期維護成本高，應區分最小可重跑 fixture 與可外置的 generated evidence。

19. **[已證實] 成功但 0 commit 的 JPEG 仍重新編碼。** 只有 blocking failure 走 source bytes
    exact-copy；`status=succeeded` 的 no-op 頁仍經 OpenCV decode/encode。第六頁 validation 即使
    0 render，也有 mean absolute diff 0.0479、max diff 3。EXIF/ICC/一般 metadata 同樣未保存。

20. **[風險推論] grayscale/BGRA compatibility 未完整驗證。** source decoder 可接受 L/BGRA，
    但 CTD、style、render 多處假設 BGR 三 channel。需要真實 alpha/grayscale fixtures，而不是只看
    schema 可表達這些格式。

21. **[已證實] 效能優化仍未落到主要熱點。** detector 預設雙尺度 FP32；已有 5 頁 FP16 parity
    evidence（box ratio 1、mean IoU 約 .9927、mask IoU .9985、small recall 1）卻未啟用 half。
    OCR 逐 group、translation 每頁重建 client，group/dedup/collision 多處 O(n²)。

22. **[已證實] per-ROI atomic 不等於整頁 transaction。** 較早 ROI 可 commit、較晚 ROI 可
    rollback；這可能是合理的 partial-success policy，但 API/report 必須明確，不能把 page
    `succeeded` 解讀為所有 group 成功。

23. **[已證實] 兩套 production path 增加維護面積。** pipeline 同時 import legacy
    `plan_text_layout`/`render_text_into_group` 與 RAQM path，translation 又有 legacy production
    client和新 structured dormant client；重複 contract、normalization、retry 與 tests 容易漂移。

### 5.9 `repair` 的驗證證據

歷史 `VALIDATION_REPORT.md` 的可證實結果：

| 類型 | 結果 | 能證明什麼 | 不能證明什麼 |
|---|---|---|---|
| full pytest at historical `1e58b46` | 516 passed / 2 failed / 1 skipped / 2 deselected | 大量 unit/contracts 有覆蓋 | current HEAD 全綠、真模型/付費 API 品質 |
| targeted baseline | 13 passed / 2 skipped | 被 archive 的 fingerprint assertions 已隔離 | 後續全部改動無 regression |
| historical fixed visual V1 | 38/38 `new_better`, critical 0 | 固定 crop/boundary/text 的 RAQM 輸出較佳 | current live detector/OCR group distribution |
| historical live production 前五頁（`1e58b46`） | 2/59 group commit | 當時 rollback/safety gate 有效 | current-head 自動翻譯完成率 |
| historical 第六頁 nonbenchmark | 0/17 commit | no-op 可安全完成 stage | 輸出 byte-identical；實際已有 JPEG 微小差異 |

歷史 report 的前五頁詳細狀態：

| 頁面 | live groups | committed | 主要 rejection |
|---|---:|---:|---|
| `0188` | 20 | 1 | 多數 layout，另有 OCR/collision |
| `0211` | 9 | 0 | 8 layout、1 OCR |
| `caption3` | 9 | 0 | 9 layout |
| `caption4` | 8 | 1 | 其餘 layout |
| `caption5` | 13 | 0 | 12 layout、1 safe confidence |
| 額外 nonbenchmark 頁 | 17 | 0 | layout/collision/OCR |

這組歷史數據顯示 durable safety gate 確實能保守 rollback，但當時的 production utility 很低；
三個後續 layout/safe-region commits 使 current-head 數字未知。對 `repair@2953d04` 能下的精確
結論是：工程安全性與可稽核性大幅提升，但尚無 current live E2E 證據支持可接受的 utility。

---

## 6. 兩個方法的真正取捨

### 6.1 `main` 解決了什麼

- call path 短，對小型個人專案較容易理解與修改。
- conservative pixel mask、OCR/translation gates、擦除前 layout preflight 已形成基本 fail-safe。
- 固定 5 頁/38 groups 的 legacy layout evidence 顯示，對已知 group/text 可維持接近原字級。
- source-aware sanitizer 會主動清除已知線條、長音破折號、額外省略號與重複輸出。

它沒有解決的是：可重現環境、持久 identity、provider replay、可靠 exit/status、真模型 quality
calibration、panel order、完整 CJK typography、media metadata 與發布/授權治理。

### 6.2 `repair` 解決了什麼

- stage cache、resume、force-stage、checkpoint 與 offline replay。
- content hash、region revision、strict request/item/source mapping 與 raw provider evidence。
- RAQM shaping、Unicode/CLREQ-like breaking、safe-region、style sampling、actual raster verification。
- per-ROI rollback、typed issues、blocking failure source exact-copy 與 CLI status。

它尚未解決的是：live group 的可排率/寫回率、panel-aware order、production OCR calibration、
structured provider/privacy config 接線、font-role 接線、cache dependency completeness、no-op media
fidelity 與 current-head E2E validation。它還新增了本機資料 retention 與系統複雜度成本。

### 6.3 不能只用「較多 tests」選 branch

`main` 的 67 tests 少，但 fixed fixture 的 writeback 高；`repair` 有 500+ tests、contract 與安全
invariant 更完整，而最後一份可追溯的歷史 live sample 只有 2/76 commit，current-head 比率未知。
這不表示應退回 `main`，也不表示 `repair` 架構錯誤；它表示目前測試分布主要證明「不會錯擦、
狀態可追」，沒有證明「能在 production
detector/OCR distribution 下完成足夠多翻譯」。選擇與合併策略必須同時使用：

- correctness/safety：錯擦、mapping 錯位、glyph overflow、provider replay。
- utility：每頁 OCR accepted、layout accepted、ROI committed 比率。
- fidelity：人工盲評、字級比例、panel order、繁中與角色一致性、背景修復。
- operations：可重跑、費用、磁碟、失敗 exit、metadata、部署與授權。

## 7. 希望第三方優先回答的問題

1. 在保留「不確定就不擦」的前提下，`repair` 歷史 smoke 的 2/76 writeback 應如何拆解並在
   current HEAD 重跑？是 detector grouping、OCR 長度、safe mask、font floor、line-count、
   geometry/collision 哪一個 gate 造成主要 rejection？需要哪些分層 telemetry 才能用數據調整，
   而不是放寬所有安全門檻？

2. 應直接以 `repair` 為唯一 production path，逐步移除 legacy/dormant duplication；還是先把
   RAQM safe-region/atomic render 小幅 backport 到 `main`？判準應包含 migration cost、replay需求、
   live完成率與測試可重現性。

3. 如何把 panel detection、bubble/region graph 與 manual override 接到真正的 ORDER stage，並讓
   translation request identity 在重新排序後仍穩定？

4. OCR 是否應以頁級 `PageOCRStager` 做跨 group batching，保存 token evidence，建立真 crop/no-text
   calibration corpus，再以 calibrated confidence 取代 script heuristic？如何兼顧英文 SFX、短字與
   furigana？

5. translation production 應如何接入 structured client，讓 provider-side JSON schema、
   `data_collection=deny`、ZDR、connection reuse、item-level retry、raw replay 同時成立？source-aware
   sanitizer 應保留舊規則、改為 validator issue，還是交由 typography 決定線條字元？

6. layout solver 應如何在不大幅縮字下提高可排率？請特別分析 live group boundary、safe-region
   morphology、line-count constraint、neutral-sans-only、CLREQ breaks、actual fallback glyph、outline、
   tate-chu-yoko 與 neighbor occupancy，而不是只調 beam size。

7. 如何定義 release gate：至少多少 live pages/groups、多少 commit ratio、多少人工盲評、允許多少
   source-preserved/false erase、哪些格式與 metadata 必須 byte/fidelity-safe？

8. 如何統一 Conda + Poetry、lock、CI、model/font checksum、license/third-party notices、可重跑
   validation fixtures，以及 `output/.manga-translator` 的 retention/GC/備份政策？

## 8. 給第三方的函式與檔案索引

即使無法看 repository，以下索引可用來要求維護者提供最小必要 source excerpt 或 trace。

### `main@ba0e3a4`

| 主題 | 檔案 / symbol |
|---|---|
| CLI / config | `src/manga_translator/cli.py: main, run, test, detect_only, doctor`; `config.py: AppConfig.from_yaml` |
| page orchestration | `pipeline.py: run_pipeline, process_single_page` |
| dedup/order/collision | `pipeline.py: _merge_duplicate_groups, _refresh_group_order, _resolve_render_collisions` |
| translation dispatch | `pipeline.py: _request_translations, _translate_groups` |
| CTD adapter | `detector.py: detect_text_regions, postprocess_regions, _build_groups, _conservative_text_mask` |
| vendored detector | `ctd/inference.py: TextDetector`; `ctd/utils/textblock.py: group_output, TextBlock` |
| OCR | `manga_ocr_runtime.py: MangaOcrRuntime`; `ocr.py: ocr_group_detailed, assess_ocr_result` |
| OpenRouter | `translator.py: _request_with_retry, _payload, sanitize_translation_text, validate_translation` |
| layout/render | `typesetter.py: _infer_original_geometry, plan_text_layout, render_text_into_group` |
| inpaint | `inpainter.py: inpaint_regions, _inpaint_one_mask` |
| image/debug | `image_io.py`; `artifacts.py: dump_debug_artifacts` |

### `repair@2953d04`

| 主題 | 檔案 / symbol |
|---|---|
| staged orchestration | `pipeline.py: _build_pipeline_stage_runners, process_single_page_staged, run_pipeline` |
| DAG / cache / lease | `stages/runner.py: STAGE_DAG, StageRunner`; `stages/adapters.py: build_pipeline_stage_specs` |
| storage | `storage/job_store.py: JobStore`; `storage/artifact_store.py: ArtifactStore` |
| durable document | `domain/models.py: PageDocument, RegionIdentity, RegionRevision`; `domain/issues.py` |
| reconciliation | `domain/reconcile.py` |
| stage state | `stages/state.py: PipelineStageState`; `stages/base.py: StageInputs, StageOutputs, StageSpec` |
| detection/grouping | `detector.py`; `stages/detect.py`; vendored `ctd/` |
| style/safe region | `style/extract.py: extract_style_fingerprint`; `typography/safe_region.py: build_safe_region` |
| production OCR | `ocr.py: _get_model, ocr_group_detailed`; `manga_ocr_runtime.py: MangaOcrRuntime` |
| dormant OCR staging | `stages/ocr.py: PageOCRStager`; `ocr_confidence.py` |
| production translation | `translator.py: translate_*_mapped, _request_with_retry, _payload` |
| strict mapping | `contracts/mapping.py`; `contracts/translation.py` |
| dormant structured client | `translation/client.py: StructuredTranslationClient`; `translation/provider.py` |
| RAQM layout | `typography/layout.py: LayoutRequest`; `typography/solver.py: solve_layout`; `typography/fonts.py` |
| atomic render | `typography/render.py: atomic_inpaint_render`; `stages/render.py: render_page_atomic` |
| CLI operations | `cli.py: run, inspect, replay, cache, doctor` |

## 9. 最小結論

`main` 是「保守但單體、容易理解」的實作；它已有擦除前預演，但環境規範、CLI status、
detector confidence、設定有效性、OCR/排版正確性與 replay/provenance 都有具體缺口。

`repair` 是「可追溯、可續跑、每個 ROI 可 rollback」的重構；其安全工程明顯更完整，但最後一份
可追溯的歷史 production sample 只寫回 2/76 groups，current-head 比率未知，而且多個看似完成的
進階模組尚未接線，驗證報告也落後 current branch。現階段最需要的不是再增加抽象層，而是把
production call path、配置、
telemetry 與 live acceptance corpus 對齊，先找出低 writeback 的主要因果，再決定哪些安全門檻
可以在有證據下調整。
