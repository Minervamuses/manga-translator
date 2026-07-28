# manga-translator 0.3.2

日文漫畫翻譯工具：本機完成文字偵測、OCR、原文局部清除與繁體中文排版；OpenRouter 只接收 OCR 後的文字，不會收到漫畫圖片。

0.3.2 的重點不是再增加一層「縮字直到塞得下」，而是重做整個寫回邏輯：先從原文字形 mask 推回原字級、欄數、文字塊中心與佔用範圍，再於同一個對話框／字幕框內重新排版。只有原尺寸附近完全無法放入時才允許有限縮字；預設低於可讀門檻就保留原文，不先擦掉再塞入極小中文。

## 0.3.2 核心修正

### 1. 原字級與原空間優先，不再機械式一路縮小

舊版的核心做法是：取得一個矩形，從大字開始嘗試，放不下就逐級縮小，直到能塞進矩形。這確實是機械式 fit，也會把偵測框稍微偏小、翻譯稍長、重複 OCR 等上游問題全部轉化成「字突然很小」。

新版改為：

- 從真實文字 mask 的投影與 detector 字級提示估計原字級。
- 重建原文字塊中心、原欄數、欄距、字距與寬高範圍。
- 先在同一個對話框／字幕框的安全底色區域擴張，再調整欄數與字距。
- 只要原字級 92% 以上存在可行方案，就禁止選擇更小的方案。
- 預設最低只允許縮到原字級約 85%；仍放不下就保留原文，不產生難以閱讀的小字。
- 翻譯較短時會以平衡分欄、字距／行距與置中填回原空間，而不是縮成一小撮文字留在角落。
- 翻譯較長時會增加欄數或行數，而不是把剩餘字硬塞進最後一欄。

### 2. 排版在擦除前先完成預演

每段譯文會先產生固定的 `TextLayoutPlan`，確認：

- 字級仍在可讀範圍。
- 文字真正畫出的 block 沒有和別段譯文碰撞。
- 位置仍落在原文字區或安全的對話框留白內。

只有預演通過的群組才會進入 inpainting。排版不合格時，原日文仍完整保留，不會留下空白對話框。

### 3. 完整對白框與欄位碎片不再重複 OCR／重複翻譯

多解析度 detector 常同時產生：

- 涵蓋整句的大框。
- 個別直排欄位的小框。
- 上半句／下半句碎片。

舊流程可能把它們串成「完整句 + 半句 + 個別欄」，導致翻譯內容本身重複，最後看起來像排版重疊。新版將 whole-region OCR 與 leaf-column OCR 視為替代假設，並用 mask 包含、bbox 包含、文字覆蓋率與 OCR 完整度選出單一結果。

### 4. 以實際文字 block 做最後碰撞檢查

除了 OCR 去重、翻譯後去重與 group 幾何碰撞，0.3.2 會直接比較最終預計畫出的文字 block。兩段文字即使來源 OCR 不同，只要實際寫回會重疊，就只保留較可靠的一段。

### 5. 清除原文沒有的線條、分隔符與省略號

- 清除 `|`、`||`、`｜`、`丨`、`‖` 等 OCR／模型格式殘留。
- 原文沒有省略號時，移除模型自行加入的 `...`、`⋯`、`…`。
- 日文長音符 `ー` 不再被翻譯成中文 `——`；原文沒有真正破折號時，會移除 `—`、`―`、`─`、`━` 等線條字元。
- 相鄰的大段重複譯文會折疊；原文本來就有的短促重複仍保留。

### 6. 平坦字幕框去除殘影，但不擴大模糊範圍

- detector mask 仍只負責提供可靠文字核心，不允許退回整個矩形框擦除。
- 平坦白底／灰底字幕框會在小範圍內補抓反鋸齒文字邊緣，再以小半徑 Telea 修復。
- 補抓只在確認背景近似純色時啟用，且受對比度與最大增長比例限制，避免吃到人物、線稿或對話框邊界。
- 非整頁尺寸的意外 mask 不再被錯誤放大成整頁遮罩。

### 7. 保留 0.3.1 的 OCR runtime 修正

- 不再使用舊版 `AutoFeatureExtractor` 載入路徑。
- 改以 `ViTImageProcessor`、日文 BERT tokenizer 與 `VisionEncoderDecoderModel` 載入 `kha-white/manga-ocr-base`。
- OCR 模型每批只初始化一次；初始化失敗只回報一次並立即停止，不會逐文字框重試，也不會誤報整批成功。

詳細根因與修正見 [`UPGRADE_NOTES.md`](UPGRADE_NOTES.md)，測試及實際樣本結果見 [`VALIDATION_REPORT.md`](VALIDATION_REPORT.md)。

## 處理流程

```text
漫畫頁
  ↓
comic-text-detector（主尺寸＋可選高解析度尺寸）
  ↓
raw-supported 精確文字 mask + 候選框後處理
  ↓
多視圖 manga-ocr + 來源感知品質檢查
  ↓
整句／欄位碎片替代假設 + OCR 幾何／文字去重
  ↓
OpenRouter 固定 ID 翻譯 + 來源感知標點清洗 + 品質驗證
  ↓
翻譯後去重 + group 幾何碰撞阻擋
  ↓
重建原文字級／中心／欄數／佔用範圍
  ↓
擦除前排版預演 + 實際文字 block 碰撞檢查
  ↓
只清除通過所有檢查的精確文字像素
  ↓
使用同一份預演計畫寫回繁體中文
```

## Conda 環境

專案只用 Conda 建立與啟用 `manga` 環境；專案的 Python 套件一律由 Poetry
依 `poetry.lock` 管理。專案已停用 Poetry virtualenv，不會在 repo 內建立 `.venv`。

建立環境：

```bash
conda env create -f environment.yml
conda activate manga
poetry install --with dev
```

更新既有環境：

```bash
conda env update -f environment.yml --prune
conda activate manga
poetry install --with dev
```

`poetry install` 會把 CLI 與鎖定的專案依賴安裝到目前啟用的 Conda 環境；
請勿使用 `pip`、`venv`、`uv` 或讓 Poetry 另建 virtualenv。

完整 ZIP 已包含 comic-text-detector 模型與中文字體；manga-ocr 權重沿用 Hugging Face 本機快取，首次使用時下載，之後直接重用。

## 使用方式

把圖片放入 `input/` 第一層，執行：

```bash
manga-translate run --config config.yaml
```

輸出位於 `output/`，檔名與輸入相同。

單頁回歸：

```bash
manga-translate test \
  --config config.yaml \
  --image "input/實際檔名.jpg" \
  --dump-json \
  --save-intermediate
```

只檢查偵測：

```bash
manga-translate detect-only \
  --config config.yaml \
  --image "input/實際檔名.jpg"
```

完整 debug：

```bash
manga-translate run \
  --config config.yaml \
  --debug \
  --dump-json \
  --save-intermediate
```

`manifest.json` 會記錄 OCR 候選、採用來源、譯文狀態、拒絕原因、排版框、實際文字 block、字級與排版模式。

## 重要設定

### 原尺寸優先排版

```yaml
typesetting:
  layout_mode: "preserve"
  layout_from_mask: true
  adaptive_bubble_layout: true
  font_size_scale: 1.0
  max_font_growth_ratio: 1.15
  font_preserve_floor_scale: 0.92
  min_font_scale: 0.85
  reject_unreadable_layout: true
  balance_columns: true
  clip_render: false
```

含義：

- `font_preserve_floor_scale: 0.92`：原尺寸附近能排下時，不得為了幾何分數選更小字。
- `min_font_scale: 0.85`：預設可接受的最低字級比例。
- `reject_unreadable_layout: true`：低於門檻就保留原文，不進行擦除。
- `adaptive_bubble_layout: true`：先利用同一對話框／字幕框內的安全留白。
- `balance_columns: true`：直排欄位平均分配，避免最後一欄塞滿或其他欄大片空白。

### 精確遮罩與修復

```yaml
detection:
  keep_undetected_mask: false
  raw_support_threshold: 30
  raw_support_dilate: 2
  mask_fallback_enabled: false

inpainting:
  method: "hybrid"
  mask_dilate: 1
  extra_mask_dilate: 0
  inpaint_radius: 2.0
  allow_bbox_fallback: false
  hybrid_flat_edge_expand: 3
  hybrid_flat_edge_contrast: 10
  hybrid_flat_edge_max_growth: 3.2
```

不要為了召回率直接打開 `keep_undetected_mask` 或 `allow_bbox_fallback`；這會重新引入把非文字區域送入修復的風險。

### 最終碰撞保護

```yaml
postprocess:
  enable_render_collision_filter: true
  render_collision_mask_iou: 0.48
  render_collision_mask_containment: 0.72
  render_collision_iom: 0.78
  render_collision_containment: 0.88
```

## 調參原則

### 字仍然偏小或留白太多

先確認 `layout_mode` 仍為 `preserve`、`adaptive_bubble_layout` 與 `balance_columns` 仍為 `true`。不要直接降低 `font_size_min`；那只會重新允許難以閱讀的小字。

翻譯本身明顯比原文冗長時，應先檢查是否存在重複 OCR／重複譯文。正常情況下，0.3.2 會先增加欄數與利用安全留白，而不是縮字。

### 某段保留原文

查看 debug manifest：

- `layout_rejected`：在最低可讀字級仍放不下。
- `layout_collision_rejected`：實際文字 block 會與另一段譯文重疊。
- `render_collision_rejected`：上游群組已被判定為同位置重複框。
- `missing_text_mask`：沒有足夠可靠的文字像素可安全擦除。

這些狀態是刻意的失敗保護，不會先把原文擦掉。

### 原文仍留少量描邊

只把 `inpainting.mask_dilate` 從 `1` 小幅提高到 `2` 並測單頁。不要恢復大範圍膨脹，也不要開啟矩形 fallback。

### 漏掉小字或美術字

先增加 `additional_input_sizes` 或小幅降低 `conf_thresh`，不要先開 mask fallback：

```yaml
detection:
  additional_input_sizes: [1536]
  conf_thresh: 0.27
```

## 測試

```bash
PYTHONPATH=src python -m compileall -q src tests
PYTHONPATH=src pytest -q
```

0.3.2 通過 **67 項測試**。實際五張問題頁共 38 個文字群組全部完成排版預演；計畫字級相對原字級估計值為 0.987–1.029 倍，中位數 1.000。驗證圖片與逐群組數據位於 `validation_samples/`。

## 專案結構

```text
manga-translator/
├── config.yaml
├── environment.yml
├── glossary.json
├── pyproject.toml
├── models/
├── fonts/
├── input/
├── output/
├── samples/
├── validation_samples/
├── tests/
└── src/manga_translator/
    ├── ctd/                  # vendored comic-text-detector
    ├── detector.py           # 多尺寸偵測與 raw-supported 精確 mask
    ├── manga_ocr_runtime.py  # 相容目前 Transformers 的 OCR runtime
    ├── ocr.py                # 多視圖 OCR、替代假設與來源感知去重
    ├── translator.py         # 固定 ID 翻譯、標點清洗與內容驗證
    ├── inpainter.py          # 局部文字邊緣補抓與安全修復
    ├── typesetter.py         # 原幾何重建、排版計畫與安全寫回
    ├── pipeline.py           # 去重、排版預演與實際 block 碰撞檢查
    ├── artifacts.py          # debug overlay／manifest
    └── image_io.py           # Unicode 路徑安全讀寫
```
