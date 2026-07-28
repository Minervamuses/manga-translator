# manga-translator 0.3.2 使用指南

## 目前環境

- 專案位置：請依實際解壓位置調整，以下以專案根目錄為基準
- 執行環境：WSL `Ubuntu-24.04`
- Conda 環境：`manga`
- 翻譯服務：OpenRouter
- 翻譯模型：由 `config.yaml` 的 `openrouter.model` 指定
- OpenRouter API key：存於 WSL 的 `~/.bashrc`，不寫入專案

## 第一次解壓或更新專案

在 WSL 進入專案後執行：

```bash
cd "/path/to/manga-translator"
conda env update -f environment.yml --prune
conda activate manga
python -m pip install -e . --no-deps --no-build-isolation
source ~/.bashrc
manga-translate doctor --config config.yaml --strict-api-key
```

若 `manga` 環境尚未建立，把前兩行環境指令改成：

```bash
conda env create -f environment.yml
conda activate manga
```

此專案只使用 Conda；不要建立 `.venv`，也不要執行 `poetry install`。

## 批次翻譯圖片

### 1. 放入圖片

把要翻譯的圖片直接放進：

```text
<專案根目錄>\input
```

支援 `.jpg`、`.jpeg`、`.png`、`.bmp`、`.webp`。程式只掃描 `input` 第一層，不會遞迴處理子資料夾。圖片依自然順序處理，例如 `page2.png` 會早於 `page10.png`。

### 2. 開啟 WSL 並進入專案

PowerShell：

```powershell
wsl -d Ubuntu-24.04
```

WSL：

```bash
conda activate manga
cd "/path/to/manga-translator"
source ~/.bashrc
```

### 3. 開始翻譯

```bash
manga-translate run --config config.yaml
```

結果位於：

```text
<專案根目錄>\output
```

輸出檔名與輸入相同。

## 建議先用單張問題頁回歸

```bash
manga-translate test \
  --config config.yaml \
  --image "input/實際檔名.jpg" \
  --dump-json \
  --save-intermediate
```

這會產生：

- `test_<原檔名>`：最終結果
- `output/debug/*_regions_raw.png`：原始 detector 候選
- `output/debug/*_regions_post.png`：過濾後候選
- `output/debug/*_groups.png`：OCR／翻譯狀態
- `output/debug/*_manifest.json`：每段文字的來源、OCR 候選、品質與拒絕原因
- `output/debug/*_inpainted.png`：只清除原文後的中間圖

## 只檢查文字偵測

```bash
manga-translate detect-only \
  --config config.yaml \
  --image "input/實際檔名.jpg"
```

若偵測圖已經框到臉部、線稿或裝飾，不要直接跑整批翻譯；先檢查 `config.yaml` 是否仍保持：

```yaml
detection:
  keep_undetected_mask: false
  mask_fallback_enabled: false

inpainting:
  allow_bbox_fallback: false
```

## 完整除錯模式

```bash
manga-translate run \
  --config config.yaml \
  --debug \
  --dump-json \
  --save-intermediate
```

## 常見狀況

### OCR 顯示 `Unrecognized feature extractor`

0.3.2 保留 0.3.1 已完成的修正，已移除造成此錯誤的舊版 `AutoFeatureExtractor` 載入路徑，改用與目前 Transformers 相容的直接載入器。正常執行時只會出現一次「載入 manga-ocr 模型中」，完成後同一批圖片會共用該模型。

若 OCR runtime 仍無法初始化，程式會在批次開始時中止並顯示單一錯誤，不會對每個候選框重複嘗試，也不會把未翻譯頁面列為成功。

### 原文仍留下一點黑色描邊

把：

```yaml
inpainting:
  mask_dilate: 1
```

小幅改成 `2` 後只測一頁。不要恢復舊版的大膨脹，也不要打開 `allow_bbox_fallback`。

### 背景又出現大面積模糊

先確認下列三項沒有被改動：

```yaml
detection:
  keep_undetected_mask: false
  mask_fallback_enabled: false

inpainting:
  allow_bbox_fallback: false
```

若只在某個特殊頁面需要 mask fallback，應先用 `detect-only` 查看遮罩，完成該頁後再關閉，不建議全書常駐開啟。

### 出現 `...`、`||` 或純符號字幕

0.3.2 會在 OCR、翻譯解析與寫回前多次清洗這些符號，並依原文移除長音符誤轉出的破折號線條。若仍看到，開啟 `--dump-json`，查看該 group 的 `ocr_text`、`translation`、`status` 與 `skip_reason`，可判斷符號是在圖片 OCR 階段還是模型回覆階段出現。

### 兩段中文寫在同一位置

查看 manifest 是否有：

```text
status: render_collision_rejected
skip_reason: overlaps:<group id>
```

若沒有，而兩個來源框確實是同一區域，可小幅降低 `postprocess.render_collision_mask_iou` 或 `render_collision_iom`；不要一次降太多，以免刪掉相鄰合法台詞。

### 中文太小、留白過多或偏離原字區

確認：

```yaml
typesetting:
  render_scope: "group_mask"
  layout_mode: "preserve"
  layout_from_mask: true
  adaptive_bubble_layout: true
  max_font_growth_ratio: 1.15
  font_preserve_floor_scale: 0.92
  min_font_scale: 0.85
  reject_unreadable_layout: true
```

0.3.2 先從原文字 mask 重建字級、欄數、中心與佔用範圍，再利用同一字幕框的安全留白。原尺寸附近能排下時不會選小字；低於可讀門檻則保留日文，不先擦成空白。

manifest 中可查看：

```text
layout_bbox                 排版可用區域
layout_info.block_bbox      實際會畫出的文字範圍
rendered_font_size          最終字級
status: layout_rejected     需要低於可讀字級，已保留原文
status: layout_collision_rejected
                             實際文字 block 會重疊，已保留較弱群組原文
```

## 更換翻譯模型

修改 `config.yaml`：

```yaml
openrouter:
  model: "x-ai/grok-4.5"
```

API key 不需要寫進 `config.yaml`；保留由 `~/.bashrc` 提供的方式較安全。

## 執行前健檢

```bash
manga-translate doctor --config config.yaml --strict-api-key
```

## 注意事項

- 每次重新開啟終端機後先執行 `conda activate manga`。
- 若 shell 沒有自動載入金鑰，執行 `source ~/.bashrc`。
- 從專案根目錄執行指令，確保模型、字體與設定的相對路徑正確。
- OpenRouter 只會收到 OCR 後的純文字，不會收到漫畫圖片。
- 中止翻譯可按 `Ctrl+C`。
