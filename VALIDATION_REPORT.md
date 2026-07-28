# 0.3.2 驗證報告

## 一、自動測試

執行：

```bash
PYTHONPATH=src python -m compileall -q src tests
PYTHONPATH=src pytest -q
```

目前結果：

```text
compileall: passed
67 passed
```

新增與既有測試涵蓋：

- refined mask 必須受 raw segmentation 支持。
- 空 mask 預設保留原圖；bbox fallback 只有明確啟用時才存在。
- 平坦背景反鋸齒文字邊緣可被補抓，附近線稿不受影響。
- 意外尺寸 local mask 不得擴張成整頁遮罩。
- whole-region OCR 與 leaf-column OCR 視為替代假設，不串成重複句。
- 大框／小欄位即使 IoU 很低，也可由像素包含率辨識為同一段文字。
- OCR 與翻譯重複內容折疊，但原文合法的短促重複仍保留。
- 原文沒有省略號／直線／dash 時，不允許模型自行加入。
- 日文長音符 `ー` 不轉成 `——`。
- 原字級估計會交叉檢查 detector hint 與 mask 投影。
- 原尺寸附近有可行方案時，不會為了幾何分數選擇更小字。
- 翻譯需要低於可讀門檻時，layout 會拒絕而不是縮到極小。
- 直排文字會平衡分欄，橫排短句會使用有限 tracking 填回原寬。
- 實際計畫文字 block 的 bbox 可重現並參與碰撞檢查。
- 兩個計畫 block 重疊時，只允許較可靠者進入 inpainting。
- 排版預演與最終寫回使用同一份 plan。
- OCR runtime 只初始化一次；失敗狀態也只回報一次。
- OCR preflight 失敗時，不進入頁面處理或產生假成功摘要。
- Unicode 路徑、字體 fallback、局部 alpha compositing 與設定路徑解析。

## 二、五張實際問題頁回歸

使用專案內真實 `comictextdetector.pt`、真實輸入頁、真實 detector mask 與 0.3.2 排版／修復程式；譯文由驗證腳本固定指定，以隔離外部翻譯模型波動。沒有讀取使用者 API key，也沒有呼叫 OpenRouter。

驗證頁：

1. `__#Uf008_#Ueff9#Ue7cc (3).jpg`
2. `__#Uf008_#Ueff9#Ue7cc (4).jpg`
3. `__#Uf008_#Ueff9#Ue7cc (5).jpg`
4. `0188_ive_hwa002.jpg`
5. `0211_t_11takamatic006.jpg`

總計：

```text
實際文字群組：38
成功產生可讀排版計畫：38 / 38
排版碰撞：0
低於可讀門檻：0
矩形擦除 fallback：0
```

逐群組數據位於 `validation_samples/v032_layout_metrics.json`。

## 三、字級與空間保留量化結果

以 detector／mask 推得的原字級為分母：

```text
計畫字級 / 原字級估計
最小值：0.987
中位數：1.000
最大值：1.029
```

亦即這五張問題頁的 38 個群組，最終計畫字級全部落在原字級估計的約 98.7%–102.9%，沒有出現舊版一路縮小到難以閱讀的情況。

文字 block 相對原文字像素範圍：

```text
block 寬度比中位數：1.013
block 高度比中位數：1.000
```

這代表新排版在整體上維持原文字塊的寬高佔用，而不是只把字放進框內就算完成。短譯文因字數變少，不可能每一段同時百分之百複製原寬與原高；此時仍優先維持原字級、中心與合理字距。

## 四、使用者指出的具體問題

### 1. 前三張字幕重疊

三張彩色字幕頁重新偵測後分別得到：

```text
(3)：7 個群組，7 份排版計畫
(4)：6 個群組，6 份排版計畫
(5)：7 個群組，7 份排版計畫
```

結果：

- 完整框與欄位碎片沒有被重複翻譯。
- 實際文字 block 沒有互相覆蓋。
- 文字維持接近原字級並重新填滿原字幕框。
- 不再把多餘文字塞進最後一欄。

對照圖：

- `v032_caption_3_final_preview.jpg`
- `v032_caption_4_final_preview.jpg`
- `v032_caption_5_final_preview.jpg`

### 2. `0188_ive_hwa002.jpg` 中間偏左沒有對準對話框

真實 detector 產生 27 個 post candidates，最後收斂為 11 個文字群組。原本有完整對白框與多個欄位碎片同時存在；0.3.2 只保留完整 OCR 假設，並以原文字中心、欄數與實際 mask 範圍排版。

結果：11 / 11 群組皆有可讀計畫，沒有重複句、跨框文字或計畫 block 碰撞。

對照圖：`v032_overlap_0188_final_preview.jpg`。

### 3. `0211_t_11takamatic006.jpg` 的「謝謝指導」下方多出線條

驗證輸入故意使用：

```text
原文：ありがとうございましたーッ
模型式譯文：謝謝指導——！
```

來源感知清洗後：

```text
謝謝指導!
```

最終圖沒有 `——` 形成的額外水平線；原日文字形也已由局部安全 mask 清除。

對照圖：`v032_dash_0211_final_preview.jpg`。

## 五、修復範圍檢查

`hybrid` 修復在平坦字幕框執行以下限制：

- 原始 group mask 必須存在。
- 只在局部 bbox 內處理。
- 先確認周圍背景近似純色。
- 最多向外搜尋 3 px 的反鋸齒邊緣。
- 新 mask 像素數受最大增長比限制。
- Telea 半徑維持 2 px。

實際中間圖：

- `v032_caption_4_inpainted.jpg`
- `v032_dash_0211_inpainted.jpg`

可見原文字已清除，但人物、字幕框邊線與附近背景沒有被整塊模糊。

## 六、OCR runtime 回歸

針對先前日誌中的：

```text
Unrecognized feature extractor
逐文字框反覆載入模型
整批 OCR 失敗卻顯示失敗 0 頁
```

保留以下驗證：

```text
ViTImageProcessor + 日文 BERT tokenizer + VisionEncoderDecoderModel
成功初始化：整批共用一次
失敗初始化：快取同一錯誤，不逐框重試
批次 preflight 失敗：頁面處理次數為 0
CLI：只輸出一次可讀錯誤，不輸出成功摘要
```

測試使用可控的模擬 Transformers backend；目前容器未下載完整 Hugging Face OCR 權重，因此沒有宣稱完成線上模型的真實推論。

## 七、成品封裝驗證要求

正式 ZIP 封裝後另執行：

```text
ZIP 完整性檢查
從 ZIP 重新解壓
67 項測試
compileall
config.yaml 解析
wheel build
敏感字串掃描
```

最終結果與 SHA-256 記錄於交付訊息。
