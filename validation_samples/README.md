# 0.3.2 實際樣本驗證素材

## 0.3.2 排版／重疊／線條回歸

- `v032_caption_3_final_preview.jpg`：第一張彩色字幕問題頁的最終排版預覽。
- `v032_caption_4_final_preview.jpg`：第二張彩色字幕問題頁的最終排版預覽。
- `v032_caption_5_final_preview.jpg`：第三張彩色字幕問題頁的最終排版預覽。
- `v032_overlap_0188_final_preview.jpg`：整句框與欄位碎片去重、位置對齊回歸。
- `v032_dash_0211_final_preview.jpg`：日文長音符不再生成中文額外線條的回歸。
- `v032_caption_4_inpainted.jpg`、`v032_dash_0211_inpainted.jpg`：只移除原文後、尚未寫回中文的局部修復結果。
- `v032_*_layout.json`：逐群組原文、測試譯文、layout bbox、字級、欄位、字距與實際 block bbox。
- `v032_layout_metrics.json`：五張實際頁共 38 個群組的原字級／計畫字級與文字塊範圍量化結果。

這些譯文由驗證腳本固定指定，用來測試幾何、去重、修復與排版；不是 OpenRouter 語意品質評分。

## 早期偵測／安全遮罩回歸

- `sample_1_detector_overlay.jpg`、`sample_2_detector_overlay.jpg`：detector／group overlay。
- `sample_1_safe_mask.png`、`sample_2_safe_mask.png`：允許進入修復的 raw-supported 安全文字遮罩。
- `detector_stats.json`：候選來源與群組統計。
- `sample_1_layout_preview.jpg`、`sample_1_layout_stats.json`：早期三群組幾何回歸。

完整說明見專案根目錄的 `VALIDATION_REPORT.md`。
