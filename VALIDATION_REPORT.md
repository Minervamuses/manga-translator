# manga-translator v0.3.2 最終驗證報告

日期：2026-07-30

分支：`repair_p0_p4_completion`

受驗證的實作 HEAD：`1e58b46`（V1 報告 commit 的 SHA 另見交付訊息；Git commit 無法在自身內容中固定自己的 SHA）

## 實際命令與結果

所有 Python、測試、lint、Poetry 與 CLI 命令均由既有 Conda environment `manga` 進入；未建立 virtualenv、未安裝或更新 dependency。

```text
PYTHONPATH=src conda run -n manga python -m compileall -q src tests
passed

PYTHONPATH=src conda run -n manga python -m pytest -q -p no:cacheprovider
516 passed, 2 failed, 1 skipped, 2 deselected in 118.04s

PYTHONPATH=src conda run -n manga python -m pytest -q -p no:cacheprovider tests/test_baseline_manifest.py
13 passed, 2 skipped in 1.80s

conda run -n manga ruff check .
passed

conda run -n manga poetry check --lock
passed

conda run -n manga python -m pip check
passed

git diff --check
passed

conda run -n manga manga-translate --help
passed
```

完整 pytest 的兩個失敗都來自舊 `benchmarks/baseline/v0.3.2/manifest.json` 將 `environment.yml` SHA 當成 current regression contract。E1 依 `plan.md` 明確移除 `conda-lock` 宣告，而同一份計畫禁止刷新舊 fingerprint／gate；因此未改寫歷史 manifest，將兩個 SHA assertion 標為 archived，再以 targeted test 確認該檔其餘 13 項測試通過。依「V1 完整 pytest 只跑一次」限制，沒有為取得綠色摘要而重跑完整 suite。

## 六頁真實 smoke

前五頁在 T2 已各執行唯一一次真 detector、OCR、provider 與預設 RAQM renderer；V1 直接驗證其產物與保存狀態，沒有再次呼叫 provider。T1 固定 38-group corpus 的新排版 review sheets 已由使用者核准為 38/38 `new_better`、`critical_regression = 0`。

1. `0188_ive_hwa002.jpg`（page `5e79409955a1…`）：20 個 live groups；`g017` 安全寫回。`g000`–`g009`、`g011`、`g014`、`g015`、`g018`、`g019` 因 layout 安全檢查拒絕；`g010` OCR reject；`g012`、`g013`、`g016` collision reject。所有拒絕區域保留輸入像素，沒有先擦除。
2. `0211_t_11takamatic006.jpg`（page `01b5cbea3c23…`）：9 個 live groups；`g000`–`g003`、`g005`–`g008` layout reject，`g004` OCR reject；無安全 candidate，因此全頁保留輸入。
3. `__#Uf008_#Ueff9#Ue7cc (3).jpg`（page `d6a2c70d9472…`）：9 個 live groups，`g000`–`g008` 全部 layout reject；全頁保留輸入。
4. `__#Uf008_#Ueff9#Ue7cc (4).jpg`（page `f81020c46353…`）：8 個 live groups；`g001` 安全寫回，`g000`、`g002`–`g007` layout reject 並保留輸入。
5. `__#Uf008_#Ueff9#Ue7cc (5).jpg`（page `0b9dc8fadf8e…`）：13 個 live groups；`g000`–`g010`、`g012` layout reject，`g011` safe-region confidence reject；無安全 candidate，因此全頁保留輸入。
6. 非 benchmark 自有頁 `0822_omake_hayaten002.jpg`（page `e6ab27185440…`）：CLI 單頁 smoke 成功，10 個 stages 全部 succeeded。17 個 mapping groups 中有 16 個唯一 request IDs；15 個 layout reject、1 個 collision reject、1 個 OCR reject，沒有 render target。輸出與輸入同尺寸，JPEG 重編碼平均絕對差 `0.0479`、最大差 `3`，沒有像素通道差超過 `8`；因此沒有新擦除、空白框或排版覆寫。

前五頁各只有一份主要 provider response。第六頁的一個 `source_echo` 驗證失敗依既有邏輯觸發單句 retry，因此同一次頁面執行保存 2 份 response artifacts；16 個 request IDs 仍全部唯一且 exact mapping 完整。針對 454 個相關 artifacts、JSON、設定與本報告的 secret 掃描通過，未保存 API key。

## 已知限制

- Production live detection／OCR 與 T1 固定 38-group visual corpus 的群組邊界不同；T1 的人工核准不能外推成所有 live OCR 譯文都能安全排入。
- 六頁 smoke 中只有 `0188:g017` 與字幕頁 `(4):g001` 取得安全 writeback。其餘疑慮 groups 已逐頁列出並保留輸入，未為了提高成功率降低字級、方向、欄數、碰撞或 safe-region 門檻。
- `LayoutOverflow:shaping_failed` 是目前累積的拒絕標籤；其中部分 candidate 的實際原因是 tracking、safe mask、字級 floor 或幾何條件，不應解讀為 RAQM runtime 故障。
- 非 benchmark 頁的輸入本身已有肉眼可見的舊翻譯／排版瑕疵；本次只能證明新流程未進一步改寫或擦除，不能宣稱該頁視覺品質改善。
- OpenRouter 回傳品質仍受 OCR 雜訊影響；第六頁發生一次 `source_echo` retry。沒有再擴充 prompt、評測集或 provider gate。

## Rollback

預設 renderer 為 `typesetting.engine: raqm`。若需明確回退，只把 `config.yaml` 的 `typesetting.engine` 設為 `legacy`；程式不會在 RAQM 拒絕時靜默切換 legacy。任何拒絕 candidate 都保留輸入文字，不先 inpaint。

## 取消項目

依 `plan.md`，本次沒有建立或刷新 G0～G5、全樹 fingerprint、fresh CI、conda-lock、SBOM、wheel／release ZIP、release evidence schema、30-page corpus 或舊 P3/P4/P5／release infrastructure，也沒有 merge、push 或建立 release。
