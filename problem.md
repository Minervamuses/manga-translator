# `manga-translator` v0.3.2 第三方審查與升級簡報

> 分析快照：2026-08-02。目標讀者無法存取 repository。本文提供足以做架構選擇、
> 風險排序與驗證設計的上下文；它不是逐行程式碼審查的替代品。沒有附出的程式、
> 圖片、trace 或環境資訊一律視為未知，不應以猜測補齊。

本文只分析兩個固定的程式碼快照，沒有呼叫付費 OpenRouter API，也沒有把歷史報告
寫成這次重新執行的結果。文中的「已確認」代表能由快照內的 production call path、
設定或測試直接支持，不代表第三方已獨立重現。

## 快速導覽

- [1. 委託目標與期待交付](#1-委託目標與期待交付)
- [2. 分析範圍、證據與術語](#2-分析範圍證據與術語)
- [3. 產品契約、共同流程與關鍵設定](#3-產品契約共同流程與關鍵設定)
- [4. 一頁式分支比較](#4-一頁式分支比較與暫定判斷)
- [5. 兩個 production 方法](#5-兩個-production-方法)
- [6. 現有驗證證據](#6-現有驗證證據與不能外推的部分)
- [7. 依優先級整理的 findings](#7-依優先級整理的-findings)
- [8. 需要先建立的診斷與驗證資料](#8-需要先建立的診斷與驗證資料)
- [9. 第三方技術問題](#9-請第三方優先回答的技術問題)
- [附錄 A：最小示意 excerpts](#附錄-a關鍵行為的最小示意-excerpts)
- [附錄 B：程式位置索引](#附錄-b索取後續-excerpttrace-時的程式位置)

## 1. 委託目標與期待交付

### 1.1 主要決策

請判斷下一版應採哪一種升級路徑，並說明理由：

1. 以 `repair` 的 durable pipeline 為唯一 production 基線，補齊目前未接線與未驗證部分。
2. 以 `main` 為基線，只回補必要的 mapping、RAQM 與 atomic render 能力。
3. 提出更好的第三種遷移方案。

暫定假說是：產品的首要不變條件是「不確定就不擦原文」，因此具備 strict mapping、
stage state 與逐 ROI rollback 的 `repair` 較適合作為長期基線；但在重新量測真實寫回率前，
不能把它視為可發布版本。請把這個假說當成可反駁的起點，不是預設答案。

由於第 1.4 節的 owner 參數仍未決，請提出**條件式決策**：明列哪些時程、相容性、硬體、
成本或品質門檻會讓建議從 `repair` 改成 `main` 或第三種方案。

### 1.2 第三方應交付的內容

請依下列格式回答：

1. **決策摘要**：建議的基線／遷移方案、三個主要理由、主要反對理由、切換條件與信心等級。
2. **前五項工作**：每項列優先級、影響、修改範圍、依賴、風險、rollback 與驗證方式。
3. **低寫回率診斷方案**：定義漏斗、telemetry、匿名化 evidence 與判定根因的方法；
   現有資料不足時，不要直接猜是哪個 gate。
4. **分階段 roadmap**：先修 correctness／observability，再修 quality／performance；標明哪些
   dormant 元件應接線、刪除或延後。
5. **release gate**：提出 corpus 規模、成功／失敗指標、格式與硬體矩陣；區分 owner 必須
   決定的門檻與工程上不可妥協的不變條件。
6. **補件清單**：列出還需要的 source excerpt、trace、fixture 或環境資料，以及缺少它時
   哪一項結論無法成立。

每個重要判斷請標明本文第 2.2 節的證據等級。若提出 code-level 修改，請以本文件已提供的
excerpt、檔案／symbol 或明確假設為依據；不要假裝看過未提供的程式碼。

### 1.3 範圍與非目標

- 本次要決定 CLI 圖片翻譯 pipeline 的升級方向，不包含 GUI、Web service、其他語言或模型重訓。
- 不以放寬「不確定就保留來源像素」來換取較高寫回率。
- 未經明確批准，不執行付費 API 測試；圖片本身不得送給翻譯 provider。
- repository 中「有 class／有 tests」不等於 `manga-translate run` 已使用；只以 production call
  path 判定是否接線。
- 本文可以支持架構審查與驗證設計；若要逐行修補，第三方仍需索取最小必要 source excerpt。

### 1.4 尚未由 owner 指定的決策參數

以下資訊不在現有 repository 證據中。第三方應列出假設或要求補件，不應自行宣稱已有答案：

| 缺少的參數 | 已知下限／限制 |
|---|---|
| 可接受的 ROI 寫回率 | 未定；不能靠降低來源保護換取 |
| false erase／錯區譯文容忍度 | 產品不變條件要求已標註 release corpus 為 0 |
| 翻譯人工品質門檻 | 未定；需包含台灣繁中、語意、角色一致性 |
| Linux distro／glibc、GPU driver／CUDA、CPU-only 支援、VRAM 與每頁延遲 | 未定；checked-in 設定使用 CUDA 12.8、雙尺度 FP32 |
| 每頁 API 成本與使用額度 | 未定 |
| PNG alpha、EXIF、ICC 與 metadata 保留政策 | 未定；兩個快照目前都不完整保留 |
| durable state 保存期限與磁碟預算 | 未定；`repair` 只有手動 GC |
| 模型／字型的下載、離線安裝與更新政策 | 未定；wheel 不包含根目錄 runtime assets |
| CLI、config、SQLite state 向後相容要求 | 未定 |
| 時程與可接受的 migration 範圍 | 未定 |

## 2. 分析範圍、證據與術語

### 2.1 固定程式碼快照

| 本文名稱 | Git ref / commit | 定位 |
|---|---|---|
| `main` | `origin/main@ba0e3a46d5d864cca12166dbbbbd674e1c767154` | 單體、記憶體內 production pipeline |
| `repair` | `repair@2953d0401319cb905bee78e3c597b05deeeda43f` | durable stages、RAQM、可重播狀態 |

兩個快照的 merge base 是 `eb606f3aed1f0bc235c0c0d2d426c4a87930c7ae`。以表列快照計算，
`main` 有 1 個、`repair` 有 142 個相對 merge base 的獨有 commits；兩個 endpoint tree 有
654 個檔案差異。Git ancestry 證明 `repair` 不是 `main` 的直接後裔；兩邊最後各自加入內容相同
的 `AGENTS.md`，所以 endpoint diff 不會列出該檔，這不改變 ancestry。`main` 在上述程式碼
快照之後的本地 commits 只處理本審查文件，不納入程式分析。

`repair/VALIDATION_REPORT.md` 所寫的 branch `repair_p0_p4_completion` 與 implementation
`1e58b46` 是歷史驗證身分，不是本文的 `repair@2953d04`。所有「目前」敘述都改用明確 SHA，
避免文件隨 branch 移動而失真。

未追蹤的兩份 review logs 以及 `build/`、`dist/` 沒有完整 branch、SHA、命令和逐 gate 統計，
因此未當作可追溯成功證據。

### 2.2 證據等級與優先級

finding 中的事實與風險推論分別使用以下證據等級；同一 finding 若並列多個等級，必須明說
各自對應哪個命題：

- **F（confirmed fact）**：表列快照的 production code、設定或 tests 可直接確認。
- **H（historical run）**：已提交報告記錄過，但不是在本次分析或目標快照重新執行。
- **R（risk inference）**：由資料流或演算法合理推導，仍需專門樣本驗證實際發生率。
- **U（unknown / missing evidence）**：現有材料不足，不能下結果性結論。

優先級定義：

- **P0**：阻擋基線選擇、release，或可能違反來源保護、mapping、隱私與結果誠實性。
- **P1**：顯著影響品質、可重現性、可操作性或維護正確性。
- **P2**：效能、發布治理與長期維護項目；不表示可以永久忽略。

### 2.3 術語

| 術語 | 本文定義 |
|---|---|
| region | detector 產生的一個文字候選及其 bbox，並可能帶 pixel mask |
| group | 一個或多個 regions 合成的 OCR、翻譯與排版單位 |
| ROI | 最終局部 inpaint／render transaction 的矩形區域 |
| production path | `manga-translate run` 實際可到達的呼叫路徑 |
| fixed visual | 固定 boundary 與譯文的排版 fixture；不等於 live E2E |
| live | 使用實際 detector、OCR 與 provider 的 pipeline 路徑 |
| accepted | 通過某 stage gate；必須註明是 OCR、translation 或 layout |
| 寫回成功 | ROI 完成 inpaint、render、驗證並套用到工作頁；不用 `commit`，以免和 Git 混淆 |
| rollback | ROI 失敗後保留該區域來源像素 |
| no-op page | 頁面流程成功結束，但沒有任何 ROI 寫回成功 |
| source-preserved | 因 blocking failure 輸出來源檔副本；不等同 no-op page |
| utility | 真實輸入能完成多少正確且安全的翻譯寫回，不只是不出錯 |
| CTD | comic-text-detector，本專案的文字區域與 mask detector |
| RAQM | Pillow 使用的複雜文字 shaping/layout 引擎，依賴 HarfBuzz、FriBiDi 等 native libraries |
| CLREQ | 中文排版需求；本文的 `CLREQ-like` 只表示採用部分換行／禁則原則 |
| ZDR | zero data retention；只有實際 request 落實後才能視為 provider contract |
| IoM | intersection over minimum area，用於 bbox／mask containment 類比較 |
| SFX | 漫畫擬聲字或效果字 |
| E2E | end-to-end，從真實輸入經 production path 到最終輸出 |
| MAD | mean absolute difference，本文用於比較輸入與重新編碼輸出的像素差 |
| false erase | 未有通過驗證的寫回，卻使來源文字或非文字畫面像素被擦除／破壞 |

## 3. 產品契約、共同流程與關鍵設定

### 3.1 產品契約

版本為 0.3.2，目標是把本機日本漫畫圖片中的日文轉為台灣繁體中文。CTD 偵測與
`manga-ocr` 在本機執行；翻譯只把 OCR 後的文字送到 OpenRouter，圖片不應離開本機。

最高優先不變條件是：

> OCR、翻譯、遮罩或排版只要不夠可靠，就保留該區域的來源像素；不可先擦除原文，
> 再因後段失敗留下空白框或錯位譯文。

共同概念流程：

```text
decode image
  → comic-text-detector：regions + pixel mask
  → grouping / deduplication / reading order
  → manga-ocr：Japanese text
  → OpenRouter：Taiwan Traditional Chinese text
  → layout preflight / validation
  → inpaint only the regions that will be rendered
  → render
  → encode output and evidence
```

`repair` 另在 render 後驗證每個 ROI，失敗就 rollback；`main` 只有 render 前 layout preflight，
沒有同等的 post-render transaction validation。

主要技術棧是 Python 3.11、PyTorch、Transformers、OpenCV、NumPy、Pillow、httpx、
Pydantic 與 Click。正式 target 是 Linux；Windows 開發必須使用 WSL Bash。權威環境政策是
Conda 管 interpreter／native runtime、Poetry 管 project packages／lock，不能以 repository
`.venv`、pip 或 uv 取代。

### 3.2 安全與行為相關的設定快照

| 區塊 | 兩者共同值 | `repair` 的差異／注意事項 |
|---|---|---|
| OpenRouter | model `x-ai/grok-4.5`；batch 20；temperature `.2`；timeout 90 s；小頁優先 page mode（最多 6000 chars／120 items） | schema 有 `data_collection=deny`、`zdr=true`，production payload 未送出 |
| detector | `comictextdetector.pt`；CUDA；1024 + 1536；FP16 off；NMS `.35`；confidence `.30`；mask `.30` | 增加 durable identity 與 typed issue，但底層 confidence 遺失仍存在 |
| mask safety | raw support threshold 30、dilate 2；segmentation／bbox fallback 預設 off | 再加 edge/flood-fill safe region，production confidence gate `.48` |
| OCR | 多視圖 group OCR；一般 `.46`、短文 `.66`、fallback `.74`、agreement `.70` | config 新增 pinned revision、batch 4、max length 300，但四個 runtime 欄位未傳入 `_get_model()` |
| layout | 方向 auto；10–180 px；一般 floor `.85`、hard floor `.62`；bubble expand `.72` | 預設 RAQM；safe containment `.995`；逐 ROI atomic render |
| inpaint | hybrid；mask dilation 1；Telea radius 2；只處理有合法譯文與 layout 的 groups | 逐 ROI copy → inpaint → render → verify → 最後寫回 |
| assets | `Iansui-Regular.ttf`、`NotoSansCJKtc-Regular.otf`、CTD model、manga-ocr weights | 有 font-role framework，但 production 候選固定 neutral sans |

外部 model 名稱與可用性只代表 checked-in config；本文沒有驗證 provider 在分析日仍提供該模型。

## 4. 一頁式分支比較與暫定判斷

| 面向 | `main@ba0e3a4` | `repair@2953d04` |
|---|---|---|
| orchestration | 單體 `process_single_page()` | 固定 10-stage DAG + `StageRunner` |
| 中間狀態 | process memory；debug JSON／圖為輔 | SQLite + content-addressed artifacts + `PageDocument` |
| resume / replay | 無 | cache、resume、force-stage、provider response artifact、offline replay |
| identity / mapping | 排序後短 ID；response 接受位置 fallback | region UUID/revision + request/item/source hash exact mapping |
| OCR | 多視圖 heuristic；逐 group | production 仍是多視圖逐 group；較新的頁級 batching／calibration 未接線 |
| reading order | 全頁中心座標排序 | production 仍相同；panel-aware 模組未接線 |
| typesetting | Pillow 逐 glyph 手排 | Pillow RAQM、Unicode/CLREQ-like breaker、actual raster verification |
| render transaction | 全部 active groups inpaint 後逐 group render | 每個 ROI 驗證成功後才寫回，可個別 rollback |
| status / failure output | OCR preflight exception 可非零；部分 page-loop／no-input／子命令仍 exit 0，page failure 以正常檔名重編碼 | `run` 的 blocking failure 可 exit 1 並寫到 `output/failed/`；部分子命令仍有 exit-0 缺口 |
| 已知證據 | 固定 38 groups 可產生 layout plan；不是 live E2E 寫回率 | 歷史 live 前五頁 2/59，額外頁 0/17；`2953d04` 未重跑 |
| 主要優點 | call path 短、較容易理解 | mapping、狀態、rollback、稽核與重播較完整 |
| 主要風險 | 重現、status、mapping、typography、provenance 弱 | 系統複雜、production 與 dormant code 分裂、真實可用率未知 |

暫定判斷：若以來源保護與錯誤 mapping 為硬性條件，`repair` 的架構較接近可長期維護的
production 基線；但要先以 `repair@2953d04` 重新量測完整漏斗，找出 layout／OCR／safe-region
的主要拒絕原因。不能用 tests 數量或固定 fixture 的結果替代這一步。

## 5. 兩個 production 方法

### 5.1 `main@ba0e3a4`

實際 call chain：

```text
cli.run()
  → AppConfig.from_yaml()
  → run_pipeline()
     → initialize_ocr_model()          # 在逐頁 try/except 外
     → process_single_page()
        → read_image()
        → detect_text_regions()
        → ocr_group_detailed()         # 每個 group
        → merge duplicates / refresh order
        → translate groups
        → resolve collisions
        → preflight layout plans
        → inpaint_regions()
        → render_text_into_group()      # 每個通過的 group
        → dump debug artifacts
     → write_image()
```

偵測使用 vendored comic-text-detector，先跑 1024，再跑 1536，合併 bbox／mask。refined mask
必須落在 thresholded raw segmentation 的鄰域內；segmentation component 與 bbox fallback 預設
關閉。這會偏向漏翻，而不是冒險擦除線稿。grouping 與 OCR dedup 主要依 bbox IoM／containment、
中心距離、方向、mask overlap 與文字相似度做 pairwise union。

OCR 使用固定 `kha-white/manga-ocr-base`，比較 raw、mask-isolated、contrast、threshold 與
leaf-region views。`OCRCandidate.quality` 是字元腳本、長度與重複模式 heuristic，不是模型
probability。全頁 reading order 只有中心座標排序：直排為 `(-center_x, center_y)`，沒有 panel、
bubble graph 或人工 override。

一般小頁優先整頁送 OpenRouter。parser 接受 dict、list、純字串、編號行與位置 fallback；
group ID 會隨重新 grouping／sorting 改變，也沒有 durable source hash。這提高 provider 格式錯誤
被錯配到另一 group 的風險。

legacy typesetter 從 source mask 估字級與行／欄，先產生 layout plan，通過 collision 檢查後才
進 inpaint。這個「先排版、後擦除」順序已接入 production，是 `main` 的重要安全措施；但 fit
計算主要用主字體與理論 bbox，實畫時可能切 fallback font，且直排按 codepoint 分欄，不是完整
CJK typography。

### 5.2 `repair@2953d04`

固定 10-stage DAG：

```text
SOURCE → DETECT ─┬→ STYLE ───────┐
                 ├→ SAFE_REGION ─┼→ LAYOUT ─────────────┐
                 ├→ OCR ─────┐   │                      │
                 └→ ORDER ───┴→ TRANSLATE ──────────────┼→ INPAINT_RENDER → ENCODE
SOURCE ──────────────────────────────────────────────────┘
DETECT ──────────────────────────────────────────────────┘
```

`JobStore` 以 SQLite 保存 job/page/stage、fingerprint、issues、leases 與 `PageDocument`；
`ArtifactStore` 以 SHA-256 content addressing、atomic replace 與 read-time hash verification 保存
artifact。stage fingerprint、checkpoint、`--resume`、`--force-stage` 與 provider-response lease
使流程可續跑，也降低同一 request 重複付費的機率。

region 具有 UUID identity、revision SHA 與 merge/split lineage；translation request 又有 request
ID、item ID 與 source SHA，response 必須 exact mapping，不能依 list position 猜測。request hash
仍包含當次 units 順序與 geometry key，所以尚未做到跨 regroup／reorder 完全穩定。

STYLE 從 source mask 周圍抽樣 fill、stroke、shadow、angle 等；SAFE_REGION 用 edge barrier、
protected text 與 flood-fill 建 safe mask。LAYOUT 用 `regex` grapheme、`uniseg`/CLREQ-like break
與 Pillow RAQM 做 deterministic search，每個 candidate 實際 shape/raster 後檢查 glyph、clipping、
safe containment、neighbor collision 與原幾何比例。

INPAINT_RENDER 採逐 ROI transaction：先複製原 ROI，在副本 inpaint、render、驗證，成功才套用
到工作頁。這是 per-ROI atomic，不是整頁 all-or-nothing；較早 ROI 可以寫回成功、較晚 ROI 可以
rollback。

#### 已接線與未接線邊界

| 子系統 | `manga-translate run` 狀態 |
|---|---|
| durable 10 stages、store、cache、resume、replay | 已接線 |
| region revision、strict mapping、raw provider response artifact | 已接線 |
| safe region、RAQM solver、atomic ROI render | 已接線且預設啟用 |
| `stages.ocr.PageOCRStager` 頁級 batching／view cache | 未接線；production 使用 legacy group OCR |
| OCR token calibration framework | 未接線且沒有 fitted calibration artifact |
| panel detection、panel-aware order、manual override | 未接線 |
| `StructuredTranslationClient` 與 provider-side JSON schema | 未接線；production 使用 `translator.py` |
| job-level HTTP connection reuse | 未接線 |
| EntityLedger、translation memory、visual escalation、repair coordinator | 關閉或未接線 |
| style-driven font role selection | 未接線；production 固定 neutral sans |

### 5.3 Failure 與輸出語意

`main` 在 batch 前初始化 OCR，而且在逐頁 try/except 外；初始化失敗會中止整批，CLI 會把
`OCRInitializationError` 轉成非零的 `ClickException`。進入 page loop 後的例外則會把來源圖片
decode 後重新 encode 到正常輸出檔名，`run` 只印 failed count 而不 raise，所以這些頁面失敗時
orchestrator 可能看到 exit 0、正常檔名與已改變的 JPEG bytes；no-input、`test` 與 `detect-only`
另有各自的 exit-0 缺口。

`repair` 的 `run` 把 blocking failure 反映到 `PageResult`／`BatchResult`，未使用 `--allow-partial`
時可 exit 1。它會把來源路徑當下的檔案以 `shutil.copyfile` 複製到
`output/failed/<stem>.source-preserved.<ext>`，並移除正常輸出；通常可 byte-for-byte 保留來源，
但不是從已雜湊的 SOURCE artifact materialize。若輸入在執行中被替換，副本可能與已記錄的
source SHA 不一致。

`layout_rejected` 與 `layout_collision_rejected` 在 `repair` 被列為 non-blocking；即使 0 個 ROI
寫回成功，頁面仍可能 `succeeded` 並重新 encode JPEG。`test` 找不到檔與 `detect-only` 解碼失敗
也仍直接 return，留下 exit 0。故「batch succeeded」只能表示沒有 blocking failure，不能表示
完成翻譯或媒體 byte-identical。

## 6. 現有驗證證據與不能外推的部分

先區分每組 evidence 實際涵蓋的元件；「報告稱 real」只表示歷史報告如此記錄，本次沒有獨立
重現：

| Evidence | CTD | OCR | provider／譯文 | 人工評估 |
|---|---|---|---|---|
| `main` unit tests | 無真 CTD forward | fake backend／synthetic | 無付費 API；固定資料 | 無正式評估 |
| `main` fixed visual | 報告稱真 input／CTD／mask | 真模型未證明 | 固定譯文；未呼叫 OpenRouter | 報告內視覺檢查，無盲評 rubric |
| 歷史 `repair` tests | model／API／GPU markers 預設排除 | 主要 unit／contract | 主要 synthetic | 無正式評估 |
| 歷史 `repair` fixed visual | fixed boundaries | fixed text，不測 live OCR | 固定譯文 | 報告記錄使用者核准；無盲評 rubric |
| 歷史 `repair` 前五頁 | 報告稱 real | 報告稱 real | 報告稱 real provider | 無正式盲評 |
| 歷史 `repair` 第六頁 | 報告稱 real | 報告稱 real | report 記錄 real provider，含一次 retry | 無正式盲評 |

| 對象 | 已提交結果 | 可支持的窄結論 | 不可支持的結論 |
|---|---|---|---|
| `main@ba0e3a4` tests | `compileall` passed；67 tests passed | unit／synthetic cases 曾通過 | 乾淨 Conda + Poetry 可重現、真模型／API 品質 |
| `main` fixed visual | 5 頁、38 groups 全有 layout plan；字級比 `.987–1.029`；collision 0 | 固定 boundary／譯文下 legacy layout planning 可行 | live OCR、provider、E2E 寫回率；它不是 38/38 production writeback |
| 歷史 `repair@1e58b46` tests | 516 passed / 2 failed / 1 skipped / 2 deselected | 多數 unit／contract 有覆蓋 | `repair@2953d04` 全綠 |
| 歷史 `repair` fixed visual | 報告記錄使用者核准 38/38 `new_better` | 該固定 corpus 的偏好記錄 | 盲評或一般化品質；評分 rubric 未提供 |
| 歷史 `repair` 前五頁 | 59 groups 中 2 個 ROI 寫回成功 | safety gates 拒絕 57 groups；報告記錄拒絕區保留來源 | post-render rollback 分支被觸發、false erase 為 0、兩個寫回皆正確、`2953d04` 寫回率 |
| 歷史 `repair` 額外第六頁 | 17 groups 中 0 個寫回；JPEG MAD `.0479`、max diff `3` | no-op 能完成 stages；重新編碼仍改 bytes | 與前五頁是同質 corpus、byte-exact 或所有 ROI rollback 路徑已覆蓋 |

前五頁的 2/59 與額外頁的 0/17 可以算成總計 2/76，但兩組用途不同，不能把 2/76 當成
單一同質 benchmark。74 個未寫回 groups 的歷史分類是 66 layout、4 collision、3 OCR、
1 safe-confidence；其中 `LayoutOverflow:shaping_failed` 又把 tracking、safe mask、font floor、
geometry constraint 與真 shaping error 混在同一 reason，尚不足以判定根因。

歷史報告測的是 `1e58b46`。其後 `4b78fcd`、`a1938c6`、`d59a3de` 修改 RAQM request、safe seed
與 solver performance，但沒有找到 `repair@2953d04` 的完整 tests + live visual 重跑。因此只能說
目標快照的真實寫回率是 **U（未知）**，不能沿用 2/76，也不能假定後續 commits 已修好。

`main` 的真實 inputs 被 ignore，完整 validation script 不在 tree；歷史 OCR 測試使用 fake
backend。`repair` 的真實頁面與 provider run 也沒有整理成可分享、匿名化、可由第三方重跑的
evidence pack。現有材料缺少一致的全漏斗統計：

```text
detected
  → grouped
  → OCR accepted
  → translation mapped and validated
  → safe region accepted
  → layout accepted
  → ROI writeback succeeded / rolled back
  → page succeeded / source-preserved
```

## 7. 依優先級整理的 findings

### P0-1：`repair@2953d04` 尚無該 SHA 的 live E2E 證據

- **證據：H + U。** 歷史 live 結果如第 6 節；目標 SHA 在三個會改變 layout 行為的 commits 後
  沒有等價重跑。
- **影響：** 目前無法量化 safety gate 是否過嚴，也無法比較 `main` 與 `repair` 的 production
  utility；是否足以發布仍取決於 owner 尚未定義的 gate。
- **需要：** 先凍結 corpus、環境、config 與 SHA，產生逐 stage funnel 和具體 rejection code，
  再調任何 threshold。

### P0-2：`main` 的權威環境政策與實作互相矛盾

- **證據：F。** `AGENTS.md` 要 Conda + Poetry、禁止 pip；`main` 的 README／guide／
  `environment.yml` 使用 pip，build backend 是 setuptools，沒有 `poetry.lock` 或真正的 dev
  dependency group。
- **影響：** 第三方無法從乾淨機器重建唯一環境，tests 與模型行為也無法可靠比較。
- **需要：** 若保留 `main`，先完成與 `repair` 相同的 Conda-native + Poetry ownership、lock、
  clean-room install 與 CI；否則它不應作為 release 基線。

### P0-3：status 與檔名不能誠實代表結果

- **證據：F。** `main` 的部分 page-loop 失敗可 exit 0，並把 fallback 放在正常檔名；`repair`
  只改善 `run` 的 blocking status，其他子命令仍有 exit-0 缺口。`repair` 的 no-op page 又可標成
  `succeeded`。
- **影響：** CI、批次工具與使用者會把「流程沒 crash」誤認為「翻譯成功」。
- **需要：** 分開報告 detected、translated、layout accepted、寫回成功、rollback、no-op、
  source-preserved；所有 CLI fatal input／decode error 非零退出，且 failure output 不使用成功檔名。

### P0-4：`main` 的翻譯 mapping 可能 fail open

- **證據：F（mapping 行為）／R（實際錯配率）。** parser 接受多種鬆散格式與位置 fallback，
  duplicate ID 可被覆寫，group ID 又不穩定且沒有 durable source hash。
- **影響：** provider 格式錯誤可能把合法但錯誤的譯文寫到另一個 ROI，違反來源保護的精神。
- **需要：** 不論基線選擇，都應保留 `repair` 的 exact request/item/source mapping 與 raw response
  artifact；另評估 strict whole-request rejection 是否可安全縮小為 item-level retry。

### P0-5：`repair` 的架構文件、tests 與 production path 分裂

- **證據：F。** 第 5.2 節列出的 page OCR、calibration、panel order、structured client、font role
  等模組存在但 production 未呼叫。
- **影響：** 若依 dormant 能力選擇 `repair`，基線比較本身就會失真；維護者也可能修到錯的
  subsystem，測試綠燈不代表 CLI 行為改變。
- **需要：** 對每個 dormant 元件做「接線／刪除／明確延後」決策，建立 production-call-path
  integration test，避免繼續增加平行抽象層。

### P1-1：`repair` 公開的 provider 隱私／schema 設定未在 production 生效

- **證據：F。** `OpenRouterConfig` 有 `data_collection="deny"`、`zdr=True`，production `_payload()`
  只送 model、messages、temperature、max_tokens；沒有 provider-side `response_format` schema。
  具備這些能力的 `StructuredTranslationClient` 未接線。
- **影響：** checked-in 設定會讓維護者誤以為已有隱私與 schema guarantee；目前硬性契約只有
  「圖片不送 provider」，ZDR／data collection deny 是否為 release requirement 仍需 owner 決定。
- **需要：** 接線或刪除 dead contract，並以實際 request artifact 驗證；在此之前不能宣稱
  ZDR／data collection policy 已落實。

### P1-2：兩個快照都遺失 CTD confidence

- **證據：F。** vendored `group_output()` 迴圈取得 `conf` 卻沒有存入 `TextBlock`；
  `TextBlock.__init__` 固定 `self.prob = 1`，adapter 再把 `prob` 當 detector confidence。
- **影響：** region reliability、durable detector score 與 manifest 幾乎固定 1.0，不能校準、排序
  或解釋 false positive。
- **需要：** 保留原 detector score，為 merge/split 定義 aggregation，並用真 positive／negative
  corpus 校準；不要只改欄位名稱。

### P1-3：production OCR 仍是 heuristic、逐 group，且 `repair` config 與 evidence 未接線

- **證據：F。** 兩者 production 都逐 group 執行多 views；`repair._get_model()` 直接建立
  `MangaOcrRuntime()`，沒有傳 `model_id`、`revision`、`batch_size`、`max_length`。checked-in 值目前
  恰好等於 runtime defaults，但修改 config 只會改 fingerprint，不一定改行為。runtime 已能輸出
  token logprob、entropy、margin、truncation，document 卻只保存 heuristic score。
- **影響：** threshold 不能由模型證據校準，重複 inference 成為熱點，cache identity 也可能失真。
- **補充更正：** 純 Latin 分數不是「最高 `.38`」。公式是 `.18 + .18 + length bonus`，`.38`
  只是保底，長字串可到 `.46`；但常見短 Latin SFX 多數仍低於一般 `.46` gate，1–2 字又受
  `.66` short gate，存在系統性漏收風險。
- **需要：** 接線 page-level batching、保存 token evidence、建立人工 crop／no-text corpus，對日文、
  短字、英文 SFX、furigana 分層校準。

### P1-4：兩者 production reading order 都不理解漫畫分鏡

- **證據：F（演算法）／R（品質影響）。** production 只有全頁中心座標排序；`repair` 的 panel
  modules 與 override 未接線。
- **影響：** 跨 panel 次序與 page-context translation 可能互相污染；實際比率尚無 annotated
  corpus。
- **需要：** 先建立 panel/order ground truth，再決定接線現有模組或重做；translation item
  identity 不應因純 reorder 改變。

### P1-5：pairwise grouping／dedup 可能形成 transitive bridge

- **證據：F（union-find 行為）／R（錯誤合併率）。** A–B、B–C 各自通過 pairwise gate 時，
  A／B／C 會形成同一 component，即使 A 與 C 其實分離；缺少 component-level bubble／panel
  constraint。
- **影響：** 錯誤但通過 gate 的 group 可能合併 OCR、translation 與 union mask；fail-safe 能保護
  被拒 group，不能保護「錯誤但被接受」的 hypothesis。
- **需要：** 保存 union edge／score trace，建立 split／merge ground truth，並在 component 完成後
  再做 bubble、panel、mask 與 reading-order consistency validation。

### P1-6：layout 的安全 gate 與真實 glyph／來源比例仍有落差

- **證據：F（實作限制）／R（畫面發生率）。** `main` 以主字體與理論 bbox plan，render 時可切
  fallback font；直排不是完整 CJK typography。`repair` 雖 actual-raster verify，但 production
  固定 neutral sans，`shaping_failed` reason 過載。兩者都先以 absolute `font_size_max=180` 裁切
  來源估計，再套 `.85` 比例：來源若估 240 px，`repair` 可接受 153 px，只有原估計的 63.75%。
- **影響：** 可能錯誤接受過小字、錯誤拒絕可行 layout，或無法知道低寫回率是 safe mask、font
  floor、line count、collision 還是真 shaping。
- **需要：** 對每個 candidate 保存原始字級、未裁切比例、每道 constraint 結果、actual glyph
  bbox 與 safe/occupied overlay；分離 rejection codes 後才調 threshold。

### P1-7：公開 config 與 cache identity 仍有 no-op／漂移

- **證據：F。** 兩個快照的 Pydantic models 都未設 `extra="forbid"`，拼錯 YAML key 會被忽略；
  `api_key` 是 required，若整個 YAML key 缺失，即使環境變數存在也會先 validation fail。
  `main` 另有多個未讀或語意不符的 typesetting fields。`repair` 的 stage fingerprint 對 OpenCV
  package 名、STYLE／LAYOUT dependencies 與 native RAQM stack 記錄不完整。
- **影響：** 使用者以為調參生效，或 dependency 升級後錯誤命中舊 cache。
- **需要：** fail-fast schema、effective-config dump、每欄位 production read test，以及完整
  code/config/model/font/package/native-library fingerprint。

### P1-8：`repair` 的 sanitizer 行為與文件漂移

- **證據：F。** production sanitizer 只做 display normalization，會保留線條、額外省略號／
  破折號與長重複句；部分 README 與 test 名稱仍描述 `main` 的 source-aware 清理。
- **影響：** 維護者可能依 test 名稱或 README 判斷錯誤的 production contract；是否造成畫面線條
  regression 仍是 R，不能只由程式差異宣稱已重現。
- **需要：** 先決定產品規則應屬 translation validator 或 typography，再同步 code、tests、名稱與
  文件。

### P1-9：媒體 fidelity 與失敗來源的一致性不足

- **證據：F。** 兩者以 OpenCV color decode／encode，不能完整保留 PNG/WebP alpha、EXIF、ICC
  與 metadata；`repair` 的成功 no-op JPEG 仍會改 bytes。blocking failure 又從 mutable source path
  複製，而不是從已雜湊 artifact 回存。grayscale／BGRA 的下游相容性缺真實 fixtures。
- **影響：** 無視覺修改的頁面仍可能改媒體；執行中來源被替換時，failed copy 可能與記錄的
  source SHA 不同。
- **需要：** no-op 與需保留來源的 failure 直接 materialize hash-pinned source；定義並測試
  alpha／metadata policy 與 grayscale／BGRA fixtures。

### P1-10：durable state 缺少 retention 政策

- **證據：F。** `repair` 保存 source、masks、stage states、PNG、raw provider responses 與 SQLite，
  只有手動 `cache gc`，沒有自動 retention policy。
- **影響：** 私人漫畫與 provider response 可能在本機、備份或共用磁碟長期留存，磁碟也會持續
  成長。
- **需要：** owner 決定保存期限與稽核需求；實作 retention、secure delete、權限、容量 telemetry
  與可驗證的 GC policy。

### P1-11：build 與供應鏈證據不足

- **證據：F。** 兩個 wheel 都只封裝 `src/manga_translator`，根目錄 models、fonts、config、
  glossary 不在 wheel。`repair` 的歷史報告明載 wheel／release ZIP 驗證被取消。detector 權重由
  `torch.load()` 讀 pickle，download path 缺 SHA-256／signature 驗證；兩個快照都沒有 CI workflow。
- **影響：** 換目錄安裝可能找不到預設 assets，乾淨發布與供應鏈完整性無持續證據。
- **需要：** 定義 asset acquisition／checksum、第三方授權與 notice、clean-room build、wheel/ZIP
  smoke 和 CI。`repair` 已有 `THIRD_PARTY_NOTICES.md`，仍需驗證其涵蓋度；不能把 `main` 的缺口
  誤寫成兩者完全相同。

### P2：已知效能與維護成本

- **證據：F；FP16 parity 數字為 H。** detector 預設雙尺度 FP32；歷史只有 5 頁 FP16 parity，
  尚未形成 release gate。
- **證據：F。** OCR 逐 group，translation 每頁建立 event loop／`AsyncClient`，多處 grouping／
  collision 為 O(n²)。
- **證據：F。** `repair` 約 705 tracked files，包含大量 benchmark 圖片與 profiler JSON；audit
  價值與 clone／diff
  成本尚未分層。
- **證據：F。** legacy 與 RAQM、legacy translator 與 dormant structured client 同時存在，增加
  contract 與 tests 漂移面積。

這些項目影響延遲、資源與維護成本，但不應先於 P0/P1 的 correctness 與 observability。請在
取得 stage timing、VRAM、API cost 與 corpus 規模後，再決定 FP16、batching、connection reuse、
indexing 或 evidence 外置的順序。

## 8. 需要先建立的診斷與驗證資料

第三方目前沒有圖片、mask 或逐 candidate trace，不能直接判定歷史低寫回率的根因。建議先由
maintainer 產出不含 API key、可分享或匿名化的 evidence pack。

### 8.1 每頁與每 group 的最小 telemetry

- snapshot SHA、effective config hash、environment fingerprint、source SHA、media type／尺寸。
- detector pass、原始 confidence、region／group lineage、bbox／mask area 與 reading order。
- OCR views、token evidence、heuristic score、accept／reject reason、是否 cache hit。
- translation request/item/source hashes、provider attempt、mapping／validation result；譯文可匿名化。
- safe-region confidence、mask support、protected pixels、edge／distance summary。
- 每個 layout candidate 的原始與裁切字級、方向、line／column count、actual raster bbox、
  containment、collision 與單一明確 rejection code。
- ROI 寫回成功／rollback、殘留原字檢查是否實際執行、輸出像素／metadata 差異。
- stage duration、peak VRAM／RAM、API request count／tokens／cost、artifact bytes 與 cache hit。

### 8.2 最小 evidence pack

1. 匿名化 source crop、detector mask、safe mask、occupied mask、最佳失敗 candidate 與 output crop。
2. 每組歷史數字的 exact SHA、config、命令、input/artifact checksums 與 raw summary。
3. 清楚標示每個 case 使用 real 或 fake CTD、OCR、provider、translation text。
4. 至少包含：一般直排對白、橫排、英文 SFX、furigana、跨 panel、深色／網點背景、極大字、
   fallback glyph、PNG alpha、grayscale、JPEG EXIF／ICC 與 blocking failure。
5. 對 OCR、reading order、翻譯品質與 false erase 分別建立人工 ground truth；fixed layout corpus
   不能代替 live E2E corpus。

### 8.3 可重現性與最低 release 條件

所有命令須在 `conda activate manga` 後執行：

```bash
poetry run manga-translate doctor --config config.yaml --strict-api-key
poetry run pytest -q
poetry run ruff check .
poetry build
poetry run manga-translate test --image <approved-fixture> --dump-json
```

在 owner 補上數值門檻前，至少應要求：

- 表列 SHA 的 full suite、lint、build 與 clean-room install 全綠。
- 已標註 corpus 中 false erase、跨 group mapping、glyph overflow 為 0。
- no-op、partial success 與 blocking failure 有可區分的 status／輸出位置。
- fatal error 非零退出；需要保留來源時由 hash-pinned artifact materialize。
- 提供完整 funnel，而不是只報「pages succeeded」或 tests 數量。
- 真模型／provider 測試與 fake／fixed fixture 分開報告；付費測試需事前批准。
- owner 明確核准寫回率、人工翻譯品質、延遲、VRAM、API 成本與媒體 fidelity 門檻。

## 9. 請第三方優先回答的技術問題

### 必答：基線與 release blockers

1. 應以哪個 snapshot／第三種方案為 canonical production path？請提出 migration sequence，
   說明哪些 duplicate／dormant subsystem 應接線、刪除或延後，並列出使決策改變的 owner 參數。
2. 如何用同一 corpus 在 `repair@2953d04` 重跑完整漏斗，產生新的 rejection taxonomy，再與
   `1e58b46` 的 66 個歷史 layout rejects 對照？請拆分 safe mask、font floor、line count、
   geometry、collision、glyph 與真 shaping 原因；現有 evidence 不足時，先設計量測。
3. 從 P0 findings 選出前三個 release blockers，提出最小修復／驗證順序與 provisional gate；
   同時標示哪些 gate 必須等待 owner 決策。

### 選答／後續 workstreams

4. OCR 如何接線頁級 batching 與 token evidence，並建立涵蓋短字、英文 SFX 與 furigana 的
   calibration corpus？
5. reading order 如何加入 panel／bubble graph 與 manual override，同時讓 translation identity
   不因純 reorder 改變？
6. translation 如何落實 provider-side schema、ZDR／data policy、connection reuse、raw replay 與
   item-level recovery，而不削弱 exact mapping？sanitizer 規則應屬 validator 還是 typography？
7. RAQM solver 如何在不大幅縮字或跨出 safe region 的前提下提高寫回率？請依新 telemetry 評估
   safe morphology、font-role、CLREQ break、fallback glyph、outline、tate-chu-yoko 與 occupancy，
   不要只調 beam size 或整體放寬 gate。
8. 如何把環境、CI、asset checksum、license／notices、wheel／ZIP、media fidelity 與 durable
   retention 納入 release gate？請指出哪些項目需要 owner 先決定政策。

## 附錄 A：關鍵行為的最小示意 excerpts

以下 excerpt 只讓無 repo 讀者看見關鍵條件，不是完整 caller／schema／error-path 證明；若要做
line-level patch，仍須依所列 SHA、path、symbol 索取完整上下文。除 A.1 同時存在於兩個快照外，
其餘皆來自 `repair@2953d0401319cb905bee78e3c597b05deeeda43f`。

### A.1 兩者都遺失 detector confidence

Path：`src/manga_translator/ctd/utils/textblock.py`，`TextBlock.__init__` 與 `group_output`
（`repair` 約 lines 60、428–430；`main` 有相同行為）。

```python
# ctd/utils/textblock.py
self.prob = 1

for bbox, cls, conf in zip(*blks):
    # conf is not passed to TextBlock
    blk_list.append(TextBlock(bbox, language=LANG_LIST[cls]))
```

### A.2 `repair` 的 OCR config 未傳入 runtime

Paths：`src/manga_translator/config.py:125–134`、`src/manga_translator/ocr.py:76–85`。

```python
# config.py
class OCRConfig(BaseModel):
    model_id: str = Field(default="kha-white/manga-ocr-base", min_length=1)
    revision: str = Field(
        default="aa6573bd10b0d446cbf622e29c3e084914df9741",
        min_length=40,
        max_length=40,
        pattern=r"^[0-9a-f]{40}$",
    )
    batch_size: int = Field(default=4, ge=1, le=64)
    max_length: int = Field(default=300, ge=2, le=1024)

# ocr.py
def _get_model() -> MangaOcrRuntime:
    ...
    model = MangaOcrRuntime()
```

### A.3 兩者 production OpenRouter payload 的有效欄位

Paths：`src/manga_translator/config.py:13–28`、`src/manga_translator/translator.py:516–528`。

```python
# repair config.py
data_collection: Literal["deny", "allow"] = "deny"
zdr: bool = True

# main and repair production translator.py
return {
    "model": cfg.model,
    "messages": [{"role": "user", "content": prompt}],
    "temperature": cfg.retry_temperature if retry else cfg.temperature,
    "max_tokens": max_tokens,
}
```

### A.4 `repair` 把 layout rejection 視為 non-blocking

Path：`src/manga_translator/pipeline.py:1658–1660, 2669–2681`。

```python
def _first_blocking_group_issue(issues):
    non_blocking = {"layout_rejected", "layout_collision_rejected"}
    return next((issue for issue in issues if issue.code not in non_blocking), None)

blocking = _first_blocking_group_issue(group_issues)
return PageResult(
    ...,
    status="blocked" if blocking is not None else "succeeded",
)
```

### A.5 `repair` 的失敗副本取自 mutable source path

Path：`src/manga_translator/pipeline.py:4094–4100`。

```python
def _preserve_failed_source(page, output_dir):
    fallback_path = _failed_output_path(output_dir, page.source_path)
    (output_dir / page.source_path.name).unlink(missing_ok=True)
    fallback_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(page.source_path, fallback_path)
```

## 附錄 B：索取後續 excerpt／trace 時的程式位置

第三方無法直接使用這些路徑；它們的用途是精確指出 maintainer 應補哪段 excerpt 或 trace。

| 主題 | `main` | `repair` |
|---|---|---|
| orchestration | `pipeline.py: run_pipeline, process_single_page` | `pipeline.py: _build_pipeline_stage_runners, process_single_page_staged` |
| DAG / cache | 無 | `stages/runner.py: STAGE_DAG, StageRunner`; `stages/adapters.py` |
| state / identity | 無 | `storage/`; `domain/models.py`; `domain/reconcile.py` |
| detector / grouping | `detector.py`; `ctd/` | `detector.py`; `stages/detect.py`; `ctd/` |
| production OCR | `ocr.py`; `manga_ocr_runtime.py` | 同左；dormant `stages/ocr.py`, `ocr_confidence.py` |
| order | `pipeline.py: _refresh_group_order` | 同左；dormant `reading_order.py`, `order/panels.py` |
| production translation | `translator.py` | `translator.py`; dormant `translation/client.py` |
| layout | `typesetter.py` | `typography/layout.py`, `solver.py`, `safe_region.py`, `fonts.py` |
| render / inpaint | `inpainter.py`; `typesetter.py` | `stages/render.py`; `typography/render.py` |
| config / CLI | `config.py`; `cli.py` | `config.py`; `cli.py` |
