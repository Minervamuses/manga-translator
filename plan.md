# manga-translator v0.3.2 收尾執行計劃

本計劃只處理四件事：補完 P0～P4、正確完成 P5、依序合併、維持 Conda `manga`＋Poetry 的單一環境政策。它不是新一輪重構提案，也不增加與驗收無關的流程。

## 1. 最終完成定義

完成時必須同時成立：

1. G0～G4 都是 `passed`，沒有 waiver、空白人工資料或把 mock 當真實結果。
2. P5-01～P5-08 都從正確的 `repair_p5` 完成；舊 `reapir_p5` 不進入歷史。
3. G5 的 30 頁／3 title 品質 corpus、CPU/GPU/provider doctor、lock、授權及 release smoke 全部通過。
4. 合併順序固定為：`repair_p0_p4_completion -> repair -> repair_p5 -> repair -> main`。
5. 執行環境只有既有 Conda named environment `manga`；套件宣告與安裝由 Conda、Poetry 負責。

下列事項不在本次範圍：刪除舊 branch/worktree、改用其他模型、為沒有量測依據的效能想法先寫架構、引入新的環境或套件管理器。

## 2. 目前基準與 branch 結論

本計劃以 2026-07-29 的本機狀態為起點：

- 專案：`/mnt/c/Users/garyc/Documents/本/manga-translator-v0.3.2-complete`
- `repair`：`5a7584b0073fafb34e31d4ce13d2f5c9e7bb04f4`
- `repair_p5`：同一 commit，尚未開始正確 P5。
- `repair` 已含 P0 與已合併的 P1～P4 元件工作。
- `reapir_p5` 與 `repair` 已分岔，禁止 merge、rebase 或整支 cherry-pick。
- `repair_p1_*`、`repair_p2`、`repair_p3`、`repair_p4` 及所有 `reapir_*` 都只保留作稽核參考，不再合併。
- 根 worktree 目前只有未追蹤的 `plan.md`；開始執行時不得還有其他變更。

因此真正的活動工作流只有兩條：

1. `repair_p0_p4_completion`：補齊 P0～P4。
2. `repair_p5`：在第一條完成並合回 `repair` 後，快轉到最新 `repair` 再做 P5。

`git branch` 仍看到接近十個項目是正常的；那些是仍存在的 refs/worktrees，不代表都待合併。本次不自動刪除它們。

目前未完成狀態如下；這是執行起點，不是重新實作整個 phase：

| Phase | 已有成果 | 尚未完成 |
|---|---|---|
| P0 | 基線、mapping、failure semantics、detector/profiler 元件已存在 | 人工 truth 僅 2/38；FP16 檔只有門檻、沒有完整 real result；五頁 performance 仍是 mock；G0 未真正關閉 |
| P1 | `g1_page_document.json` 的 automated gate 為 passed | 必須用新測試確認唯一 durable truth/真 code fingerprint，並完成 real process kill/restart |
| P2 | typography/safe-region/layout/ROI 元件及大部分測試已存在 | `manga` 無 RAQM、required test 被 skip、盲評 0/30、production engine 未切換 |
| P3 | 真 batch runtime、OCR 元件及 offline real-model test 已存在 | corpus 0/300 text、0/300 no-text、0/3 titles；attempt/confidence 尚未完整持久化；GPU paired run與 production switch 未做 |
| P4 | structured client、entity、repair、visual 元件與 mock tests 已存在 | corpus 0/200、0/3 titles；production orchestration、paired human gate與 legacy 移除未做 |
| P5 | 正確 `repair_p5` 是乾淨起點 | 舊 `reapir_p5` 歷史不可用；P5-01～P5-08 必須在正確 branch 重做並通過 G5 |

## 3. 不可違反的執行規則

### 3.1 任務與 commit

- 一次只允許一個本計劃任務為 `in_progress`。
- 任務狀態只用 `pending`、`in_progress`、`blocked`、`done`。
- 每個下文列出的工作項至少一個獨立 commit；不得把兩個工作項塞進同一 commit。
- commit 前先通過該工作項的指定驗收；失敗時保持 `in_progress` 或 `blocked`，不得先標完成。
- 修正已提交工作項時，另建 `fix(<task-id>): ...` commit，不用 amend 改寫已驗收歷史。
- 缺人工 reviewer、GPU、provider credential 或法務核准時必須標 `blocked`；不得造資料、降門檻或以 mock 代替。
- 任何 mapping、OCR、翻譯或 layout 失敗，都必須在 inpaint 前停止；原文不得被擦除。

### 3.2 Conda／Poetry 邊界

允許的環境與套件流程如下：

- Conda named environment 固定為 `manga`，不得刪除或重建這個既有環境。
- Conda 管 Python、Poetry 本身、CUDA 與 RAQM/HarfBuzz/FriBiDi/FreeType 等原生函式庫及 Conda lock input。
- Poetry 管全部專案 Python dependency（包含 Pillow）、`poetry.lock`、安裝與 build；Poetry 必須直接使用 `manga` 的 Python。
- 同一 Python package 不同時由 Conda 與 Poetry 宣告；原生 library 由 Conda 提供，Pillow Python distribution 只由 Poetry 提供。
- `conda-lock` 只因原 spec 要求產生 Conda lockfile，且必須由 Conda 安裝及管理；它不是第三個環境管理器。
- 禁止建立 `venv` 或 `.venv`，也禁止 uv、PDM、Pipenv、mamba、micromamba。
- 禁止直接執行 `pip install`；`python -m pip check` 只能作安裝一致性診斷。
- 禁止 `poetry run`。所有程式、測試及 Poetry 命令都從 `conda run -n manga` 進入。
- `environment.yml` 不得有 `pip:` subsection；Conda lock 中不得有 `manager: pip` 項目。
- 任何 dependency 變更都要在同一工作項更新適用的 `environment.yml`、`conda-lock.yml`、`pyproject.toml`、`poetry.lock`、`poetry.toml`，並重跑環境驗收。

每次工作 session 先執行：

```bash
set -euo pipefail
export MT_REPO='/mnt/c/Users/garyc/Documents/本/manga-translator-v0.3.2-complete'
export MT_CONDA='/home/minervamuses/miniconda3/bin/conda'
export MT_P5_WORKTREE='/mnt/c/Users/garyc/AppData/Local/Temp/manga-p5-audit-019fa8a9'
unset VIRTUAL_ENV
export POETRY_VIRTUALENVS_CREATE=false
cd "$MT_REPO"

test -x "$MT_CONDA"
"$MT_CONDA" env list | grep -Eq '^[[:space:]]*manga[[:space:]]'
"$MT_CONDA" run -n manga python -c 'import sys; print(sys.executable); assert sys.prefix == sys.base_prefix'
"$MT_CONDA" run -n manga poetry --version
test ! -d .venv
```

`sys.prefix == sys.base_prefix` 在 Conda 環境中是預期值；這裡用來抓出疊在 Conda 上的 Python virtual environment。

### 3.3 每個程式工作項的共同 gate

在 commit 前於 WSL 專案根目錄執行：

```bash
PYTHONPATH=src "$MT_CONDA" run -n manga python -m compileall -q src tests
PYTHONPATH=src "$MT_CONDA" run -n manga python -m pytest -q -p no:cacheprovider
"$MT_CONDA" run -n manga ruff check .
"$MT_CONDA" run -n manga poetry check --lock
"$MT_CONDA" run -n manga python -m pip check
git diff --check
```

規則：

- 上述任一命令非零就不得 commit。
- GPU/API/人工 gate 另依工作項執行；共同 gate 通過不代表真實整合已通過。
- 與該工作項直接相關的測試不得 skip 或 deselect；GPU/API 測試若在共同 suite 未執行，必須在對應 gate 單獨通過。
- 測試數量記錄實際值，不把過去的 `67`、`279`、`458` 當成永遠固定的總數；原始 67-test baseline 仍須由 baseline verifier 驗證。

### 3.4 最小證據格式

只使用現有 benchmark manifest、gate JSON 與報告，不另建複雜的簽章或稽核系統。每個 gate 檔至少記錄：

- source commit、corpus hash、config/model/font hash；
- Conda environment、Python/Poetry/package 版本；
- 實際命令、退出碼與結果摘要；
- GPU/driver 或 provider/model（適用時）；
- reviewer ID、時間、item、判定及 adjudication（人工資料適用時）；
- `status` 與尚未解除的 blocker。

## 4. 開始 branch

只執行一次：

```bash
cd "$MT_REPO"
test "$(git branch --show-current)" = repair
test "$(git rev-parse HEAD)" = 5a7584b0073fafb34e31d4ce13d2f5c9e7bb04f4
test "$(git status --porcelain)" = '?? plan.md'

git switch -c repair_p0_p4_completion
git add -- plan.md
git diff --cached --check
git commit -m 'docs(plan): define P0-P5 completion and merge procedure'
test -z "$(git status --porcelain)"
```

若 branch 已存在，停止並先確認它是否正好從上述 SHA 建立；不得以 `-f` 覆寫。

## 5. 共用人工 corpus：只建立一次

為避免 P0、P2、P3、P4、G5 各做一套重複資料，建立一個可供各 gate 引用的品質 corpus：

- 固定回歸子集：現有 5 頁、38 groups，38/38 人工核對。
- 最終品質集：至少 30 頁、3 個 title。
- OCR：至少 300 個人工核對 text crops，加 300 個 no-text art crops。
- 翻譯：至少 200 個人工核對 units，含跨格、代名詞、專名、語氣、SFX、caption。
- train/dev/test 必須以 title 分割；同 title 不得跨 split。
- P2 的至少 30 groups 盲評可直接取 38-group 回歸子集，不再另造一套。
- OCR/翻譯 reviewer 可不同；翻譯 reviewer 必須具日文到台灣繁中能力。
- 人工分歧保留雙方結果與 adjudication，不得直接覆寫異議。
- 未達數量前，對應 gate 保持 `blocked`。

## 6. P0～P4 收尾工作

執行順序固定為 ENV-01、P0-C1～C4、P1-C1～C2、P2-C1～C4、P3-C1～C4、P4-C1～C4、G04-FINAL。前一項未 `done` 不開始下一項。

### ENV-01：先讓既有 `manga` 具備可驗收的 RAQM runtime

目前問題：`manga` 中 Pillow runtime 與 Conda 記錄漂移，且 Pillow 沒有 RAQM；P2 因此有一個 required skip，不能等到 P5-07 才處理。

行動：

1. 先把 `conda list -n manga --explicit`、`conda list -n manga`、`poetry show`、Pillow feature 結果寫入 `benchmarks/environment/pre_reconcile/`。
2. 在 `environment.yml` 固定 Python 3.11、Poetry 2.2.1 與 RAQM/HarfBuzz/FriBiDi/FreeType 等原生相依；不得在此重複宣告 Pillow，也不得加入 `pip:` subsection。
3. 先以現有 lock 的 Pillow 11.3.0 作唯一 target；`pyproject.toml` 必須接受此版本，`poetry.lock` 保持 11.3.0。不得接受目前 runtime 12.3.0 與 lock 11.3.0 的漂移；若 11.3.0 加 Conda FriBiDi 後仍無 RAQM，ENV-01 直接 `blocked`，先查明 wheel/native loading，不任意改版碰碰運氣。
4. 建立 `poetry.toml`，固定 `virtualenvs.create=false`、`virtualenvs.in-project=false`。
5. 將 `conda-lock=4.0.0` 列入 Conda 開發依賴；先檢查 Conda dry-run，確認只加入該工具及其必要依賴後，再由 Conda 安裝：

```bash
"$MT_CONDA" install -n manga --dry-run --json -c conda-forge conda-lock=4.0.0
"$MT_CONDA" install -n manga --yes -c conda-forge conda-lock=4.0.0
```

安裝後產生 lock：

```bash
"$MT_CONDA" run -n manga conda-lock lock \
  --conda "$MT_CONDA" --no-mamba --no-micromamba \
  -f environment.yml -p linux-64
```

6. 確認 `conda-lock.yml` 只含 `manager: conda`，再把 explicit lock render 到 repo 外：

```bash
MT_LOCK_TMP="$(mktemp -d)"
"$MT_CONDA" run -n manga conda-lock render -k explicit -p linux-64 \
  --filename-template "$MT_LOCK_TMP/conda-{platform}.lock" conda-lock.yml
"$MT_CONDA" install -n manga --dry-run --json --file "$MT_LOCK_TMP/conda-linux-64.lock"
```

7. 人工檢查 dry-run transaction；只有計劃內的新增、升降版、替換可接受。若出現未規劃移除、CUDA/Torch 來源切換或 `manga` 重建，停止並修 lock。
8. dry-run 合格後才以同一 explicit file 執行；禁止對既有 `manga` 執行 `conda-lock install`：

```bash
"$MT_CONDA" install -n manga --yes --file "$MT_LOCK_TMP/conda-linux-64.lock"
```

9. 確認 Poetry Python realpath 等於 `manga` Python，才執行 Poetry install：

```bash
MANGA_PY="$("$MT_CONDA" run -n manga python -c 'import os,sys; print(os.path.realpath(sys.executable))' | tail -n 1)"
POETRY_PY="$("$MT_CONDA" run -n manga poetry env info --executable | tail -n 1)"
test "$(realpath "$POETRY_PY")" = "$MANGA_PY"
export POETRY_CACHE_DIR="$(mktemp -d)"
"$MT_CONDA" run -n manga poetry install --no-interaction
```

10. 重跑 feature/版本檢查，確認 runtime Pillow 等於 Poetry lock，Pillow 位於 `manga`，且所需 RAQM/HarfBuzz/FriBiDi/FreeType runtime 可從 Conda prefix 解析。

驗收：

- `features.check_feature("raqm")` 為 true，P2 shaping 測試 skip=0。
- `poetry env info --executable` 與 `manga` Python 是同一檔案。
- 不存在 `.venv`/`VIRTUAL_ENV`，也沒有第二個 Python env。
- runtime import 與 Poetry lock 都是 Pillow 11.3.0；Conda dependency input 不重複宣告 Pillow。
- 共同 gate 全過。

Commit：`build(ENV-01): align manga RAQM runtime with Poetry locks`

### P0-C1：封死 mapping 與 no-erase 漏洞

行動：

1. 追蹤 detect→OCR→translate→layout→render 的 production call path，確認只用 exact ID，不以 list position、`zip()` 或重複 ID 後值覆蓋作對應。
2. 加入 missing、duplicate、unknown、extra、reorder、swap、跨頁 ID mutation tests。
3. 強制只有 mapping、OCR、translation、layout 全有效的 group 才能建立非零 inpaint mask。
4. 在 production 邊界封閉可關掉 safe-no-erase 的設定；舊 renderer 的替換與刪除留給 P2，不在此重寫。
5. 用 stub renderer 測 inpaint、render、encode failure；pipeline 不得提交該 group 的擦除結果，inpaint mask 必須為零。

驗收：

- 合法 mapping 100%；所有破壞 mutation 在 render 前非零失敗。
- 任一 group failure 都保留原文；safe-no-erase 100%。
- render/inpaint failure 不留下空框；P2 再驗實際 ROI renderer 的 pixel-level atomicity。
- `0188` g004/g005 的正確對應仍被 regression test 固定。

Commit：`fix(P0-C1): enforce exact mapping and atomic no-erase rendering`

### P0-C2：先完成唯一可執行的 real evidence runners

目前缺口：`detector_fp16_parity.json` 只有 thresholds；GPU pytest 不會保存 metrics；performance 的 `real_run.runner_status` 仍是 `not_implemented`。在這些工具修好前不得宣稱跑過 real gate。

行動：

1. 為 benchmark CLI 增加唯一 parity 入口：`detector-parity --profile regression_v032 --output benchmarks/detector_fp16_parity.json --require-real`。
2. parity runner 必須實際各跑 FP32/FP16 五頁，寫 boxes/scores/mask/small-text 統計、model/config/source/environment hash；缺 CUDA/model/五頁任一項時非零退出且不覆蓋舊 evidence。
3. 完成 performance real runner，接真 detector、真 OCR、provider、layout/render/encode；增加 `--require-real`，real run blocked 時 CLI 必須非零退出。
4. runner protocol 固定每頁 1 cold、2 warmup、5 measured；mock 只供 unit test，不能出現在 real measurements。
5. 為成功、缺 GPU、缺 model、缺 API authorization、partial write 建 deterministic tests。

驗收：下列兩個命令都已存在、`--help` 正確、blocked 時非零、成功時原子寫完整 evidence；共同 gate 全過。

```bash
PYTHONPATH=src "$MT_CONDA" run -n manga python -m manga_translator.benchmark detector-parity --help
PYTHONPATH=src "$MT_CONDA" run -n manga python -m manga_translator.benchmark performance --help
```

Commit：`feat(P0-C2): implement fail-closed real benchmark runners`

### P0-C3：完成 38-group truth、真實 parity 與五頁 baseline

行動：

1. 由 contact sheet 逐筆人工核對 38 groups，填真實 `verified_by`、`verified_at` 與必要修正；現有兩筆也要重新看圖，不得只以 `execution-spec` 或「fixture 存在」當人工 verified。
2. 保留 `0188`：g004 是自我介紹／`セシリー・キャンベル`；g005 是「男が酒場で…」。
3. 在 RTX 5070 Ti、相同 detector model/config 跑五頁 FP32/FP16 parity；門檻為 box count ratio ≥0.90、mean IoU ≥0.85、matched score MAE ≤0.05、mask IoU ≥0.90、small-text recall ≥0.90。
4. 用真 detector、真 OCR、獲准的 provider 跑五頁 baseline；manifest 必須 `mock=false`，保存 run ID、hash、p50/p95/worst、VRAM/RSS/API timing。
5. 執行唯一命令：

```bash
PYTHONPATH=src "$MT_CONDA" run -n manga python -m manga_translator.benchmark validate regression_v032 --require-verified
PYTHONPATH=src "$MT_CONDA" run -n manga python -m manga_translator.benchmark detector-parity --profile regression_v032 --output benchmarks/detector_fp16_parity.json --require-real
PYTHONPATH=src "$MT_CONDA" run -n manga python -m manga_translator.benchmark performance --profile v032_baseline --require-real
PYTHONPATH=src "$MT_CONDA" run -n manga python -m manga_translator.dev verify-baseline --mode regression
```

驗收：38/38 verified、parity 全達門檻、real baseline 完整、mock sample 不計入 real 指標、四個命令全為 0。

Commit：`test(P0-C3): record verified real baseline evidence`

### P0-C4：關閉 G0

行動：重新執行 P0 targeted tests與共同 gate；建立/更新 `benchmarks/gates/g0_correctness_baseline.json`，引用 P0-C3 的 exact run IDs/hashes。只有 blockers 空、所有結果仍匹配目前相關 fingerprint 才設 `passed`。

驗收：G0 validator 非零拒絕缺 reviewer、缺 real metrics、threshold failure、mock-as-real 或 stale fingerprint；正常資料通過。

Commit：`test(P0-C4): pass G0 correctness baseline gate`

### P1-C1：確認 PageDocument 是唯一 durable truth

行動：

1. 先新增 architecture tests，從正式單頁與 batch entrypoint 執行 detect→encode。
2. 每個 stage 必須讀寫同一個持久 PageDocument revision；DetectionResult/TextGroup/TextRegion 只可作 stage-local adapter，不可形成第二份 durable state。
3. 建立 persistent group identity/revision；group-level OCR、translation、layout result 只能持久化一次，再以關聯連到所有 member region revisions。禁止把同一結果複寫到每個 member、replay 時再任取第一筆。
4. 增加 multi-region group round-trip/resume test，驗 group member order 改變也不會重複或交換 OCR/translation record。
5. debug manifest、inspect、replay 都由 PageDocument 產生；禁止尾端才重建 PageDocument。
6. 對 source、code、config、model、font、prompt、schema、glossary 分別做 mutation test；fingerprint 必須只失效該 stage 與 downstream，不得使用固定字串假裝 code revision。
7. 若現有實作已通過上述新測試，只提交測試；若失敗，修到通過，不重寫已正確元件。G1 狀態留到 P1-C2 的 process restart 完成後再更新。

驗收：canonical serialize round-trip bytes 相同、group result 僅一份、無永久雙寫、mutation invalidation 精確、舊/錯 fingerprint cache 被拒絕。

Commit：`fix(P1-C1): make PageDocument and real fingerprints authoritative`

### P1-C2：完成真實 crash/restart/replay 驗收

行動：

1. 用子程序在 detect 後、OCR 後、provider response 保存後、render 前各強制 kill 一次。
2. 重新啟動並 `--resume`；記錄 detector load/forward、OCR forward、provider request 次數。
3. 對已成功 stage，resume 額外呼叫數必須都是 0；中斷 stage 只能重做未原子提交的部分。
4. 斷網且不載模型執行 replay；輸出 canonical PageDocument 與 uninterrupted run hash 必須相同。
5. 更新 `benchmarks/gates/g1_page_document.json`，移除 real process restart 的 deferred validation 後才維持 `passed`。G1 用可計數的本機 provider transport 證明不重呼；真 provider capability 留在 G5 doctor 驗證，不讓 P1 不必要地依賴付費 API。

驗收：四個 kill point 可續作、無重複 provider request、replay 無網路/模型、輸出 hash 相同、共同 gate 全過。

Commit：`test(P1-C2): prove process restart and replay correctness`

### P2-C1：完成 RAQM 排版與 production-safe renderer 接線

行動：

1. 在 ENV-01 的 RAQM runtime 跑 shaping、UAX #14/UAX #50、CLREQ、font fallback、safe-region、solver、ROI render 全套測試，required skip 必須為 0。
2. 修完仍走 codepoint render、`_balanced_chunks`、全白 clip mask、先整頁擦除、假 config key 的 production 路徑。
3. production renderer 只接受 shaped runs、合法 break、真 safe mask 與 accepted LayoutPlan。
4. hard constraints 固定為 glyph coverage、CLREQ violation=0、collision=0、無 clipping、alpha containment ≥99.5%。
5. 無解、缺 glyph、RAQM 失敗、safe confidence 不足都回 `LayoutOverflow`/typed issue 並保留原文。

驗收：直/橫排與標點 golden 正確、missing glyph=0、CLREQ hard violation=0、collision=0、alpha containment≥99.5%、ROI 外 diff=0、原子回滾通過。

Commit：`fix(P2-C1): wire RAQM layout and atomic ROI rendering`

### P2-C2：產生五頁視覺結果與完成盲評

行動：

1. 以 38/38 verified regression groups 重新產生 source overlay、safe mask、style、shaped runs、layout alpha、inpainted ROI、final preview、metrics。
2. 先跑 hard metrics；任一失敗時不得發盲評，也不得切 engine。
3. 對至少 30 個 groups 進行 blind A/B，評可讀性、字級、空隙、位置、風格、顏色/描邊、整體偏好。
4. reviewer 不得從檔名或 sheet 得知 A/B 身分；評完後才用 key 彙總。

驗收：38-group hard metrics 全過、盲評≥30、required skip=0、無 critical regression、candidate 可讀性與整體偏好都不低於 legacy。

Commit：`test(P2-C2): record typography metrics and blind review`

### P2-C3：切換 production engine 並移除 legacy

行動：只在 P2-C2 驗收全過後，把新 engine 設為唯一 production 路徑；刪除舊逐字 renderer、強迫填滿 objective，以及沒有作用的 layout config key。不得改 benchmark 結果或降低門檻來配合切換。

驗收：正式 entrypoint 只可達新 engine；舊 symbol/config 的 negative search test 通過；五頁輸出與 P2-C2 accepted candidate hash 相同；共同 gate 全過。

Commit：`feat(P2-C3): switch verified typography engine and remove legacy`

### P2-C4：關閉 G2

行動：在 P2-C3 tip 重跑 P2 targeted tests、五頁 hard metrics與共同 gate；更新 `benchmarks/gates/g2_typography_visual.json`，引用盲評 hash及 production switch source hash，blockers 空後才設 `passed`。

驗收：G2 validator 可拒絕 stale review、required skip、legacy reachable或任一 hard metric failure；正常資料通過。

Commit：`test(P2-C4): pass typography visual gate`

### P3-C1：補齊 staged OCR 的 durable state 與 production 接線

行動：

1. 固定 OCR model revision、processor/preprocess revision、generation config；空 revision 必須拒絕。
2. 確認 batch 1/2/4/8 是一次 processor/model batch，輸入輸出順序 100% 對應，不在 Python 逐 crop forward。
3. 將每個 attempted view、token score、provisional score、disagreement、confidence kind、OCR error/no-text decision 寫入 PageDocument/stage store。
4. resume 已完成 region 時 OCR forward=0；錯誤、不確定及 no-text 都可離線 replay。
5. 把 staged page-level OCR 接到正式 orchestrator；此時先保留 baseline acceptance profile，直到 P3-C2 校準通過。

驗收：batch mapping 100%、attempt 狀態完整、resume forward=0、無 process-local dict 作唯一 cache、共同 gate全過。

Commit：`fix(P3-C1): persist and activate staged OCR orchestration`

### P3-C2：OCR corpus、校準與 GPU paired evidence

行動：

1. 從共用品質 corpus 完成 ≥300 text、≥300 no-text、≥3 titles；以 title 分 train/dev/test。
2. 只用 train/dev 訓練/選 confidence calibration、no-text classifier 與 `dialogue`、`short_cjk`、`latin_sfx` thresholds；test 只跑一次最終報告，不回頭調參。
3. calibration artifact 綁 model/preprocess/corpus hash；hash 不符必須拒載。未校準值只能叫 `heuristic`，不可叫 probability。
4. 在同一目標 GPU、model、config、test split 比較 baseline 與 staged OCR。
5. accepted candidate 條件：各主要類別 normalized CER ≤ baseline、no-text FP ≤ baseline、1–2 字與 Latin/SFX retention ≥ baseline、accepted-output CER ≤ baseline，且 batching images/s > baseline。

驗收：corpus 數量與 title split 合格、校準優於未校準 Brier score、品質不退、GPU batching 有正收益，evidence 綁 exact fingerprints。

Commit：`test(P3-C2): record held-out OCR calibration and GPU evidence`

### P3-C3：切換 OCR profile 並移除 serial legacy

行動：只在 P3-C2 candidate accepted 後切換新 acceptance profile；刪除舊 serial orchestrator 與誤導性的舊 confidence 命名。不得更動 test split、threshold 或 P3-C2 evidence。

驗收：production 只走 staged OCR；baseline/candidate accepted set 在相同輸入可重播；resume forward=0；共同 gate 全過。

Commit：`feat(P3-C3): switch calibrated OCR and remove serial legacy`

### P3-C4：關閉 G3

行動：在 P3-C3 tip 重跑 OCR contract、artifact fingerprint、target GPU paired benchmark與共同 gate；更新 `benchmarks/gates/g3_ocr_quality_throughput.json`，blockers 空後才設 `passed`。

驗收：G3 validator 可拒絕 corpus 不足、title leakage、stale calibration、品質退步、無 throughput 正收益或 legacy reachable；正常資料通過。

Commit：`test(P3-C4): pass OCR quality throughput gate`

### P4-C1：把已完成翻譯元件接成唯一 production contract

行動：

1. production 由 PageDocument active OCR revisions 建 ordered TranslationUnit；request ID 依最終順序產生但映回持久 region ID。
2. provider 只接受 strict structured output，仍由 P0 exact-ID validator 檢查完整集合；missing/duplicate/extra/unknown 全 batch 拒絕。
3. approved entity ledger 是 hard constraint；candidate 不得自動升級；translation memory 必須符合 source/context/order/entity revision。
4. validator 不破壞 raw response；只有有 issue 的 units 可 targeted repair，只有 `LayoutOverflow` units 可 compact repair，且次數有上限。
5. visual context 預設關閉；只有明確 trigger、使用者 opt-in、符合 data policy/ZDR 才能送低解析 overlay。
6. 所有 production translation/repair/visual 結果只寫 PageDocument translation record；resume 已存 raw response 時 provider request=0。

驗收：exact mapping、entity、repair、privacy、resume tests 全過；未 opt-in 圖片上傳=0；legacy parser 不再是 production caller。

Commit：`fix(P4-C1): wire audited translation contract into production`

### P4-C2：翻譯人評與 paired evidence

行動：

1. 從共用品質 corpus 完成 ≥200 units、≥3 titles，由合格日文→台灣繁中 reviewer 評 baseline/candidate；同一 unit 使用相同 provider/model 條件。
2. 記錄 mapping、漏譯、幻覺、MQM critical/major/minor、approved-name consistency、台灣用語、overflow、repair success/cost。
3. mapping error 必須為 0；approved name consistency 必須 100%；critical 與 major 數都不得高於 baseline。
4. compact repair 只能出現在 overflow subset，且修後仍過語義/layout gate；否則保留原文。
5. visual escalation 另報 trigger 比例、品質差、成本與 privacy profile，不與 text-only 結果混算。

驗收：人工數量、資格與 adjudication 合格；所有 accepted candidate 條件通過；evidence 綁 exact provider/model/prompt/corpus fingerprints。

Commit：`test(P4-C2): record paired translation human evidence`

### P4-C3：切換翻譯流程並移除 legacy

行動：只在 P4-C2 candidate accepted 後，刪除 loose JSON/plain/numbered parser、舊 context/window production path 及重複 sanitizer；PageDocument translation record 成為唯一 production truth。不得修改 P4-C2 人評內容或門檻。

驗收：正式 entrypoint 只走 strict structured contract；legacy symbols 不可達；resume provider request=0；共同 gate 全過。

Commit：`feat(P4-C3): switch verified translation flow and remove legacy`

### P4-C4：關閉 G4

行動：在 P4-C3 tip 重跑 translation contract、privacy defaults、paired benchmark與共同 gate；更新 `benchmarks/gates/g4_translation_entity_privacy.json`，blockers 空後才設 `passed`。

驗收：G4 validator 可拒絕 corpus/reviewer 不足、stale provider/prompt、mapping/name failure、品質退步、圖片未 opt-in或 legacy reachable；正常資料通過。

Commit：`test(P4-C4): pass translation entity privacy gate`

### G04-FINAL：刷新 P0～P4 整合 tip 的 gate

後續 phase 可能改到較早 gate 的 code/config dependency，因此 P4 完成後不可直接沿用舊 `passed`。

行動：

1. 在 final completion tip 重算 G0～G4 各自的 code/config/model/font/prompt/corpus fingerprint。
2. 所有 deterministic validators 與共同 gate 一律重跑。
3. 對 GPU parity、crash/restart、視覺盲評、OCR paired、translation paired/provider evidence，比對其 dependency fingerprint；相同才可沿用原 evidence，任何一項不同就重跑該項。
4. 人工判定可在 source item/candidate image/text hash 全相同時沿用；內容 hash 改變就只重審受影響 item，不要求無關 reviewer 重做。
5. 更新 G0～G4 gate 檔到同一 final tree hash。不得只改 source SHA、不得保留 stale `passed`。

驗收：五個 gate 都 `passed`、blockers 空、fingerprint 指向同一 final tree，且所有 reused evidence 都有相等 dependency hash。

Commit：`test(G0-G4): refresh integrated remediation gates`

## 7. 合併 P0～P4 completion 到 repair

只有 G0～G4 全為 `passed` 才執行：

```bash
cd "$MT_REPO"
test "$(git branch --show-current)" = repair_p0_p4_completion
test -z "$(git status --porcelain)"
git merge-base --is-ancestor 5a7584b0073fafb34e31d4ce13d2f5c9e7bb04f4 HEAD
test -z "$(git rev-list --merges 5a7584b0073fafb34e31d4ce13d2f5c9e7bb04f4..HEAD)"

P0_P4_TIP="$(git rev-parse HEAD)"
git switch repair
test "$(git rev-parse HEAD)" = 5a7584b0073fafb34e31d4ce13d2f5c9e7bb04f4
git merge --no-ff --no-commit "$P0_P4_TIP"
test "$(git rev-parse MERGE_HEAD)" = "$P0_P4_TIP"
```

在尚未 commit 的 merge tree 上重跑共同 gate、G0～G4 validators、五頁 end-to-end smoke。任一失敗或 conflict：

```bash
git merge --abort
git switch repair_p0_p4_completion
```

在 source branch 修正、commit、重驗，再重做 merge；不得在 merge 中直接做未追蹤修補。

全部通過後：

```bash
REPAIR_BEFORE="$(git rev-parse HEAD)"
P0_P4_TIP="$(git rev-parse MERGE_HEAD)"
git commit -m 'merge: complete Phase 0-4 remediation'
test "$(git show -s --format=%P HEAD)" = "$REPAIR_BEFORE $P0_P4_TIP"
test -z "$(git status --porcelain)"
```

## 8. 將正確 P5 branch 快轉到最新 repair

```bash
cd "$MT_P5_WORKTREE"
test "$(git branch --show-current)" = repair_p5
test -z "$(git status --porcelain)"
test "$(git rev-parse HEAD)" = 5a7584b0073fafb34e31d4ce13d2f5c9e7bb04f4
git merge --ff-only repair
test "$(git rev-parse HEAD)" = "$(git rev-parse repair)"
```

若 temp worktree 已不存在，用 `git worktree list` 確認後，以 `git worktree add <新路徑> repair_p5` 重建；不得新建另一個 P5 branch。

舊 `reapir_p5` 只允許 `git show`/`git diff` 讀取；禁止 merge、rebase、整支 cherry-pick。P5 的 code、test、evidence 都在 `repair_p5` 重新實作並逐項驗收。

## 9. 正確完成 P5

### P5-01：quality-complete baseline 與明確優化決策

行動：

1. 在 G2/G3/G4 已通過的版本，用與 P0-C2 相同五頁、硬體、model/config、1 cold＋2 warmup＋5 measured 重跑。
2. 報告 v0.3.2 與 quality-complete 的 stage wall time、p95、VRAM/RSS、API wait、I/O、cache miss 與品質 gate。
3. 對 P5-02～P5-05 各寫一個 `implement` 或 `retain baseline` 決定，引用 run ID 與 hotspot；無量測支持時固定選 `retain baseline`。
4. 不覆蓋舊 baseline，不把 profiler overhead 算成 model inference。

驗收：報告格式可並列 v0.3.2、quality-complete 與後續 candidate；本工作項至少完成前兩者，且都有 source/environment/corpus hash；後四項各有唯一 yes/no 決定。

Commit：`perf(P5-01): record quality-complete baseline decisions`

### P5-02：ModelRegistry 與 VRAM lifecycle

行動：

1. 依 P5-01 決定執行；若是 `retain baseline`，只提交決策結果與防回歸 test，不加入抽象層。
2. 若是 `implement`，由 ModelRegistry 唯一管理 detector/OCR load、reuse、CPU move、unload、revision、device、dtype；移除 process-global singleton。
3. `high_vram` 允許常駐但 GPU stage 序列化；`low_vram` detect 完後釋放/移 CPU 再載 OCR。
4. OOM 只縮當次 batch並重試一次，不永久 forced CPU；model key 必須含 path/revision/hash/device/dtype。

驗收：品質完全相同、重複頁不重載、OOM 不污染下一 job；採用 low-VRAM 預設前必須實測 peak VRAM 下降。

Commit：`perf(P5-02): apply measured model lifecycle decision`

### P5-03：bounded pipeline

行動：

1. 依 P5-01 決定執行；無可重疊 hotspot 就保留串行 baseline。
2. 若實作，只建三個 bounded queues：GPU analyze、network translate、CPU render/encode；queue size 有固定保守上限。
3. 同一 GPU 不並行 model stage；queue 傳 artifact refs/PageDocument，不複製無界 full-resolution ndarray。
4. cancellation、fatal error、Ctrl-C 與 restart 必須原子保存狀態，輸出仍按 page order。
5. 只有同 corpus 實測 throughput 提升且 p95 不退，才設為預設。

驗收：queue 有界、GPU 不並行、kill/restart 無重複 provider request或漏頁、品質 gate 相同。

Commit：`perf(P5-03): apply measured bounded-pipeline decision`

### P5-04：NMS profile gate

行動：

1. 只有 P5-01 指出 NMS/postprocess 是可觀測 hotspot，才以鎖定版本的 `torchvision.ops.nms/batched_nms` 建 candidate；否則明確保留現況。
2. candidate 保存原 index，equal-score 使用 deterministic secondary order；比較 retained set、mask 與 downstream groups。
3. parity 任一差異或 page time 無正收益就不切換；若不再需要 torchvision，移除該直接依賴。

驗收：CPU/GPU tie 穩定、retained boxes/grouping 完全相同；只有實測更快才能移除 Python NMS loop。

Commit：`perf(P5-04): close measured NMS decision`

### P5-05：adaptive high-resolution detector gate

行動：

1. 依 P5-01 決定執行；若是 `retain baseline`，只提交決策結果與現有 baseline 防回歸 test，不建立 candidate。
2. 若是 `implement`，保留 `1024+1536` baseline；candidate 先跑主尺寸，只在 small-char risk、低信心、mask residual 未被 box 覆蓋或明確 profile override 時跑高解析。
3. threshold 只在 detection dev set 調整；test 比 group recall、small-text recall、mask IoU、false positives、額外 pass 比例與時間。
4. 任一品質/安全指標退步即保留固定 baseline；只有品質 non-inferior 且總 detector time 下降才切預設。

驗收：trigger 可追蹤、held-out small-text recall 不退、總時間實測下降，或有完整 no-change 決定。

Commit：`perf(P5-05): close adaptive-detector decision`

### P5-06：atomic image I/O

行動：

1. decode 套用 EXIF orientation並保存原值；保留 ICC、alpha、原始格式 metadata。
2. 明定 JPEG quality/subsampling、PNG lossless、ICC/alpha policy，不依模糊副檔名猜設定。
3. temp file 與目標同目錄，完成 flush/fsync 後 `os.replace`；encode error/kill 不留半檔。
4. 測 rotated JPEG、ICC、RGBA PNG、Unicode/Windows/WSL path。

驗收：round-trip metadata符合 policy、失敗不覆蓋舊檔、不留半成品、共同 gate 全過。

Commit：`feat(P5-06): enforce atomic image I/O policy`

### P5-07：依賴、lock、CI、wheel 與 strict doctor

行動：

1. 稽核 direct imports；移除未使用 direct dependency、補齊實際 direct dependency。不得順便升級無關套件。
2. 固定五個權威檔：`environment.yml`、`conda-lock.yml`、`pyproject.toml`、`poetry.lock`、`poetry.toml`。
3. 重跑 ENV-01 的 lock render→Conda dry-run→Conda install→Poetry install 順序；禁止 `conda-lock install` 作用於既有 `manga`。
4. `doctor --json --strict` 檢查 Conda env、import path、package/model/font hash、Torch/CUDA/driver/SM、OCR revision、RAQM/native libs、SQLite、輸出路徑、provider capability。
5. CI 使用 fresh runner；`conda-lock.yml` 是 committed source，explicit lock 是每次在 repo 外重新 render 的 derived file。CI 只在 ephemeral base bootstrap 同版 `conda-lock`，再由 Conda explicit lock 建立名為 `manga` 的環境：

```bash
export CI_CONDA='/opt/conda/bin/conda'
CI_LOCK_TMP="$(mktemp -d)"
! "$CI_CONDA" env list | grep -Eq '^[[:space:]]*manga[[:space:]]'
"$CI_CONDA" install -n base --yes -c conda-forge conda-lock=4.0.0
"$CI_CONDA" run -n base conda-lock render -k explicit -p linux-64 \
  --filename-template "$CI_LOCK_TMP/conda-{platform}.lock" conda-lock.yml
"$CI_CONDA" create -n manga --yes --file "$CI_LOCK_TMP/conda-linux-64.lock"
```

6. 在 CI 的 `manga` 內確認 Poetry 不建 venv，再做 Poetry install/build；不得建立 venv。
7. `poetry build` 後用 Python 標準庫解開 wheel 到暫存目錄，從解開內容做 import、CLI help、一張自有 fixture smoke；不使用 pip 安裝 wheel。
8. CPU doctor 必跑；GPU/provider doctor 使用同一 source/archive SHA。缺 GPU/credential 就 `blocked`，不得以 CPU/mock 替代。

驗收：五檔一致、fresh `manga` 可重建、RAQM 有效、CI/ruff/tests/build/smoke 全過、strict doctor 對缺能力非零退出。

Commit：`build(P5-07): lock Conda Poetry release environment`

### P5-08：assets、licenses、SBOM 與 release package

行動：

1. `assets/manifest.json` 的每個 model/font/vendor asset 固定來源 revision、SHA-256、size、license；fetch 先下載 temp、驗 hash後 atomic rename。
2. 模型/字型改一 byte 時，fetch verify、doctor、release build 都必須失敗。
3. 補齊 `LICENSE`、`THIRD_PARTY_NOTICES.md`、`licenses/`；任何來源/再散布權不明項目先由法務/owner 核准，未核准不得打包。
4. 由 Conda/Poetry lock 以專案 script 產生 CycloneDX SBOM，不為此引入另一個套件管理器。
5. release 排除 input/output、benchmark 私有圖、cache、pyc、secret、provider raw response。
6. 產出 ZIP 與 release manifest/hash；fresh extraction 配 P5-07 locked `manga` 跑 doctor、unit smoke、一張自有 fixture。

驗收：所有 asset hash/授權明確、SBOM 與 archive 對得上、fresh extraction 通過、無敏感或未授權內容。

Commit：`release(P5-08): verify assets licenses SBOM and archive`

## 10. G5 最終 release gate

P5-08 後以 `repair_p5` 當 source 重跑一次最終 gate：

- 回歸：5 頁、38 groups 全 verified，mapping 100%、swap=0、safe no-erase=100%。
- 品質：≥30 頁、≥3 titles、≥300 text、≥300 no-text、≥200 translation units，title-disjoint split。
- 排版：missing glyph=0、CLREQ hard violation=0、collision=0、alpha containment≥99.5%、ROI 外 diff=0。
- OCR/翻譯：held-out 主要類別不退，approved-name consistency=100%。
- 韌性：partial failure 預設非零、replay/resume 無重做、I/O 原子、asset/lock 可重現。
- 效能：所有聲稱都有相同 corpus、固定硬體與 run ID；未達 1.5× 等方向性目標可如實記錄，但不可虛報。
- 環境：CPU、GPU、provider strict doctor 綁同一 archive SHA。
- 發行：法務/owner 核准完成，archive 不含未授權內容、cache、secret、provider raw response。

建立 `benchmarks/gates/g5_release.json`；只有全部 hard gate 通過才設 `passed`。

Commit：`release(G5): pass final quality and release gate`

## 11. 合併 P5 回 repair

在根 worktree 執行：

```bash
cd "$MT_REPO"
test "$(git branch --show-current)" = repair
test -z "$(git status --porcelain)"
test "$(git -C "$MT_P5_WORKTREE" branch --show-current)" = repair_p5
test -z "$(git -C "$MT_P5_WORKTREE" status --porcelain)"
git merge-base --is-ancestor repair repair_p5
test "$(git rev-list --count repair..repair_p5)" -gt 0
test -z "$(git rev-list --merges repair..repair_p5)"
! git merge-base --is-ancestor reapir_p5 repair_p5

P5_TIP="$(git rev-parse repair_p5)"
git merge --no-ff --no-commit "$P5_TIP"
test "$(git rev-parse MERGE_HEAD)" = "$P5_TIP"
```

在 merge tree 上重跑共同 gate、G0～G5 validators、五頁 end-to-end、30-page final gate、build/fresh smoke。失敗或 conflict 時 `git merge --abort`，回 `repair_p5` 修正並 commit；不得在 merge 中修。

通過後：

```bash
REPAIR_BEFORE="$(git rev-parse HEAD)"
P5_TIP="$(git rev-parse MERGE_HEAD)"
git commit -m 'merge: complete verified Phase 5'
test "$(git show -s --format=%P HEAD)" = "$REPAIR_BEFORE $P5_TIP"
test -z "$(git status --porcelain)"
```

## 12. 合入 main 與交付

先同步遠端；沒有網路/權限就標 `blocked`，不可假設遠端未變：

```bash
cd "$MT_REPO"
test -z "$(git status --porcelain)"
git fetch origin
ORIGIN_MAIN_TIP="$(git rev-parse origin/main)"
REPAIR_TIP="$(git rev-parse repair)"
git merge-base --is-ancestor "$ORIGIN_MAIN_TIP" "$REPAIR_TIP"
git switch main
test -z "$(git status --porcelain)"
git merge --ff-only "$ORIGIN_MAIN_TIP"
test "$(git rev-parse HEAD)" = "$ORIGIN_MAIN_TIP"
MAIN_BEFORE="$(git rev-parse HEAD)"
git merge --no-ff --no-commit "$REPAIR_TIP"
test "$(git rev-parse MERGE_HEAD)" = "$REPAIR_TIP"
```

若祖先檢查失敗，表示遠端 main 在本計劃執行期間出現新歷史；停止並標 `blocked`，另行稽核該差異，不把未知遠端變更臨時塞進本次 release。

在 main 的 merge tree 上再跑共同 gate、G0～G5 validators、build/fresh smoke。失敗或 conflict 時 `git merge --abort`，回 `repair`/`repair_p5` 修正，不在 main 直接修。

通過後：

```bash
MAIN_BEFORE="$(git rev-parse HEAD)"
REPAIR_TIP="$(git rev-parse MERGE_HEAD)"
git commit -m 'merge: release manga-translator v0.3.2 remediation'
test "$(git show -s --format=%P HEAD)" = "$MAIN_BEFORE $REPAIR_TIP"
test -z "$(git status --porcelain)"
```

push 前由使用者確認 remote 與 release 權限；只允許 fast-forward push，禁止 force push。交付清單：

- `main`、`repair`、`repair_p5` 的 final SHA；
- 兩個 remediation merge commit 與 main merge commit；
- G0～G5 gate 檔與 benchmark run IDs；
- Conda/Poetry 五個權威檔 hash；
- release ZIP、SBOM、release manifest/hash；
- 仍保留、但不再待合併的舊 branch/worktree 清單。

舊 branch/worktree 的刪除是後續獨立、需使用者明確批准的清理工作，不是本計劃完成條件。
