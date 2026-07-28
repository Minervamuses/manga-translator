"""CLI 入口

用法：
    manga-translate run --config config.yaml
    manga-translate run --config config.yaml --debug
    manga-translate test --config config.yaml --image page001.png
"""

from __future__ import annotations

import json
from pathlib import Path

import click
from rich.console import Console

from .config import AppConfig
from .domain.issues import StageName
from .domain.serialization import canonical_document_bytes
from .image_io import read_image, write_image
from .ocr import OCRInitializationError
from .pipeline import process_single_page, run_pipeline
from .storage import ArtifactStore, JobStore
from .storage.artifact_store import assess_storage_path
from .translator import load_glossary

console = Console()


@click.group()
def cli():
    """manga-translator: 日漫翻譯工具"""


def _apply_runtime_overrides(
    cfg: AppConfig,
    *,
    no_grouping: bool,
    page_context_mode: str | None,
    render_scope: str | None,
) -> AppConfig:
    if no_grouping:
        cfg.postprocess = cfg.postprocess.model_copy(
            update={
                "enable_grouping": False,
                "enable_ocr_dedup": False,
                "enable_group_translate": False,
            }
        )
    if page_context_mode:
        cfg.openrouter = cfg.openrouter.model_copy(update={"page_context_mode": page_context_mode})
    if render_scope:
        cfg.typesetting = cfg.typesetting.model_copy(update={"render_scope": render_scope})
    return cfg


@cli.command()
@click.option("--config", "-c", default="config.yaml", help="設定檔路徑")
@click.option("--debug", "-d", is_flag=True, help="輸出 debug 標註圖")
@click.option("--dump-json", is_flag=True, help="輸出每頁 canonical PageDocument JSON")
@click.option("--save-intermediate", is_flag=True, help="輸出中間圖（original/inpainted/blanked）")
@click.option("--prep-manual", is_flag=True, help="輸出手動校正素材（會啟用 debug/json/intermediate）")
@click.option("--allow-partial", is_flag=True, help="部分頁面失敗時仍以成功退出")
@click.option("--resume", is_flag=True, help="沿用 fingerprint 相同且 artifact 完整的 stage")
@click.option(
    "--force-stage",
    type=click.Choice([stage.value for stage in StageName]),
    default=None,
    help="強制重跑指定 stage 與所有 downstream stage",
)
@click.option("--job", "job_id", default="default", show_default=True, help="持久 job ID")
@click.option("--state-dir", type=click.Path(path_type=Path), default=None)
@click.option("--no-grouping", is_flag=True, help="停用 grouping（回退舊模式）")
@click.option("--page-context-mode", type=click.Choice(["window", "page"]), default=None)
@click.option(
    "--render-scope",
    type=click.Choice(["region", "group_bbox", "group_mask"]),
    default=None,
)
def run(
    config: str,
    debug: bool,
    dump_json: bool,
    save_intermediate: bool,
    prep_manual: bool,
    allow_partial: bool,
    resume: bool,
    force_stage: str | None,
    job_id: str,
    state_dir: Path | None,
    no_grouping: bool,
    page_context_mode: str | None,
    render_scope: str | None,
):
    """執行完整翻譯流水線"""
    console.print("[bold cyan]manga-translator[/] 啟動中...\n")

    cfg = AppConfig.from_yaml(config)
    cfg = _apply_runtime_overrides(
        cfg,
        no_grouping=no_grouping,
        page_context_mode=page_context_mode,
        render_scope=render_scope,
    )
    if prep_manual:
        debug = True
        dump_json = True
        save_intermediate = True
    try:
        result = run_pipeline(
            cfg,
            debug=debug,
            dump_json=dump_json,
            save_intermediate=save_intermediate,
            prep_manual=prep_manual,
            resume=resume,
            force_stage=StageName(force_stage) if force_stage else None,
            job_id=job_id,
            state_dir=state_dir,
        )
    except OCRInitializationError as error:
        raise click.ClickException(str(error)) from error
    if not result.succeeded and not (allow_partial and result.partial):
        raise click.exceptions.Exit(1)


def _durable_root(config: str, state_dir: Path | None) -> Path:
    if state_dir is not None:
        return state_dir.expanduser().resolve()
    return (AppConfig.from_yaml(config).paths.output_dir / ".manga-translator").resolve()


@cli.command("inspect")
@click.option("--config", "-c", default="config.yaml", help="設定檔路徑")
@click.option("--state-dir", type=click.Path(path_type=Path), default=None)
@click.option("--job", "job_id", required=True, help="持久 job ID")
@click.option("--page", "page_id", default=None, help="可選的 page SHA-256")
def inspect_job(config: str, state_dir: Path | None, job_id: str, page_id: str | None) -> None:
    """顯示 canonical PageDocument、stage、issue 與 artifact 狀態。"""

    root = _durable_root(config, state_dir)
    with JobStore(root / "jobs.sqlite3", ArtifactStore(root / "artifacts")) as store:
        pages = []
        for row in store.list_pages(job_id=job_id, page_id=page_id):
            document = store.load_page_document(job_id=job_id, page_id=str(row[0]))
            if document is None:
                continue
            pages.append(
                {
                    "page_id": document.source.page_id,
                    "document_artifact": {
                        "sha256": str(row[1]),
                        "path": str(store.artifacts.path_for(str(row[1]))),
                    },
                    "active_regions": [
                        {
                            "region_id": str(identity.region_id),
                            "revision_id": identity.active_revision_id,
                        }
                        for identity in document.region_identities
                    ],
                    "stages": [
                        {
                            "stage": stage.stage.value,
                            "fingerprint": stage.fingerprint,
                            "status": stage.status.value,
                            "cache_hit": stage.cache_hit,
                            "artifacts": [
                                {
                                    "sha256": sha256,
                                    "path": str(store.artifacts.path_for(sha256)),
                                }
                                for sha256 in stage.output_hashes
                            ],
                        }
                        for stage in document.stages
                    ],
                    "stage_attempts": [
                        {
                            "stage": str(attempt[0]),
                            "fingerprint": str(attempt[1]),
                            "status": str(attempt[2]),
                            "input_hashes": json.loads(str(attempt[3])),
                            "output_hashes": json.loads(str(attempt[4])),
                            "started_at": attempt[5],
                            "finished_at": attempt[6],
                            "cache_hit_count": attempt[7],
                            "last_cache_hit_at": attempt[8],
                        }
                        for attempt in store.list_stage_runs(
                            job_id=job_id, page_id=document.source.page_id
                        )
                    ],
                    "issues": [issue.model_dump(mode="json") for issue in document.issues],
                }
            )
    if not pages:
        raise click.ClickException("找不到符合的 job/page")
    click.echo(json.dumps({"job_id": job_id, "pages": pages}, ensure_ascii=False, indent=2))


@cli.group()
def cache() -> None:
    """管理 durable artifact cache。"""


@cache.command("gc")
@click.option("--config", "-c", default="config.yaml", help="設定檔路徑")
@click.option("--state-dir", type=click.Path(path_type=Path), default=None)
def cache_gc(config: str, state_dir: Path | None) -> None:
    """清理沒有任何持久引用的 artifacts。"""

    root = _durable_root(config, state_dir)
    with JobStore(root / "jobs.sqlite3", ArtifactStore(root / "artifacts")) as store:
        result = store.gc()
    click.echo(
        f"removed database_records={result.database_records} artifact_files={result.files}"
    )


@cli.command()
@click.option("--config", "-c", default="config.yaml", help="設定檔路徑")
@click.option("--state-dir", type=click.Path(path_type=Path), default=None)
@click.option("--job", "job_id", required=True, help="持久 job ID")
@click.option("--page", "page_id", required=True, help="page SHA-256")
@click.option("--output", type=click.Path(path_type=Path), required=True)
@click.option("--output-image", type=click.Path(path_type=Path), default=None)
def replay(
    config: str,
    state_dir: Path | None,
    job_id: str,
    page_id: str,
    output: Path,
    output_image: Path | None,
) -> None:
    """不連網、不載模型，重播已保存的 canonical manifest 與 encode artifact。"""

    root = _durable_root(config, state_dir)
    with JobStore(root / "jobs.sqlite3", ArtifactStore(root / "artifacts")) as store:
        document = store.load_page_document(job_id=job_id, page_id=page_id)
        if document is None:
            raise click.ClickException("找不到符合的 job/page")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(canonical_document_bytes(document))
        if output_image is not None:
            encode = next(
                (stage for stage in document.stages if stage.stage is StageName.ENCODE), None
            )
            if encode is None or not encode.output_hashes:
                raise click.ClickException("PageDocument 沒有 encode artifact")
            output_image.parent.mkdir(parents=True, exist_ok=True)
            output_image.write_bytes(store.artifacts.read_bytes(encode.output_hashes[0]))
    click.echo(str(output))


@cli.command()
@click.option("--config", "-c", default="config.yaml", help="設定檔路徑")
@click.option("--image", "-i", required=True, help="單張測試圖片路徑")
@click.option("--debug", "-d", is_flag=True, default=True, help="輸出 debug 標註圖")
@click.option("--dump-json", is_flag=True, help="輸出 debug manifest json")
@click.option("--save-intermediate", is_flag=True, help="輸出中間圖")
@click.option("--prep-manual", is_flag=True, help="輸出手動校正素材")
@click.option("--no-grouping", is_flag=True, help="停用 grouping")
@click.option("--page-context-mode", type=click.Choice(["window", "page"]), default=None)
@click.option(
    "--render-scope",
    type=click.Choice(["region", "group_bbox", "group_mask"]),
    default=None,
)
def test(
    config: str,
    image: str,
    debug: bool,
    dump_json: bool,
    save_intermediate: bool,
    prep_manual: bool,
    no_grouping: bool,
    page_context_mode: str | None,
    render_scope: str | None,
):
    """測試單張圖片（用於調參）"""
    console.print("[bold cyan]manga-translator[/] 測試模式\n")

    cfg = AppConfig.from_yaml(config)
    cfg = _apply_runtime_overrides(
        cfg,
        no_grouping=no_grouping,
        page_context_mode=page_context_mode,
        render_scope=render_scope,
    )
    if prep_manual:
        debug = True
        dump_json = True
        save_intermediate = True
    cfg.paths.output_dir.mkdir(parents=True, exist_ok=True)
    glossary = load_glossary(cfg.paths.glossary)

    image_path = Path(image)
    if not image_path.exists():
        console.print(f"[red]找不到圖片：{image_path}[/]")
        return

    try:
        page_result = process_single_page(
            image_path,
            cfg,
            glossary,
            debug=debug,
            dump_json=dump_json,
            save_intermediate=save_intermediate,
            prep_manual=prep_manual,
        )
    except OCRInitializationError as error:
        raise click.ClickException(str(error)) from error

    if not page_result.succeeded or page_result.image is None:
        message = page_result.issues[0].message if page_result.issues else "頁面處理失敗"
        raise click.ClickException(message)

    output_path = cfg.paths.output_dir / f"test_{image_path.name}"
    if not write_image(output_path, page_result.image):
        raise click.ClickException(f"無法寫入圖片：{output_path}")
    console.print(f"\n[bold green]測試完成，結果儲存於：{output_path}[/]")


@cli.command()
@click.option("--config", "-c", default="config.yaml", help="設定檔路徑")
@click.option("--image", "-i", required=True, help="單張圖片路徑")
def detect_only(config: str, image: str):
    """只做偵測（不翻譯），輸出標註圖，用來調整偵測參數"""
    from .detector import detect_text_regions, draw_debug_regions

    cfg = AppConfig.from_yaml(config)
    cfg.paths.output_dir.mkdir(parents=True, exist_ok=True)

    image_path = Path(image)
    img = read_image(image_path)
    if img is None:
        console.print(f"[red]無法讀取：{image_path}[/]")
        return

    detection = detect_text_regions(img, cfg.detection, cfg.postprocess)
    regions = detection.regions
    fallback_count = sum(region.source == "mask_fallback" for region in detection.regions_raw)
    console.print(
        f"偵測到 [cyan]{len(regions)}[/] 個文字區域，"
        f"其中 mask fallback [yellow]{fallback_count}[/] 個"
    )

    debug_img = draw_debug_regions(img, regions, detection.groups)
    output_path = cfg.paths.output_dir / f"detect_{image_path.name}"
    if not write_image(output_path, debug_img):
        raise click.ClickException(f"無法寫入圖片：{output_path}")
    console.print(f"[green]標註圖儲存於：{output_path}[/]")


@cli.command()
@click.option("--config", "-c", default="config.yaml", help="設定檔路徑")
@click.option(
    "--strict-api-key",
    is_flag=True,
    help="將 API key 缺失視為失敗（預設僅警告）",
)
def doctor(config: str, strict_api_key: bool):
    """檢查執行環境與必要資源是否就緒"""
    ok_count = 0
    failed = 0
    warned = 0

    def ok(msg: str) -> None:
        nonlocal ok_count
        ok_count += 1
        console.print(f"[green]OK[/] {msg}")

    def warn(msg: str) -> None:
        nonlocal warned
        warned += 1
        console.print(f"[yellow]WARN[/] {msg}")

    def fail(msg: str) -> None:
        nonlocal failed
        failed += 1
        console.print(f"[red]FAIL[/] {msg}")

    try:
        cfg = AppConfig.from_yaml(config)
        ok(f"設定檔可讀取：{Path(config).resolve()}")
    except Exception as e:  # noqa: BLE001 - doctor reports every configuration failure
        fail(f"設定檔無法載入：{e}")
        raise click.exceptions.Exit(1)

    try:
        import cv2  # noqa: F401

        ok("OpenCV (cv2) 可用")
    except Exception as e:  # noqa: BLE001 - doctor converts import failures to diagnostics
        fail(f"OpenCV (cv2) 不可用：{e}")

    try:
        from .manga_ocr_runtime import check_runtime_dependencies

        versions = check_runtime_dependencies()
        ok(
            "OCR 執行元件可用："
            f"Transformers {versions['transformers']} / Torch {versions['torch']}"
        )
    except Exception as e:  # noqa: BLE001 - runtime health checks may expose backend failures
        fail(f"OCR 執行元件不可用：{e}")

    try:
        import shapely  # noqa: F401

        ok("Shapely 幾何後處理套件可用")
    except Exception as e:  # noqa: BLE001 - doctor converts import failures to diagnostics
        fail(f"Shapely 不可用（文字偵測器需要）：{e}")

    try:
        import pyclipper  # noqa: F401

        ok("pyclipper polygon offset 加速可用")
    except Exception:  # noqa: BLE001 - optional accelerator failure is non-fatal
        warn("pyclipper 不可用；會自動改用 Shapely，功能正常但幾何後處理稍慢")

    cfg.paths.input_dir.mkdir(parents=True, exist_ok=True)
    ok(f"輸入目錄存在：{cfg.paths.input_dir}")

    cfg.paths.output_dir.mkdir(parents=True, exist_ok=True)
    ok(f"輸出目錄存在：{cfg.paths.output_dir}")
    state_assessment = assess_storage_path(cfg.paths.output_dir / ".manga-translator")
    if state_assessment.kind == "network":
        fail(f"durable state 不可位於網路 share：{state_assessment.reason}")
    elif state_assessment.kind == "unknown":
        warn(f"無法確認 durable state 是否為本機磁碟：{state_assessment.reason}")
    else:
        ok("durable state 位於本機磁碟")

    model_path = cfg.detection.model_path.resolve()
    if model_path.exists():
        ok(f"偵測模型存在：{model_path}")
    else:
        fail(f"偵測模型不存在：{model_path}（請先執行 scripts/download_models.sh）")

    glossary_path = cfg.paths.glossary
    if glossary_path.exists():
        try:
            with open(glossary_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            entries = raw.get("entries", raw)
            if isinstance(entries, dict):
                ok(f"字典可讀取：{glossary_path}（{len(entries)} 筆）")
            else:
                fail(f"字典格式錯誤：{glossary_path}（預期為 JSON object）")
        except Exception as e:  # noqa: BLE001 - malformed glossary is a doctor diagnostic
            fail(f"字典讀取失敗：{glossary_path}（{e}）")
    else:
        warn(f"字典不存在：{glossary_path}（可選）")

    font_path = cfg.paths.font
    if font_path.exists():
        try:
            from PIL import ImageFont

            ImageFont.truetype(str(font_path), 24)
            ok(f"字體可用：{font_path}")
        except Exception as e:  # noqa: BLE001 - Pillow can surface backend-specific errors
            fail(f"字體無法載入：{font_path}（{e}）")
    else:
        fail(
            "字體不存在："
            f"{font_path}（請放入 config.yaml 指定的主字體）"
        )

    fallback_font = cfg.paths.font_fallback
    if fallback_font.exists():
        try:
            from PIL import ImageFont

            ImageFont.truetype(str(fallback_font), 24)
            ok(f"fallback 字體可用：{fallback_font}")
        except Exception as e:  # noqa: BLE001 - Pillow can surface backend-specific errors
            fail(f"fallback 字體無法載入：{fallback_font}（{e}）")
    else:
        warn(f"fallback 字體不存在：{fallback_font}（將無法缺字替補）")

    api_key = (cfg.openrouter.api_key or "").strip()
    api_placeholder = api_key in {"", "YOUR_OPENROUTER_API_KEY"}
    if api_placeholder:
        msg = "OpenRouter API key 尚未設定（最後再填即可）"
        if strict_api_key:
            fail(msg)
        else:
            warn(msg)
    else:
        ok("OpenRouter API key 已設定")

    console.print(
        f"\n[bold]檢查完成：[/] [green]{ok_count} OK[/] / "
        f"[yellow]{warned} WARN[/] / [red]{failed} FAIL[/]"
    )
    if failed > 0:
        raise click.exceptions.Exit(1)


def main():
    cli()


if __name__ == "__main__":
    main()
