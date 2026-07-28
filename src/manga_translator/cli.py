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
from .image_io import read_image, write_image
from .ocr import OCRInitializationError
from .pipeline import process_single_page, run_pipeline
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
@click.option("--dump-json", is_flag=True, help="輸出每頁 debug manifest json")
@click.option("--save-intermediate", is_flag=True, help="輸出中間圖（original/inpainted/blanked）")
@click.option("--prep-manual", is_flag=True, help="輸出手動校正素材（會啟用 debug/json/intermediate）")
@click.option("--allow-partial", is_flag=True, help="部分頁面失敗時仍以成功退出")
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
        )
    except OCRInitializationError as error:
        raise click.ClickException(str(error)) from error
    if not result.succeeded and not (allow_partial and result.partial):
        raise click.exceptions.Exit(1)


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
