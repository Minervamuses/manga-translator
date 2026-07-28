from __future__ import annotations

from click.testing import CliRunner

from manga_translator import cli as cli_module
from manga_translator.ocr import OCRInitializationError


def test_run_command_reports_ocr_initialization_failure_as_single_cli_error(
    tmp_path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
openrouter:
  api_key: test
  model: test/model
paths:
  input_dir: ./input
  output_dir: ./output
  glossary: ./glossary.json
  font: ./font.ttf
  font_fallback: ./fallback.ttf
detection:
  model_path: ./model.pt
  device: cpu
""".strip(),
        encoding="utf-8",
    )

    calls = 0

    def fail_once(*args, **kwargs) -> None:
        nonlocal calls
        calls += 1
        raise OCRInitializationError("OCR backend startup failed")

    monkeypatch.setattr(cli_module, "run_pipeline", fail_once)

    result = CliRunner().invoke(
        cli_module.cli,
        ["run", "--config", str(config_path)],
    )

    assert result.exit_code == 1
    assert calls == 1
    assert result.output.count("OCR backend startup failed") == 1
    assert "完成！" not in result.output
