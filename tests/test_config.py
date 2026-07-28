from __future__ import annotations

from pathlib import Path

from manga_translator.config import AppConfig, OpenRouterConfig


def test_config_paths_are_relative_to_yaml(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    config_path = project / "config.yaml"
    config_path.write_text(
        """
openrouter:
  api_key: YOUR_OPENROUTER_API_KEY
  model: test/model
paths:
  input_dir: assets/input
  output_dir: assets/output
  glossary: glossary.json
  font: fonts/main.ttf
  font_fallback: fonts/fallback.otf
detection:
  model_path: models/detector.pt
""".strip(),
        encoding="utf-8",
    )

    cfg = AppConfig.from_yaml(config_path)

    assert cfg.paths.input_dir == (project / "assets/input").resolve()
    assert cfg.paths.output_dir == (project / "assets/output").resolve()
    assert cfg.detection.model_path == (project / "models/detector.pt").resolve()


def test_api_key_can_come_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-secret")
    cfg = OpenRouterConfig(api_key="YOUR_OPENROUTER_API_KEY", model="test/model")
    assert cfg.api_key == "env-secret"
