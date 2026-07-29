from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from click.testing import CliRunner

from manga_translator.cli import cli
from manga_translator.domain.issues import StageName
from manga_translator.stages.base import FingerprintDependencies, StageOutputs, StageSpec
from manga_translator.stages.fingerprint import stage_fingerprint
from manga_translator.stages.runner import downstream_of
from manga_translator.storage.artifact_store import ArtifactStore
from manga_translator.storage.job_store import JobStore
from manga_translator.translation.entities import EntityLedger
from manga_translator.translation.memory import MemoryKey, TranslationMemory, build_memory_key


def _store(tmp_path: Path) -> JobStore:
    store = JobStore(tmp_path / "jobs.sqlite3", ArtifactStore(tmp_path / "artifacts"))
    store.create_job("job")
    return store


def test_ocr_aliases_reuse_one_candidate_and_model_cannot_auto_approve(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        ledger = EntityLedger(store, job_id="job", chapter_id="chapter-1")
        first = ledger.propose("ミカ", aliases=("ミ力",), provenance={"source": "model"})
        second = ledger.propose("ミ力", aliases=("ミカ",), provenance={"source": "ocr"})

        assert first.entity_id == second.entity_id
        assert second.status == "candidate"
        assert ledger.approved_constraints() == {}
        assert ledger.prompt_entries() == (
            {
                "source": "ミカ",
                "target": None,
                "status": "candidate",
                "constraint": "hint",
            },
        )


def test_only_human_approval_turns_an_entity_into_a_hard_constraint(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        ledger = EntityLedger(store, job_id="job", chapter_id="chapter-1")
        candidate = ledger.propose("綾波", aliases=("綾波レイ",))
        approved = ledger.approve(candidate.entity_id, "綾波零", reviewer_id="reviewer-7")

        assert approved.status == "approved"
        assert ledger.approved_constraints() == {"綾波": "綾波零", "綾波レイ": "綾波零"}
        assert ledger.list_candidates() == ()
        assert ledger.prompt_entries()[0]["constraint"] == "hard"


def test_glossary_import_is_trusted_approved_and_round_trips(tmp_path: Path) -> None:
    glossary = tmp_path / "glossary.json"
    glossary.write_text(
        json.dumps({"entries": {"先生": "老師", "魔王": "魔王"}}, ensure_ascii=False),
        encoding="utf-8",
    )
    with _store(tmp_path) as store:
        ledger = EntityLedger(store, job_id="job", chapter_id="chapter-1")
        imported = ledger.import_glossary(glossary)
        output = tmp_path / "exported.json"
        exported = ledger.export_glossary(output)

        assert {entity.status for entity in imported} == {"approved"}
        assert exported == {"entries": {"先生": "老師", "魔王": "魔王"}}
        assert json.loads(output.read_text(encoding="utf-8")) == exported


def test_merge_preserves_aliases_without_duplicate_characters(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        ledger = EntityLedger(store, job_id="job", chapter_id="chapter-1")
        target = ledger.propose("アル", aliases=("アル君",))
        source = ledger.propose("ア儿", aliases=("アルくん",))
        merged = ledger.merge(source.entity_id, target.entity_id, reviewer_id="reviewer")

        assert set(merged.aliases) == {"アル", "アル君", "ア儿", "アルくん"}
        assert ledger.get(source.entity_id).status == "merged"  # type: ignore[union-attr]


def test_alias_bridge_requires_a_reviewed_merge(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        ledger = EntityLedger(store, job_id="job", chapter_id="chapter-1")
        first = ledger.propose("アリス")
        second = ledger.propose("ありす")

        with pytest.raises(ValueError, match="reviewed merge"):
            ledger.propose("Alice", aliases=(first.canonical_source, second.canonical_source))

        assert len(ledger.list()) == 2


def test_merge_preserves_approval_and_rejects_conflicting_approvals(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        ledger = EntityLedger(store, job_id="job", chapter_id="chapter-1")
        approved_source = ledger.propose("勇者")
        candidate_target = ledger.propose("勇者さま")
        ledger.approve(approved_source.entity_id, "勇者", reviewer_id=" reviewer ")

        merged = ledger.merge(
            approved_source.entity_id,
            candidate_target.entity_id,
            reviewer_id=" reviewer ",
        )
        assert merged.status == "approved"
        assert merged.approved_zh_tw == "勇者"
        assert merged.provenance["merge_approval_reviewer"] == "reviewer"

        other = ledger.propose("ヒーロー")
        ledger.approve(other.entity_id, "英雄", reviewer_id="reviewer")
        with pytest.raises(ValueError, match="different translations"):
            ledger.merge(other.entity_id, merged.entity_id, reviewer_id="reviewer")


def test_entity_inputs_and_provenance_fail_closed(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        ledger = EntityLedger(store, job_id="job", chapter_id="chapter-1")
        with pytest.raises(ValueError, match="kind"):
            ledger.propose("人物", kind=" ")
        with pytest.raises(ValueError, match="Out of range float values"):
            ledger.propose("人物", provenance={"confidence": math.nan})
        with pytest.raises(TypeError, match="strings to strings"):
            ledger.import_glossary({"勇者": 1})  # type: ignore[dict-item]
        assert ledger.list() == ()


def test_translation_memory_requires_exact_context_order_and_entity_revision(
    tmp_path: Path,
) -> None:
    with _store(tmp_path) as store:
        memory = TranslationMemory(store, job_id="job", chapter_id="chapter-1")
        revision = "a" * 64
        key = build_memory_key(
            "カフェ\u0301", context={"speaker": "mika"}, order=[1, 2], entity_revision_hash=revision
        )
        suggestion = memory.put(key, "咖啡店", provenance={"source": "model"})
        assert not suggestion.reusable

        approved = memory.put(
            key,
            "咖啡店",
            status="approved",
            reviewer_id="reviewer-3",
            provenance={"source": "human"},
        )
        assert approved.reusable
        assert memory.lookup(
            "カフェ\u0301",
            context={"speaker": "mika"},
            order=[1, 2],
            entity_revision_hash=revision,
        ).reusable  # type: ignore[union-attr]
        assert (
            memory.lookup(
                "カフェ\u0301",
                context={"speaker": "other"},
                order=[1, 2],
                entity_revision_hash=revision,
            )
            is None
        )

        protected = memory.put(key, "模型錯譯", status="suggestion")
        assert protected.reusable
        assert protected.target_zh_tw == "咖啡店"


def test_translation_memory_rejects_forged_keys_and_non_json_provenance(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        memory = TranslationMemory(store, job_id="job", chapter_id="chapter-1")
        key = build_memory_key("原文", context={}, order=[], entity_revision_hash="a" * 64)
        forged = MemoryKey("b" * 64, key.source_nfc, key.context_hash, key.order_hash, "a" * 64)

        with pytest.raises(ValueError, match="does not match"):
            memory.put(forged, "譯文")
        with pytest.raises(ValueError, match="lowercase SHA-256"):
            build_memory_key("原文", context={}, order=[], entity_revision_hash="Z" * 64)
        with pytest.raises(ValueError, match="Out of range float values"):
            memory.put(key, "譯文", provenance={"score": math.inf})
        assert memory.lookup_key(key) is None


def test_glossary_revision_only_changes_translate_fingerprint_and_downstream() -> None:
    run = lambda _context, _inputs: StageOutputs(())
    ocr = StageSpec(StageName.OCR, (), run, "code")
    translate_v1 = StageSpec(
        StageName.TRANSLATE,
        (),
        run,
        "code",
        fingerprint_dependencies=FingerprintDependencies(glossary_revision="v1"),
    )
    translate_v2 = StageSpec(
        StageName.TRANSLATE,
        (),
        run,
        "code",
        fingerprint_dependencies=FingerprintDependencies(glossary_revision="v2"),
    )
    assert stage_fingerprint(ocr, upstream_output_hashes=(), config={}) == stage_fingerprint(
        ocr, upstream_output_hashes=(), config={"unrelated": "change"}
    )
    assert stage_fingerprint(
        translate_v1, upstream_output_hashes=(), config={}
    ) != stage_fingerprint(translate_v2, upstream_output_hashes=(), config={})
    assert downstream_of(StageName.TRANSLATE) == {
        StageName.TRANSLATE,
        StageName.LAYOUT,
        StageName.INPAINT_RENDER,
        StageName.ENCODE,
    }


def test_entity_cli_import_list_and_export(tmp_path: Path) -> None:
    glossary = tmp_path / "glossary.json"
    glossary.write_text('{"entries":{"勇者":"勇者"}}', encoding="utf-8")
    state = tmp_path / "state"
    runner = CliRunner()
    common = ["--state-dir", str(state), "--job", "job", "--chapter", "c1"]

    imported = runner.invoke(cli, ["entities", "import-glossary", *common, str(glossary)])
    listed = runner.invoke(cli, ["entities", "list", *common, "--status", "approved"])
    output = tmp_path / "out.json"
    exported = runner.invoke(cli, ["entities", "export-glossary", *common, str(output)])

    assert imported.exit_code == listed.exit_code == exported.exit_code == 0
    assert json.loads(listed.output)[0]["approved_zh_tw"] == "勇者"
    assert json.loads(output.read_text(encoding="utf-8")) == {"entries": {"勇者": "勇者"}}
