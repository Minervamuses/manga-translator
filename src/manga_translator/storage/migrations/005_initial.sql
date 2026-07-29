CREATE TABLE chapter_entities (
    job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
    chapter_id TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    canonical_source TEXT NOT NULL CHECK(length(trim(canonical_source)) > 0),
    approved_zh_tw TEXT,
    kind TEXT NOT NULL CHECK(length(trim(kind)) > 0),
    scope TEXT NOT NULL CHECK(length(trim(scope)) > 0),
    status TEXT NOT NULL CHECK(status IN ('candidate', 'approved', 'rejected', 'merged')),
    provenance_json TEXT NOT NULL
        CHECK(json_valid(provenance_json) AND json_type(provenance_json) = 'object'),
    first_seen TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_seen TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    merged_into TEXT,
    PRIMARY KEY(job_id, chapter_id, entity_id),
    CHECK(status != 'approved' OR length(trim(approved_zh_tw)) > 0),
    CHECK(status != 'merged' OR merged_into IS NOT NULL),
    CHECK(merged_into IS NULL OR merged_into != entity_id)
);

CREATE TABLE entity_aliases (
    job_id TEXT NOT NULL,
    chapter_id TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    alias TEXT NOT NULL CHECK(length(trim(alias)) > 0),
    normalized_alias TEXT NOT NULL CHECK(length(trim(normalized_alias)) > 0),
    source TEXT NOT NULL CHECK(length(trim(source)) > 0),
    PRIMARY KEY(job_id, chapter_id, normalized_alias),
    FOREIGN KEY(job_id, chapter_id, entity_id)
        REFERENCES chapter_entities(job_id, chapter_id, entity_id) ON DELETE CASCADE
);

CREATE TABLE translation_memory (
    job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
    chapter_id TEXT NOT NULL,
    memory_key TEXT NOT NULL CHECK(length(memory_key) = 64),
    source_nfc TEXT NOT NULL CHECK(length(trim(source_nfc)) > 0),
    context_hash TEXT NOT NULL CHECK(length(context_hash) = 64),
    order_hash TEXT NOT NULL CHECK(length(order_hash) = 64),
    entity_revision_hash TEXT NOT NULL CHECK(length(entity_revision_hash) = 64),
    target_zh_tw TEXT NOT NULL CHECK(length(trim(target_zh_tw)) > 0),
    status TEXT NOT NULL CHECK(status IN ('suggestion', 'approved')),
    reviewer_id TEXT,
    provenance_json TEXT NOT NULL
        CHECK(json_valid(provenance_json) AND json_type(provenance_json) = 'object'),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY(job_id, chapter_id, memory_key),
    CHECK(status != 'approved' OR length(trim(reviewer_id)) > 0)
);

CREATE INDEX idx_chapter_entities_status
    ON chapter_entities(job_id, chapter_id, status);
CREATE INDEX idx_entity_aliases_entity
    ON entity_aliases(job_id, chapter_id, entity_id);
CREATE INDEX idx_translation_memory_source
    ON translation_memory(job_id, chapter_id, source_nfc);
