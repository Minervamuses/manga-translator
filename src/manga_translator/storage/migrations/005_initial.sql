CREATE TABLE chapter_entities (
    job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
    chapter_id TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    canonical_source TEXT NOT NULL,
    approved_zh_tw TEXT,
    kind TEXT NOT NULL,
    scope TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('candidate', 'approved', 'rejected', 'merged')),
    provenance_json TEXT NOT NULL,
    first_seen TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_seen TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    merged_into TEXT,
    PRIMARY KEY(job_id, chapter_id, entity_id),
    CHECK(status != 'approved' OR approved_zh_tw IS NOT NULL),
    CHECK(status != 'merged' OR merged_into IS NOT NULL)
);

CREATE TABLE entity_aliases (
    job_id TEXT NOT NULL,
    chapter_id TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    alias TEXT NOT NULL,
    normalized_alias TEXT NOT NULL,
    source TEXT NOT NULL,
    PRIMARY KEY(job_id, chapter_id, normalized_alias),
    FOREIGN KEY(job_id, chapter_id, entity_id)
        REFERENCES chapter_entities(job_id, chapter_id, entity_id) ON DELETE CASCADE
);

CREATE TABLE translation_memory (
    job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
    chapter_id TEXT NOT NULL,
    memory_key TEXT NOT NULL CHECK(length(memory_key) = 64),
    source_nfc TEXT NOT NULL,
    context_hash TEXT NOT NULL CHECK(length(context_hash) = 64),
    order_hash TEXT NOT NULL CHECK(length(order_hash) = 64),
    entity_revision_hash TEXT NOT NULL CHECK(length(entity_revision_hash) = 64),
    target_zh_tw TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('suggestion', 'approved')),
    reviewer_id TEXT,
    provenance_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY(job_id, chapter_id, memory_key),
    CHECK(status != 'approved' OR reviewer_id IS NOT NULL)
);

CREATE INDEX idx_chapter_entities_status
    ON chapter_entities(job_id, chapter_id, status);
CREATE INDEX idx_entity_aliases_entity
    ON entity_aliases(job_id, chapter_id, entity_id);
CREATE INDEX idx_translation_memory_source
    ON translation_memory(job_id, chapter_id, source_nfc);
