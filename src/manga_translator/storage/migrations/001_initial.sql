CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    config_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS artifacts (
    sha256 TEXT PRIMARY KEY CHECK(length(sha256) = 64),
    media_type TEXT NOT NULL,
    size_bytes INTEGER NOT NULL CHECK(size_bytes >= 0),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS pages (
    job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
    page_id TEXT NOT NULL CHECK(length(page_id) = 64),
    document_artifact_sha256 TEXT NOT NULL REFERENCES artifacts(sha256),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (job_id, page_id)
);

CREATE TABLE IF NOT EXISTS stage_runs (
    stage_run_id INTEGER PRIMARY KEY,
    job_id TEXT NOT NULL,
    page_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    fingerprint TEXT NOT NULL CHECK(length(fingerprint) = 64),
    status TEXT NOT NULL,
    input_hashes_json TEXT NOT NULL DEFAULT '[]',
    output_hashes_json TEXT NOT NULL DEFAULT '[]',
    started_at TEXT,
    finished_at TEXT,
    UNIQUE(job_id, page_id, stage, fingerprint),
    FOREIGN KEY(job_id, page_id) REFERENCES pages(job_id, page_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS region_identities (
    region_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    page_id TEXT NOT NULL,
    active_revision_id TEXT NOT NULL,
    lineage_json TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY(job_id, page_id) REFERENCES pages(job_id, page_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS region_revisions (
    revision_id TEXT PRIMARY KEY CHECK(length(revision_id) = 64),
    region_id TEXT NOT NULL REFERENCES region_identities(region_id) ON DELETE CASCADE,
    revision_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS issues (
    issue_id INTEGER PRIMARY KEY,
    job_id TEXT NOT NULL,
    page_id TEXT NOT NULL,
    region_id TEXT,
    stage TEXT NOT NULL,
    code TEXT NOT NULL,
    severity TEXT NOT NULL,
    issue_json TEXT NOT NULL,
    FOREIGN KEY(job_id, page_id) REFERENCES pages(job_id, page_id) ON DELETE CASCADE,
    FOREIGN KEY(region_id) REFERENCES region_identities(region_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS entities (
    job_id TEXT NOT NULL,
    page_id TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    entity_json TEXT NOT NULL,
    PRIMARY KEY(job_id, page_id, entity_id),
    FOREIGN KEY(job_id, page_id) REFERENCES pages(job_id, page_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS artifact_references (
    owner_type TEXT NOT NULL,
    owner_id TEXT NOT NULL,
    sha256 TEXT NOT NULL REFERENCES artifacts(sha256) ON DELETE CASCADE,
    PRIMARY KEY(owner_type, owner_id, sha256)
);

CREATE INDEX IF NOT EXISTS idx_stage_runs_status ON stage_runs(status);
CREATE INDEX IF NOT EXISTS idx_artifact_references_sha ON artifact_references(sha256);
