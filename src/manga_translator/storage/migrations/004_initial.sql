ALTER TABLE stage_runs ADD COLUMN run_token TEXT;

CREATE TABLE IF NOT EXISTS page_run_claims (
    job_id TEXT NOT NULL,
    page_id TEXT NOT NULL,
    claim_token TEXT NOT NULL,
    lease_expires_at_ms INTEGER NOT NULL CHECK(lease_expires_at_ms > 0),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY(job_id, page_id),
    FOREIGN KEY(job_id, page_id) REFERENCES pages(job_id, page_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_page_run_claims_expiry
ON page_run_claims(lease_expires_at_ms);
