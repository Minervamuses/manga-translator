CREATE TABLE IF NOT EXISTS provider_response_claims (
    owner_id TEXT PRIMARY KEY,
    claim_token TEXT NOT NULL,
    lease_expires_at_ms INTEGER NOT NULL CHECK(lease_expires_at_ms > 0),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_provider_response_claims_expiry
ON provider_response_claims(lease_expires_at_ms);
