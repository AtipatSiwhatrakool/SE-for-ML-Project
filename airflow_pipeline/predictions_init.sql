CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS prediction_logs (
    request_id UUID PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    filename TEXT,
    predicted_class TEXT NOT NULL,
    confidence_score DOUBLE PRECISION NOT NULL,
    requires_manual_review BOOLEAN NOT NULL,
    inference_time_ms DOUBLE PRECISION NOT NULL,
    brightness DOUBLE PRECISION NOT NULL,
    blur_score DOUBLE PRECISION NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    image_bytes BYTEA NOT NULL,
    image_mime TEXT NOT NULL,
    review_status TEXT NOT NULL DEFAULT 'pending'
        CHECK (review_status IN ('pending', 'approved', 'rejected')),
    final_class TEXT,
    reviewed_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_prediction_logs_timestamp
    ON prediction_logs (timestamp);

CREATE INDEX IF NOT EXISTS idx_prediction_logs_review_status
    ON prediction_logs (review_status);

CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'reviewer')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

INSERT INTO users (username, password_hash, role) VALUES
    ('user1', crypt('user1', gen_salt('bf', 10)), 'user'),
    ('user2', crypt('user2', gen_salt('bf', 10)), 'user'),
    ('reviewer1', crypt('reviewer1', gen_salt('bf', 10)), 'reviewer')
ON CONFLICT (username) DO NOTHING;
