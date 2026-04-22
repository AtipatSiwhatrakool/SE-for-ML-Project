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
    height INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_prediction_logs_timestamp
    ON prediction_logs (timestamp);
