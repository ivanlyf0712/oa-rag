-- init.sql – Database schema for the OCR & RAG project
-- Run with: psql -U ocr -d invoices -f init.sql

-- Enable pgvector extension (required for embeddings)
CREATE EXTENSION IF NOT EXISTS vector;

-- Invoices table (existing)
CREATE TABLE IF NOT EXISTS invoices (
    id SERIAL PRIMARY KEY,
    invoice_number VARCHAR(100),
    date VARCHAR(50),
    vendor_name VARCHAR(200),
    total_amount VARCHAR(50),
    currency VARCHAR(10),
    raw_text TEXT,
    embedding VECTOR(1024),
    source_file VARCHAR(500),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Business card contacts
CREATE TABLE IF NOT EXISTS contacts (
    id SERIAL PRIMARY KEY,
    full_name VARCHAR(200),
    job_title VARCHAR(200),
    company VARCHAR(200),
    phone VARCHAR(50),
    email VARCHAR(200),
    website VARCHAR(300),
    address TEXT,
    raw_text TEXT,
    embedding VECTOR(1024),
    source_file VARCHAR(500),
    userid VARCHAR(100) UNIQUE,
    created_at TIMESTAMP DEFAULT NOW()
);

-- WeChat Work messages
CREATE TABLE IF NOT EXISTS messages (
    id SERIAL PRIMARY KEY,
    msgid VARCHAR(100),
    open_kfid VARCHAR(100),
    external_userid VARCHAR(100),
    send_time TIMESTAMPTZ,
    origin INTEGER,
    servicer_userid VARCHAR(100),
    msgtype VARCHAR(20),
    content TEXT,
    raw_json JSONB,
    embedding VECTOR(1024),
    label VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Enable pg_trgm (for fast ILIKE / text search if needed later)
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Indexes for fast search
CREATE INDEX IF NOT EXISTS idx_messages_open_kfid ON messages(open_kfid);
CREATE INDEX IF NOT EXISTS idx_messages_external_userid ON messages(external_userid);
CREATE INDEX IF NOT EXISTS idx_messages_servicer_userid ON messages(servicer_userid);
CREATE INDEX IF NOT EXISTS idx_messages_label ON messages(label);
CREATE INDEX IF NOT EXISTS idx_messages_send_time ON messages(send_time);

-- HNSW index for pgvector cosine similarity (semantic search — Tab 5)
CREATE INDEX IF NOT EXISTS messages_embedding_hnsw_idx
    ON messages USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 200);
