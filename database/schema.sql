-- Optional MySQL Workbench setup for a NEW project database.
-- Review the database name before running. No application account is created.
-- The Flask application can also create these tables automatically on startup.

CREATE DATABASE IF NOT EXISTS veritas
  CHARACTER SET utf8mb4 COLLATE utf8mb4_bin;
USE veritas;

CREATE TABLE IF NOT EXISTS users (
  id VARCHAR(128) NOT NULL,
  created_at DATETIME(6) NOT NULL,
  body JSON NOT NULL,
  email VARCHAR(254),
  PRIMARY KEY (id),
  CONSTRAINT uq_users_email UNIQUE (email)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

CREATE TABLE IF NOT EXISTS reports (
  id VARCHAR(128) NOT NULL,
  created_at DATETIME(6) NOT NULL,
  body JSON NOT NULL,
  owner_id VARCHAR(128),
  status VARCHAR(32),
  PRIMARY KEY (id),
  KEY ix_reports_owner_created (owner_id, created_at),
  KEY ix_reports_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

CREATE TABLE IF NOT EXISTS sources (
  id VARCHAR(128) NOT NULL,
  created_at DATETIME(6) NOT NULL,
  body JSON NOT NULL,
  domain VARCHAR(253),
  PRIMARY KEY (id),
  CONSTRAINT uq_sources_domain UNIQUE (domain)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

CREATE TABLE IF NOT EXISTS datasets (
  id VARCHAR(128) NOT NULL,
  created_at DATETIME(6) NOT NULL,
  body JSON NOT NULL,
  name VARCHAR(200),
  version VARCHAR(200),
  PRIMARY KEY (id),
  CONSTRAINT uq_datasets_name_version UNIQUE (name, version)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

CREATE TABLE IF NOT EXISTS password_resets (
  id VARCHAR(128) NOT NULL,
  created_at DATETIME(6) NOT NULL,
  body JSON NOT NULL,
  user_id VARCHAR(128),
  expires_at DATETIME(6),
  PRIMARY KEY (id),
  KEY ix_password_resets_user (user_id),
  KEY ix_password_resets_expiry (expires_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

CREATE TABLE IF NOT EXISTS app_schema_version (
  version INTEGER NOT NULL,
  installed_at DATETIME(6) NOT NULL,
  PRIMARY KEY (version)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

INSERT INTO app_schema_version (version, installed_at)
SELECT 1, UTC_TIMESTAMP(6)
WHERE NOT EXISTS (SELECT 1 FROM app_schema_version);
