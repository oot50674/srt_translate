"""Utilities for managing translation presets in a SQLite database."""

from __future__ import annotations

import os
import sqlite3
import json
from typing import List, Dict, Optional

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "database.db")


def get_connection() -> sqlite3.Connection:
    """Return a connection to the SQLite database."""
    return sqlite3.connect(DB_PATH)


def init_db() -> None:
    """Create tables if they do not exist."""
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS presets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                target_lang TEXT,
                batch_size INTEGER,
                custom_prompt TEXT,
                thinking_budget INTEGER,
                api_key TEXT,
                context_compression INTEGER,
                context_limit INTEGER
            )
            """
        )
        # 마이그레이션: thinking_budget 컬럼이 없으면 추가
        try:
            conn.execute("ALTER TABLE presets ADD COLUMN thinking_budget INTEGER")
        except Exception:
            pass
        # 마이그레이션: api_key 컬럼이 없으면 추가
        try:
            conn.execute("ALTER TABLE presets ADD COLUMN api_key TEXT")
        except Exception:
            pass
        # 마이그레이션: 컨텍스트 압축 옵션 컬럼 추가
        try:
            conn.execute("ALTER TABLE presets ADD COLUMN context_compression INTEGER")
        except Exception:
            pass
        try:
            conn.execute("ALTER TABLE presets ADD COLUMN context_limit INTEGER")
        except Exception:
            pass

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()


def save_preset(
    name: str,
    target_lang: Optional[str] = None,
    batch_size: Optional[int] = None,
    custom_prompt: Optional[str] = None,
    thinking_budget: Optional[int] = None,
    api_key: Optional[str] = None,
    context_compression: Optional[int] = None,
    context_limit: Optional[int] = None,
) -> None:
    """Insert or update a preset in the database."""
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO presets (name, target_lang, batch_size, custom_prompt, thinking_budget, api_key, context_compression, context_limit)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                target_lang=excluded.target_lang,
                batch_size=excluded.batch_size,
                custom_prompt=excluded.custom_prompt,
                thinking_budget=excluded.thinking_budget,
                api_key=excluded.api_key,
                context_compression=excluded.context_compression,
                context_limit=excluded.context_limit
            """,
            (
                name,
                target_lang,
                batch_size,
                custom_prompt,
                thinking_budget,
                api_key,
                context_compression,
                context_limit,
            ),
        )
        conn.commit()


def get_preset(name: str) -> Optional[Dict[str, str | int | None]]:
    """Return a single preset by name."""
    with get_connection() as conn:
        cur = conn.execute(
            """
            SELECT name, target_lang, batch_size, custom_prompt, thinking_budget, api_key,
                   context_compression, context_limit
            FROM presets
            WHERE name=?
            """,
            (name,),
        )
        row = cur.fetchone()
    if row:
        return {
            "name": row[0],
            "target_lang": row[1],
            "batch_size": row[2],
            "custom_prompt": row[3],
            "thinking_budget": row[4],
            "api_key": row[5],
            "context_compression": row[6],
            "context_limit": row[7],
        }
    return None


def list_presets() -> List[Dict[str, str | int | None]]:
    """Return all stored presets."""
    with get_connection() as conn:
        cur = conn.execute(
            """
            SELECT name, target_lang, batch_size, custom_prompt, thinking_budget, api_key,
                   context_compression, context_limit
            FROM presets
            ORDER BY name
            """
        )
        rows = cur.fetchall()
    return [
        {
            "name": name,
            "target_lang": target_lang,
            "batch_size": batch_size,
            "custom_prompt": custom_prompt,
            "thinking_budget": thinking_budget,
            "api_key": api_key,
            "context_compression": context_compression,
            "context_limit": context_limit,
        }
        for (
            name,
            target_lang,
            batch_size,
            custom_prompt,
            thinking_budget,
            api_key,
            context_compression,
            context_limit,
        ) in rows
    ]


def delete_preset(name: str) -> None:
    """Remove a preset from the database."""
    with get_connection() as conn:
        conn.execute("DELETE FROM presets WHERE name=?", (name,))
        conn.commit()


# ----- Job management functions -----


def save_job(job_id: str, job_data: Dict) -> None:
    """Saves a job to the database."""
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO jobs (id, data) VALUES (?, ?)",
            (job_id, json.dumps(job_data, ensure_ascii=False)),
        )
        conn.commit()


def get_job(job_id: str) -> Optional[Dict]:
    """Retrieves a job from the database."""
    with get_connection() as conn:
        cur = conn.execute("SELECT data FROM jobs WHERE id=?", (job_id,))
        row = cur.fetchone()
    if row:
        return json.loads(row[0])
    return None


def delete_job(job_id: str) -> None:
    """Deletes a job from the database."""
    with get_connection() as conn:
        conn.execute("DELETE FROM jobs WHERE id=?", (job_id,))
        conn.commit()


def delete_old_jobs(days: int = 30) -> None:
    """Deletes jobs older than the specified number of days."""
    with get_connection() as conn:
        # created_at이 'now'로부터 {days}일 이전인 모든 job을 삭제합니다.
        conn.execute(
            "DELETE FROM jobs WHERE created_at <= datetime('now', ?)",
            (f'-{days} days',)
        )
        conn.commit()


# Ensure table exists whenever the module is imported
init_db()
