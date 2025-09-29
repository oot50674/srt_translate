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
                -- 이하 컬럼(thinking_budget, api_key, context_compression, context_limit)은
                -- 과거 버전 호환을 위한 legacy 필드입니다. (현재 프리셋 로직에서는 저장/사용 안 함)
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
        # 추가 설정(프리셋과 분리된 전역 설정) 저장용 단순 KV 테이블
        # value는 JSON 직렬화된 문자열을 담습니다.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS config (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        conn.commit()


def save_preset(
    name: str,
    target_lang: Optional[str] = None,
    batch_size: Optional[int] = None,
    custom_prompt: Optional[str] = None,
    thinking_budget: Optional[int] = None,  # legacy (현재는 항상 None 전달)
    api_key: Optional[str] = None,          # legacy
    context_compression: Optional[int] = None,  # legacy
    context_limit: Optional[int] = None,         # legacy
) -> None:
    """Insert or update a preset in the database.

    NOTE: 비즈니스 로직 상 현재는 target_lang, batch_size, custom_prompt만 의미를 갖습니다.
    나머지 필드는 하위 호환성을 위해 그대로 저장하지만 새로운 의미는 없습니다.
    """
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
    """Return a single preset by name.

    반환값에는 legacy 필드도 포함되지만 상위 레이어(app.py)에서 필터링하여
    핵심 필드만 클라이언트에 노출합니다.
    """
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
    """Return all stored presets.

    여기서도 legacy 필드를 포함해 반환 (호환성). 필요 시 상위에서 축소.
    """
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

# ----- Config (추가 설정) 관리 함수 -----

def set_config(key: str, value) -> None:
    """단일 설정 값을 저장 (JSON 직렬화)."""
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO config (key, value) VALUES (?, ?) 
                ON CONFLICT(key) DO UPDATE SET value=excluded.value""",
            (key, json.dumps(value, ensure_ascii=False)),
        )
        conn.commit()

def set_configs(mapping: Dict[str, object]) -> None:
    """여러 설정을 한 번에 저장."""
    if not mapping:
        return
    with get_connection() as conn:
        conn.executemany(
            """INSERT INTO config (key, value) VALUES (?, ?) 
                ON CONFLICT(key) DO UPDATE SET value=excluded.value""",
            [(k, json.dumps(v, ensure_ascii=False)) for k, v in mapping.items()],
        )
        conn.commit()

def get_config_value(key: str, default=None):
    """지정된 키의 설정값을 반환 (JSON 역직렬화)."""
    with get_connection() as conn:
        cur = conn.execute("SELECT value FROM config WHERE key=?", (key,))
        row = cur.fetchone()
    if not row:
        return default
    raw = row[0]
    try:
        return json.loads(raw)
    except Exception:
        return default

def get_all_config() -> Dict[str, object]:
    """모든 설정 키/값을 반환."""
    with get_connection() as conn:
        cur = conn.execute("SELECT key, value FROM config")
        rows = cur.fetchall()
    result: Dict[str, object] = {}
    for k, v in rows:
        try:
            result[k] = json.loads(v)
        except Exception:
            result[k] = v
    return result
