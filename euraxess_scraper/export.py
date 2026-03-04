from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from . import db


def export_jsonl(conn, output_path: Path) -> int:
    rows = db.jobs_for_export(conn)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    return len(rows)


def export_parquet(conn, output_path: Path) -> int:
    rows = db.jobs_for_export(conn)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = [dict(row) for row in rows]
    table = pa.Table.from_pylist(records)
    pq.write_table(table, output_path)
    return len(rows)
