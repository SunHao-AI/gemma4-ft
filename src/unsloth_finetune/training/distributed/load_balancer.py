"""Utilities for dynamic multi-GPU inference load balancing."""

from __future__ import annotations

import json
import math
import sqlite3
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    import torch
except Exception:  # pragma: no cover - unit tests may mock torch import.
    torch = None


def _query_nvidia_smi() -> Dict[int, Dict[str, float]]:
    stats: Dict[int, Dict[str, float]] = {}
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return stats

    if result.returncode != 0:
        return stats

    for line in result.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            gpu_index = int(parts[0])
            stats[gpu_index] = {
                "utilization_pct": float(parts[1]),
                "temperature_c": float(parts[2]),
            }
        except ValueError:
            continue
    return stats


def collect_local_gpu_metrics(local_rank: int, physical_gpu: int) -> Dict[str, float]:
    """Collect local GPU memory/utilization metrics for worker heartbeats."""
    metrics = {
        "memory_alloc_gb": 0.0,
        "memory_reserved_gb": 0.0,
        "utilization_pct": 0.0,
        "temperature_c": 0.0,
    }

    if torch is not None and getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
        try:
            metrics["memory_alloc_gb"] = round(torch.cuda.memory_allocated(local_rank) / 1024**3, 3)
            metrics["memory_reserved_gb"] = round(torch.cuda.memory_reserved(local_rank) / 1024**3, 3)
        except Exception:
            pass

    smi_stats = _query_nvidia_smi()
    physical_stats = smi_stats.get(physical_gpu, {})
    if physical_stats:
        metrics["utilization_pct"] = round(float(physical_stats.get("utilization_pct", 0.0)), 1)
        metrics["temperature_c"] = round(float(physical_stats.get("temperature_c", 0.0)), 1)
    return metrics


class SQLiteTaskQueue:
    """SQLite-backed task queue for single-node multi-process scheduling."""

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(self.db_path), timeout=30, isolation_level=None, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        return connection

    def initialize(
        self,
        task_payloads: Iterable[Dict[str, Any]],
        *,
        metadata: Optional[Dict[str, Any]] = None,
        reset: bool = True,
    ) -> None:
        if reset and self.db_path.exists():
            self.db_path.unlink()

        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    sample_index INTEGER PRIMARY KEY,
                    status TEXT NOT NULL,
                    outcome TEXT NOT NULL DEFAULT '',
                    worker_rank INTEGER,
                    worker_gpu INTEGER,
                    claim_count INTEGER NOT NULL DEFAULT 0,
                    complexity_score REAL NOT NULL DEFAULT 1.0,
                    claimed_at REAL,
                    completed_at REAL,
                    image_path TEXT NOT NULL DEFAULT '',
                    query_text TEXT NOT NULL DEFAULT '',
                    last_error TEXT NOT NULL DEFAULT ''
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS worker_status (
                    worker_rank INTEGER PRIMARY KEY,
                    gpu_id INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    claimed_samples INTEGER NOT NULL DEFAULT 0,
                    processed INTEGER NOT NULL DEFAULT 0,
                    failed INTEGER NOT NULL DEFAULT 0,
                    last_heartbeat REAL NOT NULL,
                    current_batch TEXT NOT NULL DEFAULT '',
                    last_batch_seconds REAL NOT NULL DEFAULT 0.0,
                    total_inference_seconds REAL NOT NULL DEFAULT 0.0,
                    queue_wait_seconds REAL NOT NULL DEFAULT 0.0,
                    memory_alloc_gb REAL NOT NULL DEFAULT 0.0,
                    memory_reserved_gb REAL NOT NULL DEFAULT 0.0,
                    utilization_pct REAL NOT NULL DEFAULT 0.0,
                    temperature_c REAL NOT NULL DEFAULT 0.0,
                    last_error TEXT NOT NULL DEFAULT ''
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )

            rows = []
            for payload in task_payloads:
                rows.append(
                    (
                        int(payload["sample_index"]),
                        "pending",
                        float(payload.get("complexity_score", 1.0)),
                        str(payload.get("image_path", "")),
                        str(payload.get("query_text", "")),
                    )
                )
            connection.executemany(
                """
                INSERT INTO tasks (
                    sample_index, status, complexity_score, image_path, query_text
                ) VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )

            if metadata:
                connection.executemany(
                    "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                    [(str(key), json.dumps(value, ensure_ascii=False)) for key, value in metadata.items()],
                )

    def claim_batch(self, worker_rank: int, gpu_id: int, batch_size: int) -> List[int]:
        batch_size = max(1, int(batch_size or 1))
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                """
                SELECT sample_index
                FROM tasks
                WHERE status = 'pending'
                ORDER BY complexity_score DESC, sample_index ASC
                LIMIT ?
                """,
                (batch_size,),
            ).fetchall()
            sample_indices = [int(row["sample_index"]) for row in rows]
            if not sample_indices:
                connection.commit()
                return []

            placeholders = ",".join("?" for _ in sample_indices)
            params: List[Any] = [worker_rank, gpu_id, time.time(), *sample_indices]
            connection.execute(
                f"""
                UPDATE tasks
                SET status = 'claimed',
                    worker_rank = ?,
                    worker_gpu = ?,
                    claim_count = claim_count + 1,
                    claimed_at = ?
                WHERE sample_index IN ({placeholders})
                  AND status = 'pending'
                """,
                params,
            )
            connection.commit()
        return sample_indices

    def complete_batch(
        self,
        sample_indices: Iterable[int],
        *,
        failed_indices: Optional[Iterable[int]] = None,
        error_messages: Optional[Dict[int, str]] = None,
    ) -> None:
        sample_indices = [int(index) for index in sample_indices]
        failed_set = {int(index) for index in (failed_indices or [])}
        error_messages = {int(key): value for key, value in (error_messages or {}).items()}
        timestamp = time.time()
        with self._connect() as connection:
            rows = []
            for sample_index in sample_indices:
                rows.append(
                    (
                        "failed" if sample_index in failed_set else "processed",
                        timestamp,
                        str(error_messages.get(sample_index, "")),
                        sample_index,
                    )
                )
            connection.executemany(
                """
                UPDATE tasks
                SET status = 'done',
                    outcome = ?,
                    completed_at = ?,
                    last_error = ?
                WHERE sample_index = ?
                """,
                rows,
            )

    def update_worker_status(
        self,
        *,
        worker_rank: int,
        gpu_id: int,
        state: str,
        claimed_samples: int,
        processed: int,
        failed: int,
        current_batch: str = "",
        last_batch_seconds: float = 0.0,
        total_inference_seconds: float = 0.0,
        queue_wait_seconds: float = 0.0,
        memory_alloc_gb: float = 0.0,
        memory_reserved_gb: float = 0.0,
        utilization_pct: float = 0.0,
        temperature_c: float = 0.0,
        last_error: str = "",
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO worker_status (
                    worker_rank, gpu_id, state, claimed_samples, processed, failed, last_heartbeat,
                    current_batch, last_batch_seconds, total_inference_seconds, queue_wait_seconds,
                    memory_alloc_gb, memory_reserved_gb, utilization_pct, temperature_c, last_error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(worker_rank) DO UPDATE SET
                    gpu_id = excluded.gpu_id,
                    state = excluded.state,
                    claimed_samples = excluded.claimed_samples,
                    processed = excluded.processed,
                    failed = excluded.failed,
                    last_heartbeat = excluded.last_heartbeat,
                    current_batch = excluded.current_batch,
                    last_batch_seconds = excluded.last_batch_seconds,
                    total_inference_seconds = excluded.total_inference_seconds,
                    queue_wait_seconds = excluded.queue_wait_seconds,
                    memory_alloc_gb = excluded.memory_alloc_gb,
                    memory_reserved_gb = excluded.memory_reserved_gb,
                    utilization_pct = excluded.utilization_pct,
                    temperature_c = excluded.temperature_c,
                    last_error = excluded.last_error
                """,
                (
                    worker_rank,
                    gpu_id,
                    state,
                    claimed_samples,
                    processed,
                    failed,
                    time.time(),
                    current_batch,
                    last_batch_seconds,
                    total_inference_seconds,
                    queue_wait_seconds,
                    memory_alloc_gb,
                    memory_reserved_gb,
                    utilization_pct,
                    temperature_c,
                    last_error,
                ),
            )

    def get_worker_status_rows(self) -> List[Dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM worker_status ORDER BY worker_rank ASC").fetchall()
        return [dict(row) for row in rows]

    def get_task_counts(self) -> Dict[str, int]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT status, COUNT(*) AS task_count
                FROM tasks
                GROUP BY status
                """
            ).fetchall()
        counts = {"pending": 0, "claimed": 0, "done": 0}
        for row in rows:
            counts[str(row["status"])] = int(row["task_count"])
        return counts

    def get_outcome_counts(self) -> Dict[str, int]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT outcome, COUNT(*) AS task_count
                FROM tasks
                WHERE status = 'done'
                GROUP BY outcome
                """
            ).fetchall()
        counts = {"processed": 0, "failed": 0}
        for row in rows:
            outcome = str(row["outcome"] or "")
            if outcome in counts:
                counts[outcome] = int(row["task_count"])
        return counts

    def pending_count(self) -> int:
        return int(self.get_task_counts().get("pending", 0))

    def get_progress_snapshot(self) -> Dict[str, int]:
        task_counts = self.get_task_counts()
        outcome_counts = self.get_outcome_counts()
        total = sum(int(value) for value in task_counts.values())
        completed = int(task_counts.get("done", 0))
        pending = int(task_counts.get("pending", 0))
        claimed = int(task_counts.get("claimed", 0))
        failed = int(outcome_counts.get("failed", 0))
        processed = int(outcome_counts.get("processed", 0))
        return {
            "total": total,
            "completed": completed,
            "processed": processed,
            "failed": failed,
            "pending": pending,
            "claimed": claimed,
            "remaining": pending + claimed,
        }

    def validate_completion(
        self,
        *,
        expected_total: Optional[int] = None,
        expected_workers: Optional[int] = None,
        require_done_workers: bool = False,
    ) -> Dict[str, int]:
        snapshot = self.get_progress_snapshot()
        worker_rows = self.get_worker_status_rows()

        if expected_total is not None and snapshot["total"] != int(expected_total):
            raise ValueError(
                f"queue_total_mismatch: expected={int(expected_total)}, actual={snapshot['total']}"
            )
        if expected_total is not None and snapshot["completed"] != int(expected_total):
            raise ValueError(
                f"queue_not_completed: expected_done={int(expected_total)}, actual_done={snapshot['completed']}, "
                f"claimed={snapshot['claimed']}, pending={snapshot['pending']}"
            )
        if snapshot["pending"] != 0 or snapshot["claimed"] != 0:
            raise ValueError(
                f"queue_has_inflight_tasks: pending={snapshot['pending']}, claimed={snapshot['claimed']}"
            )

        if expected_workers is not None and len(worker_rows) != int(expected_workers):
            raise ValueError(
                f"worker_count_mismatch: expected={int(expected_workers)}, actual={len(worker_rows)}"
            )
        if require_done_workers:
            not_done = [
                f"rank{row.get('worker_rank')}={row.get('state')}"
                for row in worker_rows
                if str(row.get("state")) != "done"
            ]
            if not_done:
                raise ValueError(f"workers_not_done: {', '.join(not_done)}")
        return snapshot

    def load_metadata(self) -> Dict[str, Any]:
        with self._connect() as connection:
            rows = connection.execute("SELECT key, value FROM metadata").fetchall()
        metadata: Dict[str, Any] = {}
        for row in rows:
            try:
                metadata[str(row["key"])] = json.loads(row["value"])
            except json.JSONDecodeError:
                metadata[str(row["key"])] = row["value"]
        return metadata

    def build_report(self, total_seconds: float, scheduler_mode: str, world_size: int) -> Dict[str, Any]:
        counts = self.get_task_counts()
        worker_rows = self.get_worker_status_rows()
        observed = build_observed_scheduler_report(worker_rows, total_seconds=total_seconds, world_size=world_size)
        return {
            "scheduler_mode": scheduler_mode,
            "world_size": world_size,
            "task_counts": counts,
            "worker_status": worker_rows,
            "observed": observed,
            "metadata": self.load_metadata(),
        }


def build_observed_scheduler_report(
    worker_rows: Iterable[Dict[str, Any]],
    *,
    total_seconds: float,
    world_size: int,
) -> Dict[str, Any]:
    worker_rows = [dict(row) for row in worker_rows]
    busy_seconds = []
    sample_counts = []
    compute_utils = []

    for row in worker_rows:
        row_busy_seconds = float(row.get("total_inference_seconds", 0.0))
        busy_seconds.append(row_busy_seconds)
        sample_counts.append(int(row.get("processed", 0) + row.get("failed", 0)))
        compute_utils.append(float(row.get("utilization_pct", 0.0)))

    if not worker_rows:
        return {
            "worker_count": world_size,
            "makespan_seconds": round(float(total_seconds), 3),
            "avg_worker_busy_pct": 0.0,
            "avg_compute_util_pct": 0.0,
            "max_min_load_gap_seconds": 0.0,
            "max_min_load_gap_samples": 0,
            "per_worker_busy_seconds": {},
            "per_worker_samples": {},
        }

    makespan = max(float(total_seconds), max(busy_seconds) if busy_seconds else 0.0)
    avg_busy_pct = (sum(seconds / makespan for seconds in busy_seconds) / max(1, world_size)) * 100 if makespan > 0 else 0.0
    per_worker_busy = {f"rank{row['worker_rank']}": round(float(row.get("total_inference_seconds", 0.0)), 3) for row in worker_rows}
    per_worker_samples = {f"rank{row['worker_rank']}": int(row.get("processed", 0) + row.get("failed", 0)) for row in worker_rows}

    return {
        "worker_count": world_size,
        "makespan_seconds": round(makespan, 3),
        "avg_worker_busy_pct": round(avg_busy_pct, 2),
        "avg_compute_util_pct": round(sum(compute_utils) / len(compute_utils), 2) if compute_utils else 0.0,
        "max_min_load_gap_seconds": round((max(busy_seconds) - min(busy_seconds)) if busy_seconds else 0.0, 3),
        "max_min_load_gap_samples": int((max(sample_counts) - min(sample_counts)) if sample_counts else 0),
        "per_worker_busy_seconds": per_worker_busy,
        "per_worker_samples": per_worker_samples,
    }


def _normalize_sample_costs(sample_costs: Iterable[float]) -> List[float]:
    values = [max(0.0, float(value)) for value in sample_costs]
    return [value for value in values if value >= 0.0]


def simulate_scheduling(sample_costs: Iterable[float], world_size: int, strategy: str) -> Dict[str, Any]:
    sample_costs = _normalize_sample_costs(sample_costs)
    world_size = max(1, int(world_size or 1))
    worker_loads = [0.0 for _ in range(world_size)]
    worker_counts = [0 for _ in range(world_size)]

    if strategy == "static_contiguous":
        total = len(sample_costs)
        base = total // world_size
        remainder = total % world_size
        start = 0
        for worker_rank in range(world_size):
            chunk_size = base + (1 if worker_rank < remainder else 0)
            chunk = sample_costs[start : start + chunk_size]
            worker_loads[worker_rank] = sum(chunk)
            worker_counts[worker_rank] = len(chunk)
            start += chunk_size
    elif strategy == "static_round_robin":
        for index, cost in enumerate(sample_costs):
            worker_rank = index % world_size
            worker_loads[worker_rank] += cost
            worker_counts[worker_rank] += 1
    elif strategy == "dynamic_queue":
        for cost in sorted(sample_costs, reverse=True):
            worker_rank = min(range(world_size), key=lambda idx: worker_loads[idx])
            worker_loads[worker_rank] += cost
            worker_counts[worker_rank] += 1
    else:
        raise ValueError(f"未知调度策略: {strategy}")

    makespan = max(worker_loads) if worker_loads else 0.0
    avg_utilization = 0.0
    if makespan > 0:
        avg_utilization = (sum(load / makespan for load in worker_loads) / world_size) * 100

    return {
        "strategy": strategy,
        "worker_count": world_size,
        "sample_count": len(sample_costs),
        "makespan_seconds": round(makespan, 3),
        "avg_worker_busy_pct": round(avg_utilization, 2),
        "max_min_load_gap_seconds": round((max(worker_loads) - min(worker_loads)) if worker_loads else 0.0, 3),
        "max_min_load_gap_samples": int((max(worker_counts) - min(worker_counts)) if worker_counts else 0),
        "per_worker_busy_seconds": {f"rank{idx}": round(load, 3) for idx, load in enumerate(worker_loads)},
        "per_worker_samples": {f"rank{idx}": int(count) for idx, count in enumerate(worker_counts)},
    }


def build_scheduler_comparison(
    *,
    sample_costs: Iterable[float],
    world_size: int,
    scheduler_mode: str,
    static_partition_strategy: str,
    observed_report: Dict[str, Any],
) -> Dict[str, Any]:
    baselines = {
        "static_contiguous": simulate_scheduling(sample_costs, world_size, "static_contiguous"),
        "static_round_robin": simulate_scheduling(sample_costs, world_size, "static_round_robin"),
        "dynamic_queue": simulate_scheduling(sample_costs, world_size, "dynamic_queue"),
    }
    preferred_static_key = "static_round_robin" if static_partition_strategy == "round_robin" else "static_contiguous"
    preferred_static = baselines[preferred_static_key]
    observed_makespan = float(observed_report.get("makespan_seconds", 0.0))
    static_makespan = float(preferred_static.get("makespan_seconds", 0.0))
    makespan_delta = static_makespan - observed_makespan
    improvement_pct = (makespan_delta / static_makespan * 100.0) if static_makespan > 0 else 0.0

    return {
        "scheduler_mode": scheduler_mode,
        "preferred_static_baseline": preferred_static_key,
        "observed": observed_report,
        "simulated": baselines,
        "makespan_delta_seconds_vs_static": round(makespan_delta, 3),
        "improvement_pct_vs_static": round(improvement_pct, 2),
    }


def render_scheduler_report_markdown(
    *,
    model_type: str,
    comparison: Dict[str, Any],
    queue_report: Dict[str, Any],
) -> str:
    observed = comparison.get("observed", {})
    simulated = comparison.get("simulated", {})
    static_key = comparison.get("preferred_static_baseline", "static_round_robin")
    static_report = simulated.get(static_key, {})

    lines = [
        f"# {model_type} Load Balance Report",
        "",
        f"- Scheduler mode: `{comparison.get('scheduler_mode', 'unknown')}`",
        f"- Preferred static baseline: `{static_key}`",
        f"- Task counts: `{json.dumps(queue_report.get('task_counts', {}), ensure_ascii=False)}`",
        f"- Observed makespan: `{observed.get('makespan_seconds', 0.0):.3f}s`",
        f"- Static baseline makespan: `{static_report.get('makespan_seconds', 0.0):.3f}s`",
        f"- Makespan delta vs static: `{comparison.get('makespan_delta_seconds_vs_static', 0.0):.3f}s`",
        f"- Improvement vs static: `{comparison.get('improvement_pct_vs_static', 0.0):.2f}%`",
        f"- Observed avg worker busy: `{observed.get('avg_worker_busy_pct', 0.0):.2f}%`",
        f"- Observed avg compute util: `{observed.get('avg_compute_util_pct', 0.0):.2f}%`",
        f"- Observed load gap: `{observed.get('max_min_load_gap_seconds', 0.0):.3f}s / {observed.get('max_min_load_gap_samples', 0)} samples`",
        "",
        "## Simulated Baselines",
    ]

    for key, report in simulated.items():
        lines.append(
            f"- `{key}`: makespan={report.get('makespan_seconds', 0.0):.3f}s, "
            f"avg_busy={report.get('avg_worker_busy_pct', 0.0):.2f}%, "
            f"load_gap={report.get('max_min_load_gap_seconds', 0.0):.3f}s"
        )

    lines.extend(["", "## Observed Workers"])
    for worker in queue_report.get("worker_status", []):
        lines.append(
            f"- `rank{worker.get('worker_rank')}/gpu{worker.get('gpu_id')}`: "
            f"state={worker.get('state')}, claimed={worker.get('claimed_samples', 0)}, "
            f"processed={worker.get('processed', 0)}, failed={worker.get('failed', 0)}, "
            f"busy={float(worker.get('total_inference_seconds', 0.0)):.3f}s, "
            f"util={float(worker.get('utilization_pct', 0.0)):.1f}%"
        )
    lines.append("")
    return "\n".join(lines)


def safe_mean(values: Iterable[float]) -> float:
    values = [float(value) for value in values]
    return sum(values) / len(values) if values else 0.0


def stddev(values: Iterable[float]) -> float:
    values = [float(value) for value in values]
    if len(values) <= 1:
        return 0.0
    avg = safe_mean(values)
    variance = sum((value - avg) ** 2 for value in values) / len(values)
    return math.sqrt(variance)
