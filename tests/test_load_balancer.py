import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "unsloth_finetune" / "training" / "distributed" / "load_balancer.py"
SPEC = importlib.util.spec_from_file_location("load_balancer_under_test", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"无法加载模块: {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

SQLiteTaskQueue = MODULE.SQLiteTaskQueue
build_observed_scheduler_report = MODULE.build_observed_scheduler_report
build_scheduler_comparison = MODULE.build_scheduler_comparison
render_scheduler_report_markdown = MODULE.render_scheduler_report_markdown
simulate_scheduling = MODULE.simulate_scheduling


def _task_payloads(count: int):
    return [
        {
            "sample_index": index,
            "complexity_score": 1.0,
            "image_path": f"/tmp/image_{index}.jpg",
            "query_text": f"query-{index}",
        }
        for index in range(count)
    ]


def test_sqlite_task_queue_claims_without_duplicates(tmp_path: Path):
    queue = SQLiteTaskQueue(tmp_path / "queue.sqlite")
    queue.initialize(_task_payloads(5), metadata={"model_type": "finetuned"})

    batch_rank0 = queue.claim_batch(worker_rank=0, gpu_id=0, batch_size=2)
    batch_rank1 = queue.claim_batch(worker_rank=1, gpu_id=1, batch_size=2)
    batch_rank0_tail = queue.claim_batch(worker_rank=0, gpu_id=0, batch_size=2)
    batch_empty = queue.claim_batch(worker_rank=1, gpu_id=1, batch_size=2)

    assert batch_rank0 == [0, 1]
    assert batch_rank1 == [2, 3]
    assert batch_rank0_tail == [4]
    assert batch_empty == []
    assert queue.get_task_counts() == {"pending": 0, "claimed": 5, "done": 0}


def test_sqlite_task_queue_report_tracks_worker_status_and_outcomes(tmp_path: Path):
    queue = SQLiteTaskQueue(tmp_path / "queue.sqlite")
    queue.initialize(_task_payloads(4), metadata={"model_type": "base", "scheduler_mode": "dynamic_queue"})

    batch_rank0 = queue.claim_batch(worker_rank=0, gpu_id=0, batch_size=2)
    batch_rank1 = queue.claim_batch(worker_rank=1, gpu_id=1, batch_size=2)

    queue.complete_batch(batch_rank0, failed_indices=[batch_rank0[-1]])
    queue.complete_batch(batch_rank1)

    queue.update_worker_status(
        worker_rank=0,
        gpu_id=0,
        state="done",
        claimed_samples=2,
        processed=1,
        failed=1,
        total_inference_seconds=4.0,
        utilization_pct=70.0,
    )
    queue.update_worker_status(
        worker_rank=1,
        gpu_id=1,
        state="done",
        claimed_samples=2,
        processed=2,
        failed=0,
        total_inference_seconds=2.5,
        utilization_pct=55.0,
    )

    report = queue.build_report(total_seconds=5.0, scheduler_mode="dynamic_queue", world_size=2)

    assert report["task_counts"] == {"pending": 0, "claimed": 0, "done": 4}
    assert report["metadata"]["model_type"] == "base"
    assert report["observed"]["makespan_seconds"] == 5.0
    assert report["observed"]["max_min_load_gap_samples"] == 0
    assert report["observed"]["avg_compute_util_pct"] == 62.5


def test_sqlite_task_queue_progress_snapshot_and_completion_validation(tmp_path: Path):
    queue = SQLiteTaskQueue(tmp_path / "queue.sqlite")
    queue.initialize(_task_payloads(3), metadata={"model_type": "finetuned"})

    batch_rank0 = queue.claim_batch(worker_rank=0, gpu_id=0, batch_size=2)
    queue.complete_batch(batch_rank0, failed_indices=[batch_rank0[-1]])
    queue.update_worker_status(worker_rank=0, gpu_id=0, state="done", claimed_samples=2, processed=1, failed=1)

    snapshot = queue.get_progress_snapshot()
    assert snapshot == {
        "total": 3,
        "completed": 2,
        "processed": 1,
        "failed": 1,
        "pending": 1,
        "claimed": 0,
        "remaining": 1,
    }

    try:
        queue.validate_completion(expected_total=3, expected_workers=1, require_done_workers=True)
    except ValueError as exc:
        assert "queue_not_completed" in str(exc)
    else:
        raise AssertionError("validate_completion 应在任务未全部完成时失败")

    batch_rank1 = queue.claim_batch(worker_rank=1, gpu_id=1, batch_size=2)
    queue.complete_batch(batch_rank1)
    queue.update_worker_status(worker_rank=1, gpu_id=1, state="done", claimed_samples=1, processed=1, failed=0)

    completed = queue.validate_completion(expected_total=3, expected_workers=2, require_done_workers=True)
    assert completed["completed"] == 3
    assert completed["remaining"] == 0


def test_dynamic_queue_simulation_improves_uneven_workload_tail():
    sample_costs = [10.0, 9.0, 1.0, 1.0, 1.0, 1.0]

    contiguous = simulate_scheduling(sample_costs, world_size=2, strategy="static_contiguous")
    dynamic = simulate_scheduling(sample_costs, world_size=2, strategy="dynamic_queue")

    assert contiguous["makespan_seconds"] == 20.0
    assert dynamic["makespan_seconds"] == 12.0
    assert dynamic["max_min_load_gap_seconds"] < contiguous["max_min_load_gap_seconds"]
    assert dynamic["avg_worker_busy_pct"] > contiguous["avg_worker_busy_pct"]


def test_scheduler_comparison_and_markdown_report():
    observed = build_observed_scheduler_report(
        [
            {"worker_rank": 0, "processed": 3, "failed": 0, "total_inference_seconds": 12.0, "utilization_pct": 80.0},
            {"worker_rank": 1, "processed": 3, "failed": 0, "total_inference_seconds": 11.0, "utilization_pct": 75.0},
        ],
        total_seconds=12.5,
        world_size=2,
    )
    comparison = build_scheduler_comparison(
        sample_costs=[10.0, 9.0, 1.0, 1.0, 1.0, 1.0],
        world_size=2,
        scheduler_mode="dynamic_queue",
        static_partition_strategy="round_robin",
        observed_report=observed,
    )

    markdown = render_scheduler_report_markdown(
        model_type="finetuned",
        comparison=comparison,
        queue_report={
            "task_counts": {"pending": 0, "claimed": 0, "done": 6},
            "worker_status": [
                {"worker_rank": 0, "gpu_id": 0, "state": "done", "claimed_samples": 3, "processed": 3, "failed": 0, "total_inference_seconds": 12.0, "utilization_pct": 80.0},
                {"worker_rank": 1, "gpu_id": 1, "state": "done", "claimed_samples": 3, "processed": 3, "failed": 0, "total_inference_seconds": 11.0, "utilization_pct": 75.0},
            ],
        },
    )

    assert comparison["preferred_static_baseline"] == "static_round_robin"
    assert "Load Balance Report" in markdown
    assert "dynamic_queue" in markdown
    assert "rank0/gpu0" in markdown

