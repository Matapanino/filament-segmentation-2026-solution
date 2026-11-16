"""Addendum-1 admission and conservative projection; no fitted models."""
import math


def quality_disjoint(train, query):
    assert train is not None and query
    assert not set(train) & set(query), 'QUALITY_QUERY_OVERLAP'


def budget_projection(upp_update, yolo_update, yolo_batches, profile_seconds,
                      startup_seconds, inference_seconds, upp_checkpoint_bytes,
                      yolo_checkpoint_bytes):
    values = [upp_update, yolo_update, yolo_batches, profile_seconds,
              startup_seconds, inference_seconds, upp_checkpoint_bytes,
              yolo_checkpoint_bytes]
    assert all(math.isfinite(v) and v > 0 for v in values)
    training = 1.2 * 12 * (142 * upp_update + yolo_batches * yolo_update)
    # Charge all 24 complete resume checkpoints; 1.4 MB/s measured lower bound.
    upload = (12 * (upp_checkpoint_bytes + yolo_checkpoint_bytes) + 40_000_000) / 1_400_000
    per = training + max(300, startup_seconds*1.2) + max(400, inference_seconds*1.2) + upload + 180
    total = profile_seconds + 2*per
    return dict(passed=bool(per <= 10800 and total <= 21600),
                training_seconds_per_fold=training, upload_seconds_per_fold=upload,
                per_fold_seconds=per, total_seconds=total, total_hours=total/3600,
                profile_seconds=profile_seconds, job_cap_seconds=10800,
                total_cap_seconds=21600, inference_reserve_seconds=max(400,inference_seconds*1.2),
                transfer_rate_bytes_second=1_400_000)
