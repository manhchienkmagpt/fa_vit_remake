from favit.metrics import video_level_metrics


def test_video_scores_are_averaged_before_metrics():
    metrics = video_level_metrics(
        [0.1, 0.3, 0.7, 0.9],
        [0, 0, 1, 1],
        ["real", "real", "fake", "fake"],
    )
    assert metrics["video_auc"] == 1.0
    assert metrics["video_accuracy"] == 1.0
    assert metrics["num_videos"] == 2

