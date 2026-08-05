from pathlib import Path

from favit_m2tr.preprocess import resolve_ffpp_directories


PAPER_METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]


def _touch_video(directory: Path, name: str = "000.mp4") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).touch()


def test_auto_detects_kaggle_flat_layout(tmp_path):
    _touch_video(tmp_path / "original")
    for method in PAPER_METHODS:
        _touch_video(tmp_path / method, "000_003.mp4")
    original, methods, layout = resolve_ffpp_directories(
        tmp_path, "c23", PAPER_METHODS, "auto"
    )
    assert layout == "kaggle-flat"
    assert original == tmp_path / "original"
    assert methods["Deepfakes"] == tmp_path / "Deepfakes"


def test_auto_prefers_official_layout_when_present(tmp_path):
    _touch_video(tmp_path / "original_sequences" / "youtube" / "c23" / "videos")
    for method in PAPER_METHODS:
        _touch_video(
            tmp_path / "manipulated_sequences" / method / "c23" / "videos",
            "000_003.mp4",
        )
    _, _, layout = resolve_ffpp_directories(tmp_path, "c23", PAPER_METHODS, "auto")
    assert layout == "official"
