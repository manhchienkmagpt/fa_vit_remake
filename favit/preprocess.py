from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

LOGGER = logging.getLogger("favit.preprocess")
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


@dataclass(frozen=True)
class ExtractedFrame:
    path: str
    sample_index: int
    source_frame_index: int


class MTCNNFaceCropper:
    """MTCNN face cropper matching the detector family reported by the paper."""

    def __init__(self, image_size: int, margin: int, device: str) -> None:
        try:
            from facenet_pytorch import MTCNN
        except ImportError as error:
            raise RuntimeError(
                "Face extraction needs facenet-pytorch. Install with "
                "`pip install -e .[preprocess]`."
            ) from error
        self.detector = MTCNN(
            image_size=image_size,
            margin=margin,
            keep_all=False,
            select_largest=True,
            post_process=False,
            device=torch.device(device),
        )

    def __call__(self, rgb_frame: np.ndarray) -> Image.Image | None:
        face = self.detector(Image.fromarray(rgb_frame))
        if face is None:
            return None
        array = face.clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        return Image.fromarray(array, mode="RGB")


def _uniform_indices(frame_count: int, samples: int) -> list[int]:
    if frame_count <= 0:
        return []
    return np.rint(np.linspace(0, frame_count - 1, samples)).astype(int).tolist()


def extract_video(
    video_path: Path,
    output_directory: Path,
    samples: int,
    cropper: MTCNNFaceCropper,
    output_root: Path,
    overwrite: bool,
) -> list[ExtractedFrame]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        LOGGER.warning("Cannot open video: %s", video_path)
        return []
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = _uniform_indices(frame_count, samples)
    output_directory.mkdir(parents=True, exist_ok=True)
    extracted: list[ExtractedFrame] = []
    for sample_index, frame_index in enumerate(indices):
        destination = output_directory / f"{sample_index:03d}.jpg"
        if destination.exists() and not overwrite:
            extracted.append(
                ExtractedFrame(
                    destination.relative_to(output_root).as_posix(), sample_index, frame_index
                )
            )
            continue
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            LOGGER.warning("Cannot read frame %d from %s", frame_index, video_path)
            continue
        face = cropper(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if face is None:
            LOGGER.warning("No face at frame %d in %s", frame_index, video_path)
            continue
        face.save(destination, format="JPEG", quality=95, subsampling=0)
        extracted.append(
            ExtractedFrame(
                destination.relative_to(output_root).as_posix(), sample_index, frame_index
            )
        )
    capture.release()
    return extracted


def _find_videos(directory: Path) -> dict[str, Path]:
    if not directory.exists():
        raise FileNotFoundError(f"video directory does not exist: {directory}")
    return {
        path.stem: path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    }


def _video_directory(base: Path) -> Path:
    videos = base / "videos"
    return videos if videos.exists() else base


def _write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    LOGGER.info("Wrote %d rows to %s", len(rows), path)


def preprocess_ffpp(args: argparse.Namespace) -> None:
    root, output_root = args.root.resolve(), args.output.resolve()
    with args.split_json.open("r", encoding="utf-8") as handle:
        split_pairs = json.load(handle)
    if not all(isinstance(pair, list) and len(pair) == 2 for pair in split_pairs):
        raise ValueError("FF++ split JSON must contain [target, source] pairs")

    originals = _find_videos(
        _video_directory(root / "original_sequences" / "youtube" / args.compression)
    )
    manipulation_videos = {
        method: _find_videos(
            _video_directory(root / "manipulated_sequences" / method / args.compression)
        )
        for method in args.methods
    }
    cropper = MTCNNFaceCropper(args.image_size, args.margin, args.device)
    selected_ids = sorted({str(item) for pair in split_pairs for item in pair})
    real_frames: dict[str, list[ExtractedFrame]] = {}
    for video_id in tqdm(selected_ids, desc=f"FF++ {args.split} real videos"):
        if video_id not in originals:
            LOGGER.warning("Missing FF++ original video %s", video_id)
            continue
        real_frames[video_id] = extract_video(
            originals[video_id],
            output_root
            / "faces"
            / "ffpp"
            / args.compression
            / args.split
            / "real"
            / video_id,
            args.frames,
            cropper,
            output_root,
            args.overwrite,
        )

    frame_rows: list[dict[str, object]] = []
    for video_id, frames in real_frames.items():
        frame_rows.extend(
            {
                "path": frame.path,
                "label": 0,
                "video_id": f"real/{video_id}",
                "frame_index": frame.source_frame_index,
                "dataset": "ffpp",
                "split": args.split,
                "method": "real",
            }
            for frame in frames
        )

    pair_rows: list[dict[str, object]] = []
    for first, second in tqdm(split_pairs, desc=f"FF++ {args.split} fake videos"):
        for target, source in ((str(first), str(second)), (str(second), str(first))):
            fake_id = f"{target}_{source}"
            real_by_sample = {
                frame.sample_index: frame for frame in real_frames.get(target, [])
            }
            for method, videos in manipulation_videos.items():
                video_path = videos.get(fake_id)
                if video_path is None:
                    LOGGER.warning("Missing %s/%s", method, fake_id)
                    continue
                fake_frames = extract_video(
                    video_path,
                    output_root
                    / "faces"
                    / "ffpp"
                    / args.compression
                    / args.split
                    / "fake"
                    / method
                    / fake_id,
                    args.frames,
                    cropper,
                    output_root,
                    args.overwrite,
                )
                frame_rows.extend(
                    {
                        "path": frame.path,
                        "label": 1,
                        "video_id": f"fake/{method}/{fake_id}",
                        "frame_index": frame.source_frame_index,
                        "dataset": "ffpp",
                        "split": args.split,
                        "method": method,
                    }
                    for frame in fake_frames
                )
                for fake_frame in fake_frames:
                    real_frame = real_by_sample.get(fake_frame.sample_index)
                    if real_frame is not None:
                        pair_rows.append(
                            {
                                "fake_path": fake_frame.path,
                                "real_path": real_frame.path,
                                "video_id": fake_id,
                                "method": method,
                                "sample_index": fake_frame.sample_index,
                            }
                        )

    prefix = f"ffpp_{args.compression}_{args.split}"
    manifests = output_root / "manifests"
    _write_csv(
        manifests / f"{prefix}_frames.csv",
        frame_rows,
        ["path", "label", "video_id", "frame_index", "dataset", "split", "method"],
    )
    _write_csv(
        manifests / f"{prefix}_pairs.csv",
        pair_rows,
        ["fake_path", "real_path", "video_id", "method", "sample_index"],
    )


def preprocess_celebdf(args: argparse.Namespace) -> None:
    root, output_root = args.root.resolve(), args.output.resolve()
    cropper = MTCNNFaceCropper(args.image_size, args.margin, args.device)
    entries: list[tuple[int, str]] = []
    with args.test_list.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2 or parts[0] not in {"0", "1"}:
                raise ValueError(f"invalid Celeb-DF test-list line {line_number}: {line}")
            # Official Celeb-DF convention: 1=real, 0=fake. This project uses
            # 0=real and 1=fake to match equation (16) in FA-ViT.
            label = 0 if int(parts[0]) == 1 else 1
            entries.append((label, parts[1].replace("\\", "/")))

    rows: list[dict[str, object]] = []
    for label, relative_video in tqdm(entries, desc="Celeb-DF test videos"):
        video_path = root / Path(relative_video)
        if not video_path.exists():
            # Some mirrors add a `videos` level below each official folder.
            relative_path = Path(relative_video)
            video_path = root / relative_path.parent / "videos" / relative_path.name
        if not video_path.exists():
            LOGGER.warning("Missing Celeb-DF video %s", video_path)
            continue
        video_id = Path(relative_video).with_suffix("").as_posix()
        safe_id = video_id.replace("/", "__")
        frames = extract_video(
            video_path,
            output_root / "faces" / "celebdf" / "test" / str(label) / safe_id,
            args.frames,
            cropper,
            output_root,
            args.overwrite,
        )
        rows.extend(
            {
                "path": frame.path,
                "label": label,
                "video_id": video_id,
                "frame_index": frame.source_frame_index,
                "dataset": "celebdf",
                "split": "test",
                "method": "real" if label == 0 else "fake",
            }
            for frame in frames
        )
    _write_csv(
        output_root / "manifests" / "celebdf_test_frames.csv",
        rows,
        ["path", "label", "video_id", "frame_index", "dataset", "split", "method"],
    )


def _common_arguments(parser: argparse.ArgumentParser, default_frames: int) -> None:
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frames", type=int, default=default_frames)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--margin", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FA-ViT face extraction and manifest creation")
    subparsers = parser.add_subparsers(dest="dataset", required=True)
    ffpp = subparsers.add_parser("ffpp", help="preprocess one official FF++ split")
    _common_arguments(ffpp, default_frames=20)
    ffpp.add_argument("--split", choices=("train", "val", "test"), required=True)
    ffpp.add_argument("--split-json", required=True, type=Path)
    ffpp.add_argument("--compression", choices=("c23", "c40", "raw"), default="c23")
    ffpp.add_argument(
        "--methods",
        nargs="+",
        default=["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"],
    )
    celebdf = subparsers.add_parser("celebdf", help="preprocess the official Celeb-DF-v2 test list")
    _common_arguments(celebdf, default_frames=50)
    celebdf.add_argument("--test-list", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    if args.frames <= 0:
        raise ValueError("--frames must be positive")
    if args.dataset == "ffpp":
        preprocess_ffpp(args)
    else:
        preprocess_celebdf(args)


if __name__ == "__main__":
    main()
