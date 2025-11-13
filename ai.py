"""Speaker diarization entrypoint for cached pyannote pipelines."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import torchaudio

from cli_utils import add_common_io_arguments, expand_path, first_env, path_argument
from pyannote_utils import build_diarization_pipeline


SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}


def get_audio_duration(audio_path: Path) -> float:
	info = torchaudio.info(str(audio_path))
	return info.num_frames / info.sample_rate


def discover_audio_files(audio_dir: Path, skip_prefix: str) -> List[Path]:
	return [
		file_path
		for file_path in audio_dir.rglob("*")
		if file_path.is_file()
		and file_path.suffix.lower() in SUPPORTED_EXTENSIONS
		and not (skip_prefix and file_path.name.startswith(skip_prefix))
	]


def diarize_folder(
	*,
	audio_dir: Path,
	output_dir: Path,
	evaluation_id: str,
	cache_dir: Path,
	segmentation_model_id: str,
	embedding_model_id: str,
	cluster_threshold: float,
	min_cluster_size: int,
	skip_prefix: str,
):
	print("🚀 Initializing pyannote speaker diarization pipeline")
	pipeline = build_diarization_pipeline(
		cache_dir=cache_dir,
		segmentation_model_id=segmentation_model_id,
		embedding_model_id=embedding_model_id,
		cluster_threshold=cluster_threshold,
		min_cluster_size=min_cluster_size,
	)

	audio_files = discover_audio_files(audio_dir, skip_prefix)
	if not audio_files:
		print(
			f"🚫 No audio files found in '{audio_dir}' (extensions={sorted(SUPPORTED_EXTENSIONS)}, skip prefix={skip_prefix!r})."
		)
		return

	print(f"🎧 Found {len(audio_files)} audio file(s). Starting diarization...\n")
	diarization_rows: List[dict] = []

	for index, audio_file in enumerate(sorted(audio_files), 1):
		print(f"[{index}/{len(audio_files)}] Processing {audio_file.name}")
		try:
			audio_duration = get_audio_duration(audio_file)
		except Exception as exc:  # noqa: BLE001
			print(f"  Error reading {audio_file.name}: {exc}")
			continue

		try:
			diarization = pipeline(str(audio_file))
		except Exception as exc:  # noqa: BLE001
			print(f"  Error diarizing {audio_file.name}: {exc}")
			continue

		segments_added = 0
		for segment, _, speaker_label in diarization.itertracks(yield_label=True):
			start_time = max(0.0, segment.start)
			end_time = min(audio_duration, segment.end)
			if end_time <= start_time:
				continue

			duration = end_time - start_time
			diarization_rows.append(
				{
					"AudioFileName": audio_file.name,
					"Speaker": str(speaker_label),
					"StartTS": round(start_time, 3),
					"EndTS": round(end_time, 3),
					"Duration": round(duration, 3),
				}
			)
			segments_added += 1

		print(f"  ✓ {segments_added} segment(s) captured\n")

	if not diarization_rows:
		print("🤷 No speaker segments detected across provided audio files.")
		return

	output_dir.mkdir(parents=True, exist_ok=True)
	output_path = output_dir / f"SD_{evaluation_id}.csv"
	pd.DataFrame(diarization_rows).to_csv(output_path, index=False, encoding="utf-8")
	print(f"🎉 Diarization complete! Results saved to: {output_path}")


def parse_arguments(argv: Iterable[str] | None = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Speaker diarization using cached pyannote models")
	add_common_io_arguments(
		parser,
		defaults={
			"audio_dir": "ps6_04",
			"output_dir": "output",
			"evaluation_id": "04",
		},
		audio_env=("SD_AUDIO_DIR", "PIPELINE_AUDIO_DIR", "AUDIO_DIR"),
		output_env=("SD_OUTPUT_DIR", "OUTPUT_DIR"),
		eval_env=("SD_EVALUATION_ID", "PIPELINE_EVALUATION_ID", "EVALUATION_ID"),
	)

	cache_default = first_env(("SD_PYANNOTE_CACHE", "PYANNOTE_CACHE"))
	parser.add_argument(
		"--pyannote-cache",
		type=path_argument,
		default=path_argument(cache_default) if cache_default else expand_path("~/.cache/torch/pyannote"),
		help="Directory containing cached pyannote models",
	)
	parser.add_argument(
		"--segmentation-model",
		default=first_env(("SD_SEGMENTATION_MODEL",), default="pyannote/segmentation-3.0"),
		help="Segmentation model identifier",
	)
	parser.add_argument(
		"--embedding-model",
		default=first_env(("SD_EMBEDDING_MODEL",), default="pyannote/wespeaker-voxceleb-resnet34-LM"),
		help="Embedding model identifier",
	)
	parser.add_argument(
		"--cluster-threshold",
		type=float,
		default=float(first_env(("SD_CLUSTER_THRESHOLD",), default="0.7045654963945799")),
		help="Agglomerative clustering threshold",
	)
	parser.add_argument(
		"--min-cluster-size",
		type=int,
		default=int(first_env(("SD_MIN_CLUSTER_SIZE",), default="12")),
		help="Minimum number of frames per cluster",
	)
	parser.add_argument(
		"--skip-prefix",
		default=first_env(("SD_SKIP_PREFIX", "PIPELINE_SKIP_PREFIX"), default="ID"),
		help="Skip audio files whose filename starts with this prefix",
	)

	args = parser.parse_args(argv)

	if args.audio_dir is None or not Path(args.audio_dir).exists():
		raise FileNotFoundError(f"Audio directory not found: {args.audio_dir}")
	if args.output_dir is None:
		raise ValueError("--output-dir must be provided")
	if not args.evaluation_id:
		raise ValueError("--evaluation-id must be provided")

	return args


if __name__ == "__main__":
	namespace = parse_arguments()
	diarize_folder(
		audio_dir=namespace.audio_dir,
		output_dir=namespace.output_dir,
		evaluation_id=namespace.evaluation_id,
		cache_dir=namespace.pyannote_cache,
		segmentation_model_id=namespace.segmentation_model,
		embedding_model_id=namespace.embedding_model,
		cluster_threshold=namespace.cluster_threshold,
		min_cluster_size=namespace.min_cluster_size,
		skip_prefix=namespace.skip_prefix,
	)
