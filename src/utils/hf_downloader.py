"""Hugging Face download helpers for TSF-TPAMI2026.

This module keeps the data/model download logic out of ``main.py`` and the
legacy dataset loaders. It uses ``huggingface_hub.snapshot_download`` to rely on
Hugging Face Hub's retry/caching behavior, then materializes the selected
snapshot content into the local paths expected by the original TSF loaders.

The implementation is intentionally tolerant to several common Hub layouts:

1. already-expanded folders, e.g. ``datasets/UCI HAPT/HAPT_Dataset``;
2. dataset folders at repo root, e.g. ``HAPT`` or ``HAPT_Dataset``;
3. archives, e.g. ``HAPT.zip`` or ``UCI HAPT Dataset.zip``;
4. archives that contain an extra top-level directory.
"""

from __future__ import annotations

import logging
import os
import shutil
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

try:
    from huggingface_hub import snapshot_download
except ImportError as exc:  # pragma: no cover - exercised only when dependency is missing
    snapshot_download = None  # type: ignore[assignment]
    _HF_IMPORT_ERROR = exc
else:
    _HF_IMPORT_ERROR = None

LOGGER = logging.getLogger(__name__)

DEFAULT_DATASET_REPO = "crocodilegogogo/TSF-Datasets"
DEFAULT_MODEL_REPO = "crocodilegogogo/TSF-Models"
TSF_CLASSIFIER_NAME = "TSF_torch"
ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2")


@dataclass(frozen=True)
class DatasetSpec:
    """Local/HF layout description for one dataset."""

    name: str
    local_rel_path: str
    markers: tuple[str, ...]
    hf_candidates: tuple[str, ...]
    aliases: tuple[str, ...]
    allow_patterns: tuple[str, ...]


def _patterns_for(*names: str) -> tuple[str, ...]:
    """Build allow patterns for one dataset/model component.

    ``snapshot_download`` accepts glob-style patterns. We include exact folders,
    ``datasets/<name>``, and wildcard archives such as ``*HAPT*.zip`` so the
    downloader does not silently exclude compressed files whose names differ
    slightly from the local target directory.
    """
    patterns: list[str] = []
    for name in names:
        clean = name.strip("/")
        if not clean:
            continue
        last = clean.split("/")[-1]
        patterns.extend(
            [
                f"{clean}/**",
                f"datasets/{clean}/**",
                f"src/datasets/{clean}/**",
                f"{clean}.zip",
                f"{clean}.tar",
                f"{clean}.tar.gz",
                f"{clean}.tgz",
                f"*{last}*.zip",
                f"*{last}*.tar",
                f"*{last}*.tar.gz",
                f"*{last}*.tgz",
            ]
        )
        # For names with spaces, e.g. "UCI HAPT", wildcard matching is useful
        # because the upstream archive may be named "UCI HAPT Dataset.zip".
        if " " in clean:
            compact = clean.replace(" ", "*")
            patterns.extend(
                [
                    f"*{compact}*.zip",
                    f"*{compact}*.tar",
                    f"*{compact}*.tar.gz",
                    f"*{compact}*.tgz",
                ]
            )
    return tuple(dict.fromkeys(patterns))


DATASET_SPECS: dict[str, DatasetSpec] = {
    "HAPT": DatasetSpec(
        name="HAPT",
        local_rel_path="datasets/UCI HAPT/HAPT_Dataset",
        markers=("RawData/labels.txt", "activity_labels.txt"),
        hf_candidates=(
            "datasets/UCI HAPT/HAPT_Dataset",
            "UCI HAPT/HAPT_Dataset",
            "UCI HAPT",
            "HAPT_Dataset",
            "HAPT",
        ),
        aliases=("HAPT", "HAPT_Dataset", "UCI HAPT", "UCI HAPT Dataset"),
        allow_patterns=_patterns_for("HAPT", "HAPT_Dataset", "UCI HAPT", "UCI HAPT/HAPT_Dataset"),
    ),
    "Motion_Sense": DatasetSpec(
        name="Motion_Sense",
        local_rel_path="datasets/Motion-Sense",
        markers=("data_subjects_info.txt", "A_DeviceMotion_data"),
        hf_candidates=(
            "datasets/Motion-Sense",
            "Motion-Sense",
            "Motion_Sense",
            "MotionSense",
        ),
        aliases=("Motion-Sense", "Motion_Sense", "MotionSense"),
        allow_patterns=_patterns_for("Motion-Sense", "Motion_Sense", "MotionSense"),
    ),
    "SHL_2018": DatasetSpec(
        name="SHL_2018",
        local_rel_path="datasets/SHL2018",
        markers=("train", "test"),
        hf_candidates=("datasets/SHL2018", "SHL2018", "SHL_2018"),
        aliases=("SHL2018", "SHL_2018", "SHL"),
        allow_patterns=_patterns_for("SHL2018", "SHL_2018"),
    ),
    "HHAR": DatasetSpec(
        name="HHAR",
        local_rel_path="datasets/HHAR/Per_subject_npy",
        markers=(),
        hf_candidates=("datasets/HHAR/Per_subject_npy", "HHAR/Per_subject_npy", "Per_subject_npy", "HHAR"),
        aliases=("HHAR", "Per_subject_npy"),
        allow_patterns=_patterns_for("HHAR", "Per_subject_npy", "HHAR/Per_subject_npy"),
    ),
    "MobiAct": DatasetSpec(
        name="MobiAct",
        local_rel_path="datasets/MobiAct/Per_subject_no_NED_npy",
        markers=(),
        hf_candidates=(
            "datasets/MobiAct/Per_subject_no_NED_npy",
            "MobiAct/Per_subject_no_NED_npy",
            "Per_subject_no_NED_npy",
            "MobiAct",
        ),
        aliases=("MobiAct", "Per_subject_no_NED_npy"),
        allow_patterns=_patterns_for("MobiAct", "Per_subject_no_NED_npy", "MobiAct/Per_subject_no_NED_npy"),
    ),
    "Opportunity": DatasetSpec(
        name="Opportunity",
        local_rel_path="datasets/Opportunity",
        markers=("clean_opp.csv",),
        hf_candidates=("datasets/Opportunity", "Opportunity"),
        aliases=("Opportunity", "clean_opp"),
        allow_patterns=_patterns_for("Opportunity", "clean_opp"),
    ),
    "Pamap2": DatasetSpec(
        name="Pamap2",
        local_rel_path="datasets/Pamap2",
        markers=("clean_pamap.csv",),
        hf_candidates=("datasets/Pamap2", "Pamap2", "PAMAP2"),
        aliases=("Pamap2", "PAMAP2", "clean_pamap"),
        allow_patterns=_patterns_for("Pamap2", "PAMAP2", "clean_pamap"),
    ),
    "RealWorld": DatasetSpec(
        name="RealWorld",
        local_rel_path="datasets/RealWorld",
        markers=("Clean_Real_World.npy",),
        hf_candidates=("datasets/RealWorld", "RealWorld", "Real_World"),
        aliases=("RealWorld", "Real_World", "Clean_Real_World"),
        allow_patterns=_patterns_for("RealWorld", "Real_World", "Clean_Real_World"),
    ),
    "DSADS": DatasetSpec(
        name="DSADS",
        local_rel_path="datasets/DSADS",
        markers=("clean_DSADS.npy",),
        hf_candidates=("datasets/DSADS", "DSADS"),
        aliases=("DSADS", "clean_DSADS"),
        allow_patterns=_patterns_for("DSADS", "clean_DSADS"),
    ),
    "SHO": DatasetSpec(
        name="SHO",
        local_rel_path="datasets/SHO",
        markers=("Clean_SHO.npy",),
        hf_candidates=("datasets/SHO", "SHO"),
        aliases=("SHO", "Clean_SHO", "shoaib"),
        allow_patterns=_patterns_for("SHO", "Clean_SHO", "shoaib"),
    ),
}


def _require_hf_hub() -> None:
    if snapshot_download is None:
        raise RuntimeError(
            "huggingface_hub is required for automatic data/model downloads. "
            "Install it with `pip install huggingface_hub` or recreate the conda environment."
        ) from _HF_IMPORT_ERROR


def _as_path(path: str | os.PathLike[str]) -> Path:
    return Path(path).expanduser().resolve()


def _is_nonempty_dir(path: Path) -> bool:
    return path.is_dir() and any(path.iterdir())


def _has_markers(base: Path, markers: Sequence[str]) -> bool:
    if not base.exists():
        return False
    if not markers:
        return _is_nonempty_dir(base)
    return all((base / marker).exists() for marker in markers)


def _safe_remove_empty_dir(path: Path) -> None:
    if path.is_dir() and not any(path.iterdir()):
        path.rmdir()


def _copy_dir_contents(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _is_archive(path: Path) -> bool:
    lower = path.name.lower()
    return any(lower.endswith(suffix) for suffix in ARCHIVE_SUFFIXES)


def _extract_archive(archive_path: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(dst)
        return
    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path) as tf:
            tf.extractall(dst)
        return
    raise ValueError(f"Unsupported archive type: {archive_path}")


def _find_first_existing_dir(snapshot_dir: Path, candidates: Iterable[str]) -> Optional[Path]:
    for candidate in candidates:
        path = snapshot_dir / candidate
        if path.is_dir() and any(path.iterdir()):
            return path
    return None


def _find_dir_with_markers(root: Path, markers: Sequence[str]) -> Optional[Path]:
    """Find a nested directory satisfying all marker files/dirs."""
    if not markers:
        return None
    if _has_markers(root, markers):
        return root
    for path in root.rglob("*"):
        if path.is_dir() and _has_markers(path, markers):
            return path
    return None


def _norm_name(value: str) -> str:
    return "".join(ch.lower() for ch in value if ch.isalnum())


def _matches_alias(path: Path, aliases: Sequence[str]) -> bool:
    normalized = _norm_name(path.name)
    return any(_norm_name(alias) in normalized for alias in aliases)


def _find_dir_by_alias(root: Path, aliases: Sequence[str]) -> Optional[Path]:
    # Prefer deeper concrete dirs with content; this handles archives that unzip to
    # ``UCI HAPT/HAPT_Dataset`` rather than returning ``UCI HAPT`` too early.
    dirs = [p for p in root.rglob("*") if p.is_dir() and any(p.iterdir())]
    dirs.sort(key=lambda p: len(p.parts), reverse=True)
    for path in dirs:
        if _matches_alias(path, aliases):
            return path
    return None


def _find_single_useful_dir(root: Path) -> Optional[Path]:
    dirs = [p for p in root.iterdir() if p.is_dir() and any(p.iterdir())]
    if len(dirs) == 1:
        return dirs[0]
    return None


def _find_first_archive(root: Path, aliases: Iterable[str]) -> Optional[Path]:
    alias_tuple = tuple(aliases)
    archives = [p for p in root.rglob("*") if p.is_file() and _is_archive(p)]
    for archive in archives:
        if _matches_alias(archive, alias_tuple):
            return archive
    if len(archives) == 1:
        return archives[0]
    return None


def _find_marker_file(root: Path, marker: str) -> Optional[Path]:
    marker_name = Path(marker).name
    for path in root.rglob(marker_name):
        if path.is_file():
            return path
    return None


def _resolve_materialization_source(root: Path, spec: DatasetSpec) -> tuple[Optional[Path], Optional[Path]]:
    """Return ``(source_dir, source_file)`` for a downloaded/extracted snapshot."""
    source_dir = _find_first_existing_dir(root, spec.hf_candidates)
    if source_dir is not None:
        return source_dir, None

    source_dir = _find_dir_with_markers(root, spec.markers)
    if source_dir is not None:
        return source_dir, None

    source_dir = _find_dir_by_alias(root, spec.aliases)
    if source_dir is not None:
        return source_dir, None

    # Some processed datasets are a single file at repo root, e.g. clean_opp.csv.
    if len(spec.markers) == 1 and "/" not in spec.markers[0]:
        marker_file = _find_marker_file(root, spec.markers[0])
        if marker_file is not None:
            return None, marker_file

    source_dir = _find_single_useful_dir(root)
    if source_dir is not None:
        return source_dir, None

    return None, None


def _snapshot_download(
    *,
    repo_id: str,
    repo_type: str,
    revision: Optional[str],
    cache_dir: Optional[str],
    token: Optional[str],
    allow_patterns: Sequence[str],
) -> Path:
    _require_hf_hub()
    LOGGER.info("Downloading from Hugging Face repo %s (%s)...", repo_id, repo_type)
    try:
        snapshot_path = snapshot_download(  # type: ignore[misc]
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            allow_patterns=list(allow_patterns) if allow_patterns else None,
        )
    except Exception as exc:  # pragma: no cover - requires network failure path
        raise RuntimeError(
            f"Failed to download from Hugging Face repo '{repo_id}'. "
            "Check network access, repo permissions, HF_TOKEN, and the requested revision."
        ) from exc
    return Path(snapshot_path)


def _all_archive_patterns(aliases: Sequence[str]) -> tuple[str, ...]:
    patterns: list[str] = []
    for alias in aliases:
        clean = alias.strip("/")
        if not clean:
            continue
        patterns.extend(
            [
                f"*{clean}*.zip",
                f"*{clean}*.tar",
                f"*{clean}*.tar.gz",
                f"*{clean}*.tgz",
                f"**/*{clean}*.zip",
                f"**/*{clean}*.tar",
                f"**/*{clean}*.tar.gz",
                f"**/*{clean}*.tgz",
            ]
        )
    return tuple(dict.fromkeys(patterns))


def _dataset_allow_patterns(spec: DatasetSpec) -> tuple[str, ...]:
    # Include exact marker files as a safety net for processed single-file datasets.
    marker_patterns = tuple(f"**/{Path(marker).name}" for marker in spec.markers)
    return tuple(dict.fromkeys(spec.allow_patterns + _all_archive_patterns(spec.aliases) + marker_patterns))


def ensure_dataset_available(
    dataset_name: str,
    project_root: str | os.PathLike[str],
    *,
    repo_id: str = DEFAULT_DATASET_REPO,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
    force: bool = False,
) -> Path:
    """Ensure one dataset exists in the local path expected by TSF loaders."""
    spec = DATASET_SPECS.get(dataset_name)
    if spec is None:
        known = ", ".join(sorted(DATASET_SPECS))
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Known datasets: {known}")

    root = _as_path(project_root)
    destination = root / spec.local_rel_path

    if not force and _has_markers(destination, spec.markers):
        LOGGER.info("Dataset %s already exists at %s", dataset_name, destination)
        return destination

    LOGGER.info("Dataset %s is missing or incomplete at %s", dataset_name, destination)
    LOGGER.info("Fetching dataset %s from Hugging Face repo %s", dataset_name, repo_id)

    # Important for Windows users: huggingface_hub defaults to the user cache
    # directory, e.g. C:\Users\<name>\.cache\huggingface.  For this project,
    # keep the Hub cache under the project datasets folder unless the user
    # explicitly provides --hf-cache-dir.  The final materialized dataset is
    # still copied/extracted to ``destination`` below.
    effective_cache_dir = cache_dir or str(root / "datasets" / ".hf_cache")
    Path(effective_cache_dir).mkdir(parents=True, exist_ok=True)
    LOGGER.info("Using Hugging Face dataset cache at %s", effective_cache_dir)

    snapshot_dir = _snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        cache_dir=effective_cache_dir,
        token=token,
        allow_patterns=_dataset_allow_patterns(spec),
    )

    source_dir, source_file = _resolve_materialization_source(snapshot_dir, spec)

    if source_dir is None and source_file is None:
        archive = _find_first_archive(snapshot_dir, spec.aliases + spec.hf_candidates + (dataset_name,))
        if archive is None:
            snapshot_items = sorted(str(p.relative_to(snapshot_dir)) for p in snapshot_dir.rglob("*"))[:50]
            raise FileNotFoundError(
                f"Downloaded snapshot for dataset '{dataset_name}', but could not find any expected directory/archive/file. "
                f"Expected candidates: {spec.hf_candidates}. Snapshot path: {snapshot_dir}. "
                f"First snapshot items: {snapshot_items}"
            )
        LOGGER.info("Extracting dataset archive %s", archive.name)
        with tempfile.TemporaryDirectory(prefix=f"tsf_{dataset_name}_") as tmp:
            tmp_path = Path(tmp)
            _extract_archive(archive, tmp_path)
            source_dir, source_file = _resolve_materialization_source(tmp_path, spec)
            if source_dir is None and source_file is None:
                source_dir = _find_single_useful_dir(tmp_path) or tmp_path
            _safe_remove_empty_dir(destination)
            if source_file is not None:
                _copy_file(source_file, destination / source_file.name)
            else:
                _copy_dir_contents(source_dir, destination)  # type: ignore[arg-type]
    else:
        _safe_remove_empty_dir(destination)
        if source_file is not None:
            _copy_file(source_file, destination / source_file.name)
        else:
            _copy_dir_contents(source_dir, destination)  # type: ignore[arg-type]

    if not _has_markers(destination, spec.markers):
        marker_msg = ", ".join(spec.markers) if spec.markers else "non-empty directory"
        local_items = sorted(str(p.relative_to(destination)) for p in destination.rglob("*"))[:50] if destination.exists() else []
        raise FileNotFoundError(
            f"Dataset '{dataset_name}' was downloaded, but local path still does not satisfy expected marker(s): "
            f"{marker_msg}. Local path: {destination}. First local items: {local_items}. "
            "Please check whether the Hugging Face repo stores the processed dataset in the layout expected by the TSF loader."
        )

    LOGGER.info("Dataset %s is ready at %s", dataset_name, destination)
    return destination


def _model_allow_patterns(dataset_name: str, classifier_name: str) -> tuple[str, ...]:
    aliases = (dataset_name, classifier_name, f"{dataset_name}_{classifier_name}", f"{dataset_name}-{classifier_name}")
    return tuple(
        dict.fromkeys(
            [
                f"saved_models/{dataset_name}/{classifier_name}/**",
                f"src/saved_models/{dataset_name}/{classifier_name}/**",
                f"{dataset_name}/{classifier_name}/**",
                f"{classifier_name}/{dataset_name}/**",
                f"{dataset_name}/**",
                f"**/{dataset_name}/**",
                f"**/best_validation_model.pkl",
                *_all_archive_patterns(aliases),
            ]
        )
    )


def _model_candidates(dataset_name: str, classifier_name: str) -> tuple[str, ...]:
    return (
        f"saved_models/{dataset_name}/{classifier_name}",
        f"src/saved_models/{dataset_name}/{classifier_name}",
        f"{dataset_name}/{classifier_name}",
        f"{classifier_name}/{dataset_name}",
        dataset_name,
    )


def _contains_tsf_weights(path: Path) -> bool:
    return path.exists() and any(path.rglob("best_validation_model.pkl"))


def _find_model_source_dir(root: Path, dataset_name: str, classifier_name: str) -> Optional[Path]:
    source_dir = _find_first_existing_dir(root, _model_candidates(dataset_name, classifier_name))
    if source_dir is not None and _contains_tsf_weights(source_dir):
        return source_dir
    for path in root.rglob("*"):
        if path.is_dir() and _contains_tsf_weights(path):
            # Prefer directories that contain the classifier or dataset name.
            text = str(path).lower()
            if dataset_name.lower() in text or classifier_name.lower() in text:
                return path
    if _contains_tsf_weights(root):
        return root
    return None


def ensure_tsf_model_weights(
    dataset_name: str,
    classifier_name: str,
    model_dir: str | os.PathLike[str],
    *,
    repo_id: str = DEFAULT_MODEL_REPO,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
    force: bool = False,
) -> Path:
    """Ensure pretrained TSF weights exist under ``src/saved_models/<dataset>/TSF_torch``."""
    destination = _as_path(model_dir)

    if classifier_name != TSF_CLASSIFIER_NAME:
        LOGGER.info(
            "Skipping Hugging Face model download for classifier %s. Only %s is managed automatically.",
            classifier_name,
            TSF_CLASSIFIER_NAME,
        )
        return destination

    if not force and _contains_tsf_weights(destination):
        LOGGER.info("TSF model weights for %s already exist at %s", dataset_name, destination)
        return destination

    LOGGER.info("TSF model weights for %s are missing at %s", dataset_name, destination)
    LOGGER.info("Fetching TSF weights from Hugging Face repo %s", repo_id)

    # Keep model Hub cache inside the project as well instead of using the
    # platform default user cache.  This avoids silently filling C: on Windows
    # and is friendlier to shared Linux servers with project-specific storage.
    saved_models_root = destination
    for parent in destination.parents:
        if parent.name == "saved_models":
            saved_models_root = parent
            break
    effective_cache_dir = cache_dir or str(saved_models_root / ".hf_cache")
    Path(effective_cache_dir).mkdir(parents=True, exist_ok=True)
    LOGGER.info("Using Hugging Face model cache at %s", effective_cache_dir)

    snapshot_dir = _snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        cache_dir=effective_cache_dir,
        token=token,
        allow_patterns=_model_allow_patterns(dataset_name, classifier_name),
    )

    source_dir = _find_model_source_dir(snapshot_dir, dataset_name, classifier_name)
    if source_dir is not None:
        _safe_remove_empty_dir(destination)
        _copy_dir_contents(source_dir, destination)
    else:
        archive = _find_first_archive(snapshot_dir, (f"{dataset_name}_{classifier_name}", f"{dataset_name}-{classifier_name}", dataset_name, classifier_name))
        if archive is None:
            snapshot_items = sorted(str(p.relative_to(snapshot_dir)) for p in snapshot_dir.rglob("*"))[:50]
            raise FileNotFoundError(
                f"Downloaded model snapshot, but could not locate TSF weights for dataset '{dataset_name}'. "
                f"Expected directories: {_model_candidates(dataset_name, classifier_name)}. Snapshot path: {snapshot_dir}. "
                f"First snapshot items: {snapshot_items}"
            )
        LOGGER.info("Extracting model archive %s", archive.name)
        with tempfile.TemporaryDirectory(prefix=f"tsf_model_{dataset_name}_") as tmp:
            tmp_path = Path(tmp)
            _extract_archive(archive, tmp_path)
            source_dir = _find_model_source_dir(tmp_path, dataset_name, classifier_name)
            if source_dir is None:
                source_dir = tmp_path
            _safe_remove_empty_dir(destination)
            _copy_dir_contents(source_dir, destination)

    if not _contains_tsf_weights(destination):
        local_items = sorted(str(p.relative_to(destination)) for p in destination.rglob("*"))[:50] if destination.exists() else []
        raise FileNotFoundError(
            f"Model weights were downloaded, but no best_validation_model.pkl was found under {destination}. "
            f"First local items: {local_items}. Please check the Hugging Face model repository layout."
        )

    LOGGER.info("TSF model weights for %s are ready at %s", dataset_name, destination)
    return destination
