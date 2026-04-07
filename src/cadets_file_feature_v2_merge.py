import argparse
import json
import shutil
from pathlib import Path
from typing import Dict


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_EXTRACTED_ROOT = REPO_ROOT / "artifacts" / "features"
DEFAULT_BASE_CLEANED_ROOT = REPO_ROOT / "artifacts" / "features_cleaned"
DEFAULT_BASE_MODEL_READY_ROOT = REPO_ROOT / "artifacts" / "features_model_ready"
DEFAULT_V2_EXTRACTED_ROOT = REPO_ROOT / "artifacts" / "features_file_only_v2"
DEFAULT_V2_CLEANED_ROOT = REPO_ROOT / "artifacts" / "features_cleaned_file_only_v2"
DEFAULT_V2_MODEL_READY_ROOT = REPO_ROOT / "artifacts" / "features_model_ready_file_only_v2"
DEFAULT_OUTPUT_EXTRACTED_ROOT = REPO_ROOT / "artifacts" / "features_hybrid_file_v2"
DEFAULT_OUTPUT_CLEANED_ROOT = REPO_ROOT / "artifacts" / "features_cleaned_hybrid_file_v2"
DEFAULT_OUTPUT_MODEL_READY_ROOT = REPO_ROOT / "artifacts" / "features_model_ready_hybrid_file_v2"

FILE_GROUPS = [
    "file_view__file_node",
    "process_view__file_node",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge file-only v2 feature groups into the existing full feature directories."
    )
    parser.add_argument("--base-extracted-root", type=str, default=str(DEFAULT_BASE_EXTRACTED_ROOT))
    parser.add_argument("--base-cleaned-root", type=str, default=str(DEFAULT_BASE_CLEANED_ROOT))
    parser.add_argument("--base-model-ready-root", type=str, default=str(DEFAULT_BASE_MODEL_READY_ROOT))
    parser.add_argument("--v2-extracted-root", type=str, default=str(DEFAULT_V2_EXTRACTED_ROOT))
    parser.add_argument("--v2-cleaned-root", type=str, default=str(DEFAULT_V2_CLEANED_ROOT))
    parser.add_argument("--v2-model-ready-root", type=str, default=str(DEFAULT_V2_MODEL_READY_ROOT))
    parser.add_argument("--output-extracted-root", type=str, default=str(DEFAULT_OUTPUT_EXTRACTED_ROOT))
    parser.add_argument("--output-cleaned-root", type=str, default=str(DEFAULT_OUTPUT_CLEANED_ROOT))
    parser.add_argument("--output-model-ready-root", type=str, default=str(DEFAULT_OUTPUT_MODEL_READY_ROOT))
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def replace_tree(base_root: Path, output_root: Path) -> None:
    if output_root.exists():
        shutil.rmtree(output_root)
    shutil.copytree(base_root, output_root)


def overlay_group_files(base_manifest: Dict[str, object], v2_root: Path, output_root: Path) -> None:
    for window in base_manifest["windows"]:
        window_name = window["window_name"]
        for group_name in FILE_GROUPS:
            filename = f"{group_name}.tsv"
            source_path = v2_root / window_name / filename
            target_path = output_root / window_name / filename
            if not source_path.exists():
                raise FileNotFoundError(f"Missing v2 group file: {source_path}")
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target_path)


def merge_cleaned_manifest(
    base_manifest: Dict[str, object],
    v2_manifest: Dict[str, object],
    output_extracted_root: Path,
) -> Dict[str, object]:
    merged = dict(base_manifest)
    merged["group_specs"] = dict(base_manifest["group_specs"])
    for group_name in FILE_GROUPS:
        merged["group_specs"][group_name] = v2_manifest["group_specs"][group_name]
    merged["feature_manifest_path"] = str(output_extracted_root / "feature_manifest.json")
    merged["notes"] = list(base_manifest.get("notes", [])) + [
        "file_view__file_node and process_view__file_node are replaced by file-only v2 outputs.",
    ]
    return merged


def merge_model_ready_manifest(
    base_manifest: Dict[str, object],
    v2_manifest: Dict[str, object],
    output_cleaned_root: Path,
) -> Dict[str, object]:
    merged = dict(base_manifest)
    merged["group_specs"] = dict(base_manifest["group_specs"])
    for group_name in FILE_GROUPS:
        merged["group_specs"][group_name] = v2_manifest["group_specs"][group_name]
    merged["cleaned_feature_manifest_path"] = str(output_cleaned_root / "feature_manifest.json")
    merged["notes"] = list(base_manifest.get("notes", [])) + [
        "file_view__file_node and process_view__file_node are replaced by file-only v2 model-ready outputs.",
    ]
    return merged


def merge_extracted_manifest(base_manifest: Dict[str, object]) -> Dict[str, object]:
    merged = dict(base_manifest)
    merged["notes"] = list(base_manifest.get("notes", [])) + [
        "file_view__file_node and process_view__file_node are replaced by file-only v2 extracted outputs.",
    ]
    return merged


def main() -> None:
    args = parse_args()

    base_extracted_root = Path(args.base_extracted_root).expanduser().resolve()
    base_cleaned_root = Path(args.base_cleaned_root).expanduser().resolve()
    base_model_ready_root = Path(args.base_model_ready_root).expanduser().resolve()
    v2_extracted_root = Path(args.v2_extracted_root).expanduser().resolve()
    v2_cleaned_root = Path(args.v2_cleaned_root).expanduser().resolve()
    v2_model_ready_root = Path(args.v2_model_ready_root).expanduser().resolve()
    output_extracted_root = Path(args.output_extracted_root).expanduser().resolve()
    output_cleaned_root = Path(args.output_cleaned_root).expanduser().resolve()
    output_model_ready_root = Path(args.output_model_ready_root).expanduser().resolve()

    base_extracted_manifest = load_json(base_extracted_root / "feature_manifest.json")
    base_cleaned_manifest = load_json(base_cleaned_root / "feature_manifest.json")
    base_model_ready_manifest = load_json(base_model_ready_root / "feature_manifest.json")
    v2_cleaned_manifest = load_json(v2_cleaned_root / "feature_manifest.json")
    v2_model_ready_manifest = load_json(v2_model_ready_root / "feature_manifest.json")

    replace_tree(base_extracted_root, output_extracted_root)
    replace_tree(base_cleaned_root, output_cleaned_root)
    replace_tree(base_model_ready_root, output_model_ready_root)

    overlay_group_files(base_extracted_manifest, v2_extracted_root, output_extracted_root)
    overlay_group_files(base_extracted_manifest, v2_cleaned_root, output_cleaned_root)
    overlay_group_files(base_extracted_manifest, v2_model_ready_root, output_model_ready_root)

    write_json(
        output_extracted_root / "feature_manifest.json",
        merge_extracted_manifest(base_extracted_manifest),
    )
    write_json(
        output_cleaned_root / "feature_manifest.json",
        merge_cleaned_manifest(base_cleaned_manifest, v2_cleaned_manifest, output_extracted_root),
    )
    write_json(
        output_model_ready_root / "feature_manifest.json",
        merge_model_ready_manifest(base_model_ready_manifest, v2_model_ready_manifest, output_cleaned_root),
    )

    print(f"[file-feature-v2-merge] wrote extracted hybrid -> {output_extracted_root}", flush=True)
    print(f"[file-feature-v2-merge] wrote cleaned hybrid -> {output_cleaned_root}", flush=True)
    print(f"[file-feature-v2-merge] wrote model-ready hybrid -> {output_model_ready_root}", flush=True)


if __name__ == "__main__":
    main()
