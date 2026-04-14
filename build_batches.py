from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

from server_runtime import (
    build_vocab,
    ensure_output_dirs,
    load_config,
    load_manifest,
    resize_image_to_input,
    resize_mask_to_input,
    save_metrics,
    split_records,
)
from util import text_processing


def _load_rgb_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def _load_binary_mask(path: Path) -> np.ndarray:
    return (np.asarray(Image.open(path).convert("L")) > 127).astype(np.float32)


def _save_split_batches(config: Dict, records: List[Dict], split_name: str, vocab_dict: Dict[str, int]) -> int:
    dataset_root = Path(config["dataset"]["root"])
    caption_index = int(config["dataset"]["caption_index"])
    input_size = int(config["model"]["input_size"])
    num_steps = int(config["model"]["num_steps"])
    split_dir = Path(config["dataset"]["batch_dir"]) / f"{split_name}_batch"
    if split_dir.exists():
        shutil.rmtree(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)

    for index, record in enumerate(records):
        image = _load_rgb_image(dataset_root / record["image"])
        mask = _load_binary_mask(dataset_root / record["mask"])
        sentence = record["caption"][caption_index]
        text = text_processing.preprocess_sentence(sentence, vocab_dict, num_steps)

        if split_name == "train":
            image_to_save = resize_image_to_input(image, input_size)
            mask_to_save = resize_mask_to_input(mask, input_size)
        else:
            image_to_save = image
            mask_to_save = mask.astype(np.float32)

        np.savez(
            split_dir / f"plantseg_{split_name}_{index}.npz",
            text_batch=np.asarray(text, dtype=np.int64),
            im_batch=image_to_save,
            mask_batch=mask_to_save.astype(np.float32),
            sent_batch=np.asarray([sentence]),
            sample_id=np.asarray(record["id"]),
            mask_relpath=np.asarray(record["mask"]),
        )
    return len(records)


def build_plantseg_batches(config: Dict) -> Dict[str, int]:
    ensure_output_dirs(config)
    records = load_manifest(config)
    split_map = split_records(records)
    vocab_dict = build_vocab(split_map["train"], int(config["dataset"]["caption_index"]), config["dataset"]["vocab_path"])

    counts = {}
    for split_name in ("train", "val", "test"):
        counts[split_name] = _save_split_batches(config, split_map[split_name], split_name, vocab_dict)

    save_metrics(counts, Path(config["outputs"]["metrics_dir"]) / "batch_manifest.json")
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build plantseg batch files for RMI training.")
    parser.add_argument("--config", default="configs/plantseg_rmi_resnet.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    counts = build_plantseg_batches(config)
    for split_name, count in counts.items():
        print(f"{split_name}: {count} batches")


if __name__ == "__main__":
    main()
