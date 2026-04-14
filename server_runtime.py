from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset

from util import text_processing


SPECIAL_TOKENS = [text_processing.PAD_IDENTIFIER, "<go>", text_processing.EOS_IDENTIFIER, text_processing.UNK_IDENTIFIER]


def repo_root() -> Path:
    return Path(__file__).resolve().parent


def load_config(config_path: str | Path) -> Dict:
    config_path = Path(config_path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    root = repo_root()
    config["_repo_root"] = root

    def resolve(path_value: str) -> str:
        path = Path(path_value)
        return str((root / path).resolve()) if not path.is_absolute() else str(path.resolve())

    config["dataset"]["root"] = resolve(config["dataset"]["root"])
    config["dataset"]["manifest"] = resolve(config["dataset"]["manifest"])
    config["dataset"]["batch_dir"] = resolve(config["dataset"]["batch_dir"])
    config["dataset"]["vocab_path"] = resolve(config["dataset"]["vocab_path"])
    config["outputs"]["root"] = resolve(config["outputs"]["root"])
    config["outputs"]["checkpoint_best"] = resolve(config["outputs"]["checkpoint_best"])
    config["outputs"]["checkpoint_last"] = resolve(config["outputs"]["checkpoint_last"])
    config["outputs"]["metrics_dir"] = resolve(config["outputs"]["metrics_dir"])
    config["outputs"]["test_mask_dir"] = resolve(config["outputs"]["test_mask_dir"])
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def ensure_output_dirs(config: Dict) -> None:
    Path(config["dataset"]["batch_dir"]).mkdir(parents=True, exist_ok=True)
    ensure_parent(config["dataset"]["vocab_path"])
    Path(config["outputs"]["root"]).mkdir(parents=True, exist_ok=True)
    Path(config["outputs"]["metrics_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["outputs"]["test_mask_dir"]).mkdir(parents=True, exist_ok=True)


def load_manifest(config: Dict) -> List[Dict]:
    manifest_path = Path(config["dataset"]["manifest"])
    with manifest_path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    return records


def split_records(records: Iterable[Dict]) -> Dict[str, List[Dict]]:
    split_map: Dict[str, List[Dict]] = {"train": [], "val": [], "test": []}
    for record in records:
        split_name = record["split"]
        if split_name not in split_map:
            raise ValueError(f"unsupported split: {split_name}")
        split_map[split_name].append(record)
    return split_map


def tokenize_caption(text: str) -> List[str]:
    words = text_processing.SENTENCE_SPLIT_REGEX.split(text.strip())
    words = [word.lower() for word in words if len(word.strip()) > 0]
    if words and words[-1] == ".":
        words = words[:-1]
    return words


def build_vocab(records: Iterable[Dict], caption_index: int, vocab_path: str | Path) -> Dict[str, int]:
    counter: Counter = Counter()
    for record in records:
        counter.update(tokenize_caption(record["caption"][caption_index]))

    sorted_tokens = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    vocab_words = SPECIAL_TOKENS + [token for token, _ in sorted_tokens if token not in SPECIAL_TOKENS]

    vocab_path = Path(vocab_path)
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    with vocab_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(vocab_words) + "\n")
    return {token: idx for idx, token in enumerate(vocab_words)}


def load_vocab(vocab_path: str | Path) -> Dict[str, int]:
    return text_processing.load_vocab_dict_from_file(str(vocab_path))


def normalize_image(image: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
    return (tensor - mean) / std


def _resize_with_pil(array: np.ndarray, size: tuple[int, int], mode: int) -> np.ndarray:
    image = Image.fromarray(array)
    resized = image.resize(size, resample=mode)
    return np.asarray(resized)


def resize_image_to_input(image: np.ndarray, input_size: int) -> np.ndarray:
    image_height, image_width = image.shape[:2]
    scale = min(input_size / image_height, input_size / image_width)
    resized_height = max(int(round(image_height * scale)), 1)
    resized_width = max(int(round(image_width * scale)), 1)
    resized = _resize_with_pil(image.astype(np.uint8), (resized_width, resized_height), Image.BILINEAR)
    canvas = np.zeros((input_size, input_size, image.shape[2]), dtype=np.uint8)
    pad_top = (input_size - resized_height) // 2
    pad_left = (input_size - resized_width) // 2
    canvas[pad_top : pad_top + resized_height, pad_left : pad_left + resized_width, :] = resized
    return canvas


def resize_mask_to_input(mask: np.ndarray, input_size: int) -> np.ndarray:
    mask_uint8 = (mask > 0.5).astype(np.uint8) * 255
    mask_height, mask_width = mask_uint8.shape[:2]
    scale = min(input_size / mask_height, input_size / mask_width)
    resized_height = max(int(round(mask_height * scale)), 1)
    resized_width = max(int(round(mask_width * scale)), 1)
    resized = _resize_with_pil(mask_uint8, (resized_width, resized_height), Image.NEAREST)
    canvas = np.zeros((input_size, input_size), dtype=np.uint8)
    pad_top = (input_size - resized_height) // 2
    pad_left = (input_size - resized_width) // 2
    canvas[pad_top : pad_top + resized_height, pad_left : pad_left + resized_width] = resized
    return (canvas > 127).astype(np.float32)


def resize_prediction_to_mask(prediction: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    pred_uint8 = prediction.astype(np.uint8) * 255
    pred_height, pred_width = pred_uint8.shape[:2]
    target_height, target_width = target_shape
    scale = max(target_height / pred_height, target_width / pred_width)
    resized_height = max(int(round(pred_height * scale)), 1)
    resized_width = max(int(round(pred_width * scale)), 1)
    resized = _resize_with_pil(pred_uint8, (resized_width, resized_height), Image.NEAREST)
    crop_top = max((resized_height - target_height) // 2, 0)
    crop_left = max((resized_width - target_width) // 2, 0)
    cropped = resized[crop_top : crop_top + target_height, crop_left : crop_left + target_width]
    return (cropped > 127).astype(np.uint8)


class NpzBatchDataset(Dataset):
    def __init__(self, config: Dict, split: str):
        self.config = config
        self.split = split
        self.input_size = int(config["model"]["input_size"])
        batch_dir = Path(config["dataset"]["batch_dir"]) / f"{split}_batch"
        self.files = sorted(batch_dir.glob("*.npz"))
        if not self.files:
            raise RuntimeError(f"no npz files under {batch_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Dict:
        with np.load(self.files[index], allow_pickle=True) as data:
            image = data["im_batch"]
            mask = data["mask_batch"].astype(np.float32)
            text = data["text_batch"].astype(np.int64)
            sample_id = str(data["sample_id"].item())
            mask_relpath = str(data["mask_relpath"].item())
            sentence = str(data["sent_batch"][0])

        if self.split == "train":
            model_image = image
            train_mask = mask
        else:
            model_image = resize_image_to_input(image, self.input_size)
            train_mask = resize_mask_to_input(mask, self.input_size)

        return {
            "text": torch.from_numpy(text),
            "image": normalize_image(model_image),
            "train_mask": torch.from_numpy(train_mask).unsqueeze(0),
            "gt_mask": torch.from_numpy((mask > 0.5).astype(np.float32)),
            "sample_id": sample_id,
            "mask_relpath": mask_relpath,
            "sentence": sentence,
            "orig_size": tuple(mask.shape[:2]),
        }


class RunningMaskMetrics:
    def __init__(self) -> None:
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    def update(self, pred: np.ndarray, target: np.ndarray) -> None:
        pred_bool = pred.astype(bool)
        target_bool = target.astype(bool)
        self.tp += int(np.logical_and(pred_bool, target_bool).sum())
        self.fp += int(np.logical_and(pred_bool, np.logical_not(target_bool)).sum())
        self.fn += int(np.logical_and(np.logical_not(pred_bool), target_bool).sum())
        self.tn += int(np.logical_and(np.logical_not(pred_bool), np.logical_not(target_bool)).sum())

    def compute(self) -> Dict[str, float]:
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        fg_iou = tp / max(tp + fp + fn, 1)
        bg_iou = tn / max(tn + fp + fn, 1)
        fg_acc = tp / max(tp + fn, 1)
        bg_acc = tn / max(tn + fp, 1)
        dice = (2 * tp) / max(2 * tp + fp + fn, 1)
        return {
            "IoU": fg_iou,
            "Dice": dice,
            "Recall": fg_acc,
            "mIoU": (fg_iou + bg_iou) / 2.0,
            "mACC": (fg_acc + bg_acc) / 2.0,
        }


def save_metrics(metrics: Dict[str, float], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


def save_prediction_mask(pred_mask: np.ndarray, output_dir: str | Path, mask_relpath: str) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / Path(mask_relpath).name
    Image.fromarray((pred_mask.astype(np.uint8) * 255), mode="L").save(output_path)
    return output_path


def copy_config_snapshot(config: Dict, source_path: str | Path) -> Path:
    snapshot_path = Path(config["outputs"]["root"]) / "resolved_config.yaml"
    with Path(source_path).open("r", encoding="utf-8") as handle:
        snapshot_path.write_text(handle.read(), encoding="utf-8")
    return snapshot_path
