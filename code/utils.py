import numpy as np
import torch
from pathlib import Path
from typing import Optional
from stable_pretraining import data as dt
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks import ModelCheckpoint

def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


def get_column_normalizer(dataset, source: str, target: str):
    """Get normalizer for a specific column in the dataset."""
    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()

    def norm_fn(x):
        return ((x - mean) / std).float()

    normalizer = dt.transforms.WrapTorchTransform(norm_fn, source=source, target=target)
    return normalizer

class ModelObjectCallBack(Callback):
    """Callback to save a clean inference object after each epoch."""

    def __init__(
        self,
        dirpath,
        filename="model_object",
        epoch_interval: int = 1,
        model_builder=None,
    ):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.epoch_interval = epoch_interval
        self.model_builder = model_builder

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        epoch_idx = trainer.current_epoch + 1
        output_path = self.dirpath / f"{self.filename}_epoch_{epoch_idx}_object.ckpt"

        if trainer.is_global_zero and epoch_idx % self.epoch_interval == 0:
            self._dump_model(pl_module.model, output_path)

    def _dump_model(self, model, path):
        clean_model = self.model_builder(model) if self.model_builder is not None else model
        torch.save(clean_model, path)


class LatestTrainerCheckpoint(ModelCheckpoint):
    """Persist a full trainer checkpoint to a fixed path after each epoch."""

    def __init__(self, path):
        path = Path(path)
        super().__init__(
            dirpath=str(path.parent),
            filename=path.with_suffix("").name,
            save_last=False,
            save_on_train_epoch_end=True,
            every_n_epochs=1,
            enable_version_counter=False,
        )


def resolve_resume_ckpt(run_dir: Path, output_model_name: str) -> Optional[Path]:
    """Return the explicit trainer checkpoint to resume from when available."""
    ckpt_path = Path(run_dir) / f"{output_model_name}_weights.ckpt"
    return ckpt_path if ckpt_path.exists() else None


def _extract_checkpoint_state_dict(payload):
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload["state_dict"]
    if isinstance(payload, dict):
        return payload
    if hasattr(payload, "state_dict"):
        return payload.state_dict()
    raise TypeError(f"Unsupported checkpoint payload type: {type(payload)!r}")


def _strip_known_prefixes(state_dict):
    prefixes = ("model.", "module.")
    normalized = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in prefixes:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        normalized[new_key] = value
    return normalized


def warm_start_model(
    model,
    ckpt_path,
    strict: bool = False,
):
    """Load model weights from an object checkpoint or trainer checkpoint.

    With ``strict=False``, only parameters whose names and shapes match are
    loaded. This is useful when introducing a small new module such as a gate,
    or when reusing the encoder across nearby variants.
    """
    ckpt_path = Path(ckpt_path)
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    incoming = _strip_known_prefixes(_extract_checkpoint_state_dict(payload))

    if strict:
        result = model.load_state_dict(incoming, strict=True)
        print(f"[WarmStart] loaded strict checkpoint from {ckpt_path}")
        return {
            "loaded": len(incoming),
            "missing": list(result.missing_keys),
            "unexpected": list(result.unexpected_keys),
            "mismatched": [],
        }

    current = model.state_dict()
    compatible = {}
    unexpected = []
    mismatched = []
    for key, value in incoming.items():
        if key not in current:
            unexpected.append(key)
            continue
        if tuple(current[key].shape) != tuple(value.shape):
            mismatched.append((key, tuple(value.shape), tuple(current[key].shape)))
            continue
        compatible[key] = value

    load_result = model.load_state_dict(compatible, strict=False)
    missing = list(load_result.missing_keys)

    print(
        "[WarmStart] "
        f"loaded {len(compatible)} tensors from {ckpt_path} "
        f"(missing={len(missing)}, unexpected={len(unexpected)}, mismatched={len(mismatched)})"
    )
    if missing:
        print(f"[WarmStart] sample missing keys: {missing[:10]}")
    if unexpected:
        print(f"[WarmStart] sample unexpected keys: {unexpected[:10]}")
    if mismatched:
        sample = [
            f"{key}: ckpt{src_shape} -> model{dst_shape}"
            for key, src_shape, dst_shape in mismatched[:10]
        ]
        print(f"[WarmStart] sample mismatched keys: {sample}")

    return {
        "loaded": len(compatible),
        "missing": missing,
        "unexpected": unexpected,
        "mismatched": mismatched,
    }
