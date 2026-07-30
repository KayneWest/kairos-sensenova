from __future__ import annotations

import json
import os
import runpy
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class KairosFeatureAccessAudit:
    repo_root: str
    config_file: str
    pipeline_type: str | None
    vae_path: str | None
    vae_path_exists: bool
    pretrained_dit_path: str | None
    pretrained_dit_path_exists: bool
    text_encoder_path: str | None
    text_encoder_path_exists: bool
    require_clip_embedding: bool | None
    require_vae_embedding: bool | None
    fuse_vae_embedding_in_latents: bool | None
    has_image_input: bool | None
    candidate_feature_modes: list[str]
    blockers: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def audit_kairos_feature_access(
    config_file: str | Path = "kairos/configs/kairos_4b_config_DMD.py",
    repo_root: str | Path | None = None,
) -> KairosFeatureAccessAudit:
    repo = Path(repo_root or Path(__file__).resolve().parents[3]).resolve()
    config_path = _resolve_path(config_file, repo)
    _ensure_repo_imports(repo)

    cfg = _load_mmengine_config(config_path)
    pipeline = dict(cfg.get("pipeline", {}))
    pipeline_args = dict(pipeline.get("pipeline_args", {}))
    dit_config = dict(pipeline_args.get("dit_config", {}))

    vae_path = _resolve_optional_model_path(pipeline_args.get("vae_path") or cfg.get("vae_path"), repo)
    dit_path = _resolve_optional_model_path(pipeline.get("pretrained_dit") or cfg.get("pretrained_dit"), repo)
    text_encoder_path = _resolve_optional_model_path(
        pipeline_args.get("text_encoder_path") or cfg.get("text_encoder_path"),
        repo,
    )

    candidate_modes = []
    blockers = []

    if vae_path and vae_path.exists():
        candidate_modes.append("kairos_vae_first_frame_latent")
    else:
        blockers.append("Kairos/Wan VAE checkpoint is missing.")

    if bool(dit_config.get("require_clip_embedding")):
        candidate_modes.append("kairos_image_clip_feature")

    if text_encoder_path and text_encoder_path.exists():
        candidate_modes.append("qwen_vl_prompt_image_embedding")
    else:
        blockers.append("Qwen-VL text/image encoder path is missing.")

    if dit_path and dit_path.exists():
        candidate_modes.append("kairos_dit_hidden_state_future_work")
    else:
        blockers.append("Kairos DiT checkpoint is missing.")

    if not bool(dit_config.get("fuse_vae_embedding_in_latents")):
        blockers.append("Configured DiT does not fuse first-frame VAE latents into denoising latents.")

    return KairosFeatureAccessAudit(
        repo_root=str(repo),
        config_file=str(config_path),
        pipeline_type=pipeline.get("pipeline_type"),
        vae_path=str(vae_path) if vae_path else None,
        vae_path_exists=bool(vae_path and vae_path.exists()),
        pretrained_dit_path=str(dit_path) if dit_path else None,
        pretrained_dit_path_exists=bool(dit_path and dit_path.exists()),
        text_encoder_path=str(text_encoder_path) if text_encoder_path else None,
        text_encoder_path_exists=bool(text_encoder_path and text_encoder_path.exists()),
        require_clip_embedding=_optional_bool(dit_config.get("require_clip_embedding")),
        require_vae_embedding=_optional_bool(dit_config.get("require_vae_embedding")),
        fuse_vae_embedding_in_latents=_optional_bool(dit_config.get("fuse_vae_embedding_in_latents")),
        has_image_input=_optional_bool(dit_config.get("has_image_input")),
        candidate_feature_modes=candidate_modes,
        blockers=blockers,
    )


class KairosVAEFeatureExtractor:
    """
    Extracts a real Kairos/Wan VAE first-frame latent from an RGB observation.

    This is the first concrete path toward "Kairos/Sensenova participates in
    control": the policy can consume features from a native component used by
    the Kairos pipeline rather than the repo's toy drone-game world model.
    """

    def __init__(
        self,
        config_file: str | Path = "kairos/configs/kairos_4b_config_DMD.py",
        repo_root: str | Path | None = None,
        device: str = "cuda",
        dtype: str = "bfloat16",
        height: int = 480,
        width: int = 832,
        tiled: bool = True,
        tile_size: tuple[int, int] = (30, 52),
        tile_stride: tuple[int, int] = (15, 26),
    ):
        self.audit = audit_kairos_feature_access(config_file=config_file, repo_root=repo_root)
        self.repo_root = Path(self.audit.repo_root)
        self.device = device
        self.dtype_name = dtype
        self.height = int(height)
        self.width = int(width)
        self.tiled = bool(tiled)
        self.tile_size = tuple(tile_size)
        self.tile_stride = tuple(tile_stride)
        self._vae = None

    @property
    def feature_dim(self) -> int:
        return 16

    def encode_image(self, image: Any) -> dict[str, Any]:
        torch = _import_torch()
        vae = self._load_vae()
        pil_image = _coerce_pil_image(image)
        dtype = _torch_dtype(torch, self.dtype_name)

        video = _preprocess_single_frame_for_vae(
            pil_image=pil_image,
            width=self.width,
            height=self.height,
            dtype=dtype,
            device=self.device,
        )

        with torch.no_grad():
            latent = vae.encode(
                [video],
                device=self.device,
                tiled=self.tiled,
                tile_size=self.tile_size,
                tile_stride=self.tile_stride,
            )
            latent_cpu = latent.detach().to(device="cpu", dtype=torch.float32)
            pooled_mean = latent_cpu.mean(dim=(2, 3, 4)).squeeze(0)
            pooled_std = latent_cpu.std(dim=(2, 3, 4), unbiased=False).squeeze(0)
            pooled = torch.cat([pooled_mean, pooled_std], dim=0)

        return {
            "latent": latent_cpu,
            "image_features": pooled,
            "metadata": {
                "backend": "kairos_vae",
                "latent_available": True,
                "image_features_available": True,
                "feature_dim": int(pooled.numel()),
                "latent_shape": list(latent_cpu.shape),
                "pooled_mean_shape": list(pooled_mean.shape),
                "pooled_std_shape": list(pooled_std.shape),
                "height": self.height,
                "width": self.width,
                "device": self.device,
                "dtype": self.dtype_name,
                "tiled": self.tiled,
                "tile_size": list(self.tile_size),
                "tile_stride": list(self.tile_stride),
                "audit": self.audit.to_dict(),
            },
        }

    def _load_vae(self):
        if self._vae is not None:
            return self._vae
        if not self.audit.vae_path or not self.audit.vae_path_exists:
            raise FileNotFoundError("Kairos VAE checkpoint is unavailable.")

        _ensure_repo_imports(self.repo_root)
        torch = _import_torch()
        from kairos.modules.vaes import WanVideoVAE

        dtype = _torch_dtype(torch, self.dtype_name)
        try:
            vae_state_dict = torch.load(self.audit.vae_path, map_location="cpu", weights_only=True)
        except TypeError:
            vae_state_dict = torch.load(self.audit.vae_path, map_location="cpu")
        if isinstance(vae_state_dict, dict) and "state_dict" in vae_state_dict:
            vae_state_dict = vae_state_dict["state_dict"]
        converter = WanVideoVAE.state_dict_converter()
        converted = converter.from_civitai(vae_state_dict)
        vae = WanVideoVAE()
        vae = vae.eval().requires_grad_(False)
        vae.load_state_dict(converted, assign=True)
        vae = vae.to(dtype=dtype, device=self.device)
        self._vae = vae
        return vae


def save_feature_summary(feature_payload: dict[str, Any], out_path: str | Path) -> None:
    torch = _import_torch()
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        "metadata": feature_payload["metadata"],
        "image_features": feature_payload["image_features"].tolist(),
        "image_features_mean": float(feature_payload["image_features"].mean().item()),
        "image_features_std": float(feature_payload["image_features"].std(unbiased=False).item()),
    }
    path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")

    latent_path = path.with_suffix(".latent.pt")
    torch.save(
        {
            "latent": feature_payload["latent"],
            "image_features": feature_payload["image_features"],
            "metadata": feature_payload["metadata"],
        },
        latent_path,
    )


def _load_mmengine_config(config_path: Path):
    try:
        from mmengine import Config
    except ModuleNotFoundError:
        namespace = runpy.run_path(str(config_path))
        return {
            key: value
            for key, value in namespace.items()
            if not key.startswith("__")
        }
    return Config.fromfile(str(config_path))


def _ensure_repo_imports(repo_root: Path) -> None:
    third_party = repo_root / "kairos" / "third_party"
    for path in (repo_root, third_party):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    injected = f"{repo_root}:{third_party}"
    existing = os.environ.get("PYTHONPATH", "")
    if injected not in existing:
        os.environ["PYTHONPATH"] = f"{injected}:{existing}" if existing else injected
    try:
        import kairos_ext._apex_shim  # noqa: F401
    except ModuleNotFoundError:
        pass


def _resolve_path(path: str | Path, repo_root: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return candidate.resolve()


def _resolve_optional_model_path(path: Any, repo_root: Path) -> Path | None:
    if not path:
        return None
    candidate = Path(str(path))
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    if candidate.exists():
        return candidate.resolve()
    if candidate.is_symlink():
        try:
            target = os.readlink(candidate)
        except OSError:
            return candidate
        marker = "kairos-sensenova/"
        if marker in target:
            candidate = repo_root / target.split(marker, 1)[1]
            if candidate.exists():
                return candidate.resolve()
    return candidate.resolve()


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _import_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required for Kairos feature extraction.") from exc
    return torch


def _torch_dtype(torch, dtype_name: str):
    normalized = dtype_name.lower().replace("torch.", "")
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def _coerce_pil_image(image: Any):
    from PIL import Image
    import numpy as np

    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    if hasattr(image, "shape"):
        return Image.fromarray(np.asarray(image).astype("uint8")).convert("RGB")
    raise TypeError(f"Unsupported image input type: {type(image)!r}")


def _preprocess_single_frame_for_vae(pil_image: Any, width: int, height: int, dtype: Any, device: str):
    import numpy as np

    torch = _import_torch()
    image = pil_image.resize((width, height))
    tensor = torch.tensor(np.array(image, dtype=np.float32), dtype=dtype, device=device)
    tensor = tensor * (2.0 / 255.0) - 1.0
    tensor = tensor.permute(2, 0, 1).unsqueeze(1).contiguous()
    return tensor
