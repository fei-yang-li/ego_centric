"""Depth (record-level) backends.

Depth is conceptually per-frame, but runs as a ``RecordProcessor`` so a backend
can batch frames through a GPU model and write depth maps next to the frames.

Backends:

- ``none``                    -- annotate ``depth`` as not computed.
- ``depth_anything_v2``       -- Depth Anything V2 (relative depth) via
  Hugging Face ``transformers``.
- ``depth_anything_v2_metric``-- Depth Anything V2 metric checkpoints.
- ``metric3d`` / ``unidepth`` -- experimental adapters for metric monocular
  depth models (lazy import; clear error if not installed).

Depth maps are written under ``<output>/depth/`` as 16-bit PNG (metric =
millimetres; relative = per-frame normalised) or ``.npy`` float32 when
``encoding`` is ``npy``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from ego_vla.backends._common import (
    encode_depth_png16,
    load_cv2,
    missing_dependency,
    read_frame_rgb,
    resize_max_side,
)
from ego_vla.config import BackendConfig, ProcessingConfig
from ego_vla.schemas import DenseSignal, VLAFrameRecord
from ego_vla.stages import DEPTH_BACKENDS, StageContext

_NOT_COMPUTED_NOTE = "Depth is not computed by the default pipeline."

# infer_fn: takes an HxWx3 uint8 RGB array, returns an HxW float depth array.
InferFn = Callable[[Any], Any]


class NoDepth:
    backend_name = "none"

    def __init__(self, section: BackendConfig, config: ProcessingConfig) -> None:
        self._section = section

    def run(self, records: list[VLAFrameRecord], ctx: StageContext) -> None:
        for record in records:
            record.depth.backend = "none"
            record.depth.status = "not_computed"
            if _NOT_COMPUTED_NOTE not in record.depth.notes:
                record.depth.notes.append(_NOT_COMPUTED_NOTE)

    def close(self) -> None:
        return None


class DepthMapProcessor:
    """Run a per-frame depth model and persist depth maps + ``DenseSignal``.

    The model is injected as ``infer_fn`` so this orchestration is testable
    without any deep-learning dependency.
    """

    def __init__(
        self,
        infer_fn: InferFn,
        backend_name: str,
        metric: bool,
        *,
        encoding: str = "png16",
        max_side: int | None = None,
        model_name: str | None = None,
    ) -> None:
        self._infer = infer_fn
        self.backend_name = backend_name
        self._metric = metric
        self._encoding = encoding.lower()
        self._max_side = max_side
        self._model_name = model_name
        if self._encoding not in {"png16", "npy"}:
            raise ValueError("depth encoding must be 'png16' or 'npy'")

    def run(self, records: list[VLAFrameRecord], ctx: StageContext) -> None:
        if not records:
            return
        import numpy as np

        cv2 = load_cv2()
        depth_dir = ctx.output_dir / "depth"
        depth_dir.mkdir(parents=True, exist_ok=True)
        written = 0
        for record in records:
            frame_rgb = read_frame_rgb(ctx.resolve_frame_path(record))
            model_input = resize_max_side(frame_rgb, self._max_side)
            depth = np.asarray(self._infer(model_input), dtype=np.float32)
            if depth.shape[:2] != frame_rgb.shape[:2]:
                depth = cv2.resize(
                    depth,
                    (frame_rgb.shape[1], frame_rgb.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )

            stem = Path(record.observation["rgb"]["path"]).stem
            notes: list[str] = []
            if self._encoding == "npy":
                rel_path = f"depth/{stem}.npy"
                np.save(ctx.output_dir / rel_path, depth)
                notes.append("depth stored as float32 .npy")
            else:
                encoded, meta = encode_depth_png16(depth, self._metric)
                rel_path = f"depth/{stem}.png"
                cv2.imwrite(str(ctx.output_dir / rel_path), encoded)
                if meta.get("normalized"):
                    notes.append(
                        f"16-bit PNG, per-frame normalized in [{meta['min']:.4f}, "
                        f"{meta['max']:.4f}] (not comparable across frames)"
                    )
                else:
                    notes.append("16-bit PNG in millimetres (decode with value * 0.001)")

            if self._model_name:
                notes.append(f"model: {self._model_name}")
            record.depth = DenseSignal(
                path=rel_path,
                backend=self.backend_name,
                status="ok",
                coordinate_frame="camera_optical",
                metric_scale=self._metric,
                notes=notes,
            )
            written += 1

        ctx.extra.setdefault("depth", {}).update(
            {"backend": self.backend_name, "frames": written, "dir": "depth", "metric": self._metric}
        )

    def close(self) -> None:
        return None


def _build_depth_anything_infer(model_name: str, device: str) -> InferFn:
    try:
        import numpy as np
        from PIL import Image
        from transformers import pipeline
    except ImportError as exc:  # pragma: no cover - exercised only without deps
        raise missing_dependency(
            "depth_anything_v2", "torch transformers pillow", extra="depth"
        ) from exc

    pipe = pipeline("depth-estimation", model=model_name, device=0 if device.startswith("cuda") else -1)

    def infer(rgb: Any) -> Any:
        result = pipe(Image.fromarray(rgb))
        predicted = result.get("predicted_depth")
        if predicted is not None:
            return predicted.squeeze().detach().cpu().numpy().astype("float32")
        return np.asarray(result["depth"], dtype="float32")

    return infer


def _build_hub_infer(backend: str, repo: str, extra_packages: str) -> InferFn:
    try:
        import numpy as np  # noqa: F401
        import torch
    except ImportError as exc:  # pragma: no cover
        raise missing_dependency(backend, extra_packages, extra="depth") from exc

    try:
        model = torch.hub.load(repo, "metric3d_vit_small", pretrain=True)
    except Exception as exc:  # pragma: no cover - network/hub specifics
        raise RuntimeError(
            f"Could not load the {backend!r} model from torch.hub ({repo}). "
            "See the model's documentation for setup."
        ) from exc
    model.eval()

    def infer(rgb: Any) -> Any:  # pragma: no cover - requires GPU model
        import numpy as np
        import torch

        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        with torch.no_grad():
            output = model(tensor)
        depth = output[0] if isinstance(output, (tuple, list)) else output
        return depth.squeeze().cpu().numpy().astype(np.float32)

    return infer


@DEPTH_BACKENDS.register("none")
def _build_none(section: BackendConfig, config: ProcessingConfig) -> NoDepth:
    return NoDepth(section, config)


@DEPTH_BACKENDS.register("depth_anything_v2")
def _build_depth_anything_v2(
    section: BackendConfig, config: ProcessingConfig
) -> DepthMapProcessor:
    extra = dict(section.extra or {})
    model_name = str(extra.get("model", "depth-anything/Depth-Anything-V2-Small-hf"))
    return DepthMapProcessor(
        _build_depth_anything_infer(model_name, str(extra.get("device", "cpu"))),
        backend_name="depth_anything_v2",
        metric=False,
        encoding=str(extra.get("encoding", "png16")),
        max_side=extra.get("max_side"),
        model_name=model_name,
    )


@DEPTH_BACKENDS.register("depth_anything_v2_metric")
def _build_depth_anything_v2_metric(
    section: BackendConfig, config: ProcessingConfig
) -> DepthMapProcessor:
    extra = dict(section.extra or {})
    model_name = str(
        extra.get("model", "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf")
    )
    return DepthMapProcessor(
        _build_depth_anything_infer(model_name, str(extra.get("device", "cpu"))),
        backend_name="depth_anything_v2_metric",
        metric=True,
        encoding=str(extra.get("encoding", "png16")),
        max_side=extra.get("max_side"),
        model_name=model_name,
    )


@DEPTH_BACKENDS.register("metric3d")
def _build_metric3d(section: BackendConfig, config: ProcessingConfig) -> DepthMapProcessor:
    extra = dict(section.extra or {})
    return DepthMapProcessor(
        _build_hub_infer("metric3d", "yvanyin/metric3d", "torch"),
        backend_name="metric3d",
        metric=True,
        encoding=str(extra.get("encoding", "png16")),
        max_side=extra.get("max_side"),
        model_name="metric3d_vit_small",
    )


@DEPTH_BACKENDS.register("unidepth")
def _build_unidepth(section: BackendConfig, config: ProcessingConfig) -> DepthMapProcessor:
    extra = dict(section.extra or {})

    def _infer_factory() -> InferFn:
        try:
            import numpy as np
            import torch
            from unidepth.models import UniDepthV2
        except ImportError as exc:  # pragma: no cover
            raise missing_dependency(
                "unidepth",
                "torch and unidepth (pip install git+https://github.com/lpiccinelli-eth/UniDepth)",
            ) from exc

        model = UniDepthV2.from_pretrained(
            extra.get("model", "lpiccinelli/unidepth-v2-vits14")
        )
        model.eval()

        def infer(rgb: Any) -> Any:  # pragma: no cover - requires GPU model
            tensor = torch.from_numpy(rgb).permute(2, 0, 1)
            predictions = model.infer(tensor)
            return predictions["depth"].squeeze().cpu().numpy().astype(np.float32)

        return infer

    return DepthMapProcessor(
        _infer_factory(),
        backend_name="unidepth",
        metric=True,
        encoding=str(extra.get("encoding", "png16")),
        max_side=extra.get("max_side"),
        model_name="unidepth-v2",
    )
