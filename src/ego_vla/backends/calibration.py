"""Camera calibration (pre-processing) backends.

Calibration is the metric anchor for the depth and ego-motion stages.

Backends:

- ``none``               -- forward any ``camera.intrinsics`` from config.
- ``pinhole_prior``      -- approximate intrinsics from a horizontal-FOV guess
  (no calibration footage needed; not metric).
- ``opencv_checkerboard``-- OpenCV checkerboard calibration from a folder of
  calibration photos (metric intrinsics + distortion).
- ``colmap``             -- shell out to COLMAP for SfM self-calibration
  (experimental; requires the ``colmap`` binary).
"""

from __future__ import annotations

import glob
import shutil
import subprocess
import tempfile
from pathlib import Path

from ego_vla.backends._common import load_cv2, pinhole_intrinsics
from ego_vla.config import BackendConfig, ProcessingConfig
from ego_vla.schemas import CameraCalibration
from ego_vla.stages import CALIBRATION_BACKENDS, StageContext


class StaticCalibration:
    backend_name = "none"

    def __init__(self, section: BackendConfig, config: ProcessingConfig) -> None:
        self._config = config

    def run(self, ctx: StageContext) -> CameraCalibration:
        camera = self._config.camera
        intrinsics = camera.intrinsics
        notes: list[str] = []
        if intrinsics:
            status = "provided"
        else:
            status = "not_computed"
            notes.append(
                "No calibration backend; camera intrinsics are unknown. Provide "
                "camera.intrinsics or add a calibration backend for metric depth/SLAM."
            )
        return CameraCalibration(
            backend=self.backend_name,
            status=status,
            coordinate_frame=camera.coordinate_frame,
            intrinsics=intrinsics,
            image_width=ctx.video_metadata.width or None,
            image_height=ctx.video_metadata.height or None,
            metric_scale=True if intrinsics else None,
            notes=notes,
        )


class PinholePriorCalibration:
    """Approximate intrinsics from a horizontal field-of-view assumption.

    This is the common practical fallback for uncalibrated handheld video: it
    yields a usable pinhole model for visual odometry but is *not* metric and
    has no distortion model.
    """

    backend_name = "pinhole_prior"

    def __init__(self, section: BackendConfig, config: ProcessingConfig) -> None:
        extra = dict(section.extra or {})
        self._fov_deg = float(extra.get("horizontal_fov_deg", 70.0))
        self._config = config

    def run(self, ctx: StageContext) -> CameraCalibration:
        width = ctx.video_metadata.width
        height = ctx.video_metadata.height
        if not width or not height:
            raise RuntimeError(
                "pinhole_prior calibration needs the video width/height from probing."
            )
        intrinsics = pinhole_intrinsics(width, height, self._fov_deg)
        return CameraCalibration(
            backend=self.backend_name,
            status="approximate",
            coordinate_frame=self._config.camera.coordinate_frame,
            intrinsics=intrinsics,
            image_width=width,
            image_height=height,
            metric_scale=None,
            notes=[
                f"Intrinsics approximated from a horizontal FOV of {self._fov_deg} deg; "
                "usable for up-to-scale VO but not metric-calibrated.",
            ],
        )


class OpenCVCheckerboardCalibration:
    """Standard OpenCV checkerboard calibration from a folder of photos.

    Configure via ``calibration.extra``::

        {"images_dir": "calib/", "pattern_size": [9, 6], "square_size_m": 0.025}

    ``pattern_size`` is the number of *inner* corners (cols, rows).
    """

    backend_name = "opencv_checkerboard"

    def __init__(self, section: BackendConfig, config: ProcessingConfig) -> None:
        extra = dict(section.extra or {})
        self._images_dir = extra.get("images_dir")
        cols, rows = extra.get("pattern_size", [9, 6])
        self._pattern = (int(cols), int(rows))
        self._square_size = float(extra.get("square_size_m", 0.025))
        self._config = config
        if not self._images_dir:
            raise RuntimeError(
                "opencv_checkerboard requires calibration.extra.images_dir pointing "
                "to a folder of checkerboard photos."
            )

    def run(self, ctx: StageContext) -> CameraCalibration:
        import numpy as np

        cv2 = load_cv2()
        paths = sorted(
            p
            for p in glob.glob(str(Path(self._images_dir) / "*"))
            if Path(p).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )
        objp = np.zeros((self._pattern[0] * self._pattern[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0 : self._pattern[0], 0 : self._pattern[1]].T.reshape(-1, 2)
        objp *= self._square_size

        object_points: list = []
        image_points: list = []
        image_size: tuple[int, int] | None = None
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        for path in paths:
            image = cv2.imread(path)
            if image is None:
                continue
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image_size = (gray.shape[1], gray.shape[0])
            found, corners = cv2.findChessboardCorners(gray, self._pattern, None)
            if not found:
                continue
            refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            object_points.append(objp)
            image_points.append(refined)

        if len(object_points) < 3 or image_size is None:
            raise RuntimeError(
                f"Checkerboard found in only {len(object_points)} image(s) under "
                f"{self._images_dir!r}; need at least 3 valid views."
            )

        rms, matrix, dist, _, _ = cv2.calibrateCamera(
            object_points, image_points, image_size, None, None
        )
        intrinsics = {
            "fx": float(matrix[0, 0]),
            "fy": float(matrix[1, 1]),
            "cx": float(matrix[0, 2]),
            "cy": float(matrix[1, 2]),
        }
        return CameraCalibration(
            backend=self.backend_name,
            status="calibrated",
            coordinate_frame=self._config.camera.coordinate_frame,
            intrinsics=intrinsics,
            distortion=[float(v) for v in dist.ravel().tolist()],
            image_width=image_size[0],
            image_height=image_size[1],
            metric_scale=True,
            confidence=float(rms),
            notes=[
                f"Calibrated from {len(object_points)} checkerboard view(s); "
                f"RMS reprojection error {rms:.4f} px.",
            ],
        )


class ColmapCalibration:
    """COLMAP-based self-calibration (experimental, requires the colmap binary).

    Configure via ``calibration.extra``::

        {"images_dir": "frames/", "camera_model": "OPENCV"}

    Runs feature extraction + matching + a sparse reconstruction and reads the
    estimated intrinsics back from the model. GPU is optional but recommended.
    """

    backend_name = "colmap"

    def __init__(self, section: BackendConfig, config: ProcessingConfig) -> None:
        extra = dict(section.extra or {})
        self._images_dir = extra.get("images_dir")
        self._camera_model = str(extra.get("camera_model", "OPENCV"))
        self._binary = str(extra.get("binary", "colmap"))
        self._config = config
        if not self._images_dir:
            raise RuntimeError(
                "colmap calibration requires calibration.extra.images_dir with images."
            )
        if shutil.which(self._binary) is None:
            raise RuntimeError(
                f"The 'colmap' binary ({self._binary!r}) was not found on PATH. Install "
                "COLMAP (https://colmap.github.io) or use 'opencv_checkerboard'/'pinhole_prior'."
            )

    def run(self, ctx: StageContext) -> CameraCalibration:
        with tempfile.TemporaryDirectory() as work:
            work_path = Path(work)
            database = work_path / "database.db"
            sparse = work_path / "sparse"
            sparse.mkdir(parents=True, exist_ok=True)
            self._run(
                [
                    self._binary,
                    "feature_extractor",
                    "--database_path",
                    str(database),
                    "--image_path",
                    str(self._images_dir),
                    "--ImageReader.camera_model",
                    self._camera_model,
                    "--ImageReader.single_camera",
                    "1",
                ]
            )
            self._run(
                [self._binary, "exhaustive_matcher", "--database_path", str(database)]
            )
            self._run(
                [
                    self._binary,
                    "mapper",
                    "--database_path",
                    str(database),
                    "--image_path",
                    str(self._images_dir),
                    "--output_path",
                    str(sparse),
                ]
            )
            intrinsics, width, height, distortion = _read_colmap_intrinsics(
                self._binary, sparse
            )
        return CameraCalibration(
            backend=self.backend_name,
            status="calibrated",
            coordinate_frame=self._config.camera.coordinate_frame,
            intrinsics=intrinsics,
            distortion=distortion,
            image_width=width,
            image_height=height,
            metric_scale=False,
            notes=[
                "Intrinsics from COLMAP SfM self-calibration; reconstruction scale "
                "is arbitrary (use known baseline or metric depth for metric scale).",
            ],
        )

    def _run(self, command: list[str]) -> None:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"COLMAP step failed ({' '.join(command[:2])}): {result.stderr.strip()[:500]}"
            )


def _read_colmap_intrinsics(binary: str, sparse_dir: Path):
    import numpy as np  # noqa: F401  (kept for parity with other backends)

    models = sorted(p for p in sparse_dir.glob("*") if p.is_dir())
    if not models:
        raise RuntimeError("COLMAP produced no reconstruction; check image overlap/quality.")
    model = models[0]
    cameras_txt = model / "cameras.txt"
    if not cameras_txt.exists():
        subprocess.run(
            [
                binary,
                "model_converter",
                "--input_path",
                str(model),
                "--output_path",
                str(model),
                "--output_type",
                "TXT",
            ],
            capture_output=True,
            text=True,
        )
    for line in cameras_txt.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        # CAMERA_ID MODEL WIDTH HEIGHT PARAMS...
        width = int(parts[2])
        height = int(parts[3])
        params = [float(v) for v in parts[4:]]
        fx = params[0]
        fy = params[1] if len(params) > 1 else params[0]
        cx = params[2] if len(params) > 2 else width / 2.0
        cy = params[3] if len(params) > 3 else height / 2.0
        distortion = params[4:] if len(params) > 4 else None
        return {"fx": fx, "fy": fy, "cx": cx, "cy": cy}, width, height, distortion
    raise RuntimeError("Could not parse intrinsics from COLMAP cameras.txt.")


@CALIBRATION_BACKENDS.register("none")
def _build_none(section: BackendConfig, config: ProcessingConfig) -> StaticCalibration:
    return StaticCalibration(section, config)


@CALIBRATION_BACKENDS.register("pinhole_prior")
def _build_pinhole_prior(
    section: BackendConfig, config: ProcessingConfig
) -> PinholePriorCalibration:
    return PinholePriorCalibration(section, config)


@CALIBRATION_BACKENDS.register("opencv_checkerboard")
def _build_checkerboard(
    section: BackendConfig, config: ProcessingConfig
) -> OpenCVCheckerboardCalibration:
    return OpenCVCheckerboardCalibration(section, config)


@CALIBRATION_BACKENDS.register("colmap")
def _build_colmap(section: BackendConfig, config: ProcessingConfig) -> ColmapCalibration:
    return ColmapCalibration(section, config)
