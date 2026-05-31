"""Built-in pipeline backends.

Importing this package registers every built-in backend into the registries
declared in :mod:`ego_vla.stages`. Heavy or optional third-party dependencies
(MediaPipe, depth models, detectors, VLMs) are imported lazily inside the
factories so that importing this package stays cheap and dependency-free.

To add a new backend, create a module here (or anywhere) that registers a
factory, e.g.::

    from ego_vla.stages import DEPTH_BACKENDS

    @DEPTH_BACKENDS.register("depth_anything_v2")
    def _build(section, config):
        return DepthAnythingV2Processor(section, config)

then import it from this package's ``__init__`` (or call
``ego_vla.backends.load()``).
"""

from __future__ import annotations

from ego_vla.backends import (  # noqa: F401  (imported for registration side-effects)
    action,
    calibration,
    captioning,
    depth,
    ego_motion,
    objects,
    pose,
    segmentation,
)


def load() -> None:
    """No-op hook; importing the package already performs registration."""
    return None
