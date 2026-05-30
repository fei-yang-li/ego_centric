from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ego_vla.config import ProcessingConfig, load_config, write_default_config


class ConfigTests(unittest.TestCase):
    def test_default_config_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json"
            write_default_config(config_path)
            config = load_config(config_path)
        self.assertIsInstance(config, ProcessingConfig)
        self.assertEqual(config.video.sample_rate_hz, 5.0)
        self.assertEqual(config.pose.backend, "mediapipe")

    def test_unknown_config_key_fails_fast(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "bad.json"
            config_path.write_text(json.dumps({"video": {"bad_key": 1}}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(config_path)


if __name__ == "__main__":
    unittest.main()
