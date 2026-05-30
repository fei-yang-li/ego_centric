from __future__ import annotations

import unittest

from ego_vla.schemas import HandPoseEstimate, Landmark, VLAFrameRecord, json_schema_example


class SchemaTests(unittest.TestCase):
    def test_record_to_dict_drops_none_values(self) -> None:
        record = VLAFrameRecord(
            dataset_name="test",
            episode_id="episode",
            frame_id="episode/000000",
            frame_index=0,
            timestamp_sec=0.0,
            observation={"rgb": {"path": "frames/000000.jpg"}},
            camera={"intrinsics": None, "coordinate_frame": "camera"},
            hands={
                "left": HandPoseEstimate(
                    side="left",
                    source="unit",
                    coordinate_frame="image_normalized",
                    landmarks=[Landmark(name="wrist", x=0.1, y=0.2, z=None)],
                )
            },
        )
        payload = record.to_dict()
        self.assertNotIn("body_pose_3d", payload)
        self.assertNotIn("intrinsics", payload["camera"])
        self.assertNotIn("z", payload["hands"]["left"]["landmarks"][0])

    def test_schema_example_contains_vla_fields(self) -> None:
        payload = json_schema_example()
        for key in ("observation", "camera", "hands", "depth", "ego_motion", "action"):
            self.assertIn(key, payload)


if __name__ == "__main__":
    unittest.main()
