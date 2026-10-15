import unittest

from roboicl.config import load_adaptive_horizons
from roboicl.reference_setting import load_reference_setting


class ReferenceSettingTest(unittest.TestCase):
    def test_paper_task_partition_and_locked_windows(self):
        setting = load_reference_setting("configs/references/one_shot_j12_b12.json")
        self.assertEqual(len(setting["zero_shot_tasks"]), 8)
        self.assertEqual(len(setting["one_shot_references"]), 22)
        categories = {}
        for row in setting["zero_shot_tasks"] + setting["one_shot_references"]:
            categories[row["category"]] = categories.get(row["category"], 0) + 1
        self.assertEqual(categories, {"Open": 8, "Memory": 6, "Precision": 8, "Long-Horizon": 8})

    def test_reference_horizons_match_main_profile(self):
        setting = load_reference_setting("configs/references/one_shot_j12_b12.json")
        horizons = load_adaptive_horizons()
        for row in setting["one_shot_references"]:
            self.assertEqual(row["reference_horizon"], horizons[row["task"]])
            self.assertNotIn("evaluation_status", row)
            self.assertNotIn("exception", row)
            self.assertNotIn("bundle_source_episode", row["source"])
        swap_blocks = next(row for row in setting["one_shot_references"] if row["task"] == "swap_blocks")
        self.assertEqual(swap_blocks["reference_horizon"], 10)

    def test_replaced_upstream_sources_pin_their_original_revisions(self):
        setting = load_reference_setting("configs/references/one_shot_j12_b12.json")
        records = {row["task"]: row["source"] for row in setting["one_shot_references"]}
        self.assertEqual(
            records["build_tower"]["repository_revision"],
            "0a709fe7863f6c869a8d141676585a883fce4270",
        )
        self.assertEqual(
            records["organize_table"]["repository_revision"],
            "16f509f1dd2a10a2ed7312ef0de8dd7db135869b",
        )

    def test_repository_episode_alias_is_explicit(self):
        setting = load_reference_setting("configs/references/one_shot_j12_b12.json")
        records = {row["task"]: row["source"] for row in setting["one_shot_references"]}
        self.assertEqual(records["swap_blocks"]["episode"], 1)
        self.assertNotIn("repository_episode", records["swap_blocks"])
        self.assertEqual(records["play_Xylophone"]["episode"], 0)
        self.assertEqual(records["play_Xylophone"]["repository_episode"], 1)


if __name__ == "__main__":
    unittest.main()
