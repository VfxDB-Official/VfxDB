from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class VfxDBTrainingDataContractTests(unittest.TestCase):
    def test_policy_transition_blocks_training_until_downloader_converges(self) -> None:
        from tile_dataloader import _collect_samples

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            transition = root / ".vfxdb/policy-transition.json"
            write_json(transition, {"incomplete": True})
            with self.assertRaisesRegex(RuntimeError, "policy transition is incomplete"):
                _collect_samples(str(root), categories=["Alpha"], require_meta=False)

    def test_loader_enumerates_only_installed_rows_with_requested_json(self) -> None:
        from tile_dataloader import _collect_samples

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            installed_vdb = root / "Alpha/0/usable.vdb"
            installed_vdb.parent.mkdir(parents=True)
            installed_vdb.write_bytes(b"fixture")
            write_json(root / "Alpha/index/usable.json", {"name": "usable"})
            write_json(
                root / "Alpha/category_index.json",
                {
                    "samples": [
                        {
                            "id": "usable",
                            "vdb_path": "0/usable.vdb",
                            "meta_path": "index/usable.json",
                            "deleted_bad_io_sample": False,
                        },
                        {
                            "id": "removed-bad",
                            "vdb_path": "0/removed-bad.vdb",
                            "meta_path": "index/removed-bad.json",
                            "deleted_bad_io_sample": True,
                        },
                    ]
                },
            )

            records, categories = _collect_samples(
                str(root), categories=["Alpha"], require_meta=True
            )
            self.assertEqual(categories, {"Alpha": 0})
            self.assertEqual([record.id for record in records], ["usable"])
            self.assertEqual(records[0].abs_meta_path, str(root / "Alpha/index/usable.json"))

    def test_vdb_return_meta_loads_json_and_rejects_invalid_content(self) -> None:
        from tile_dataloader import MetadataLoadError, SampleRecord, VolumeMultiDataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            meta_path = root / "sample.json"
            write_json(meta_path, {"bbox_min": [0, 0, 0], "name": "sample"})
            record = SampleRecord(
                id="sample",
                category="Alpha",
                data_path="0/sample.vdb",
                meta_path="index/sample.json",
                abs_data_path=str(root / "sample.vdb"),
                abs_meta_path=str(meta_path),
                abs_cat_dir=str(root),
            )
            dataset = object.__new__(VolumeMultiDataset)
            dataset.samples = [record]
            dataset.return_meta = True

            with mock.patch.object(
                VolumeMultiDataset,
                "vdb_ext_get",
                return_value={"category": "Alpha"},
            ):
                sample = dataset._get_core(0)
            self.assertEqual(sample["meta"]["name"], "sample")

            meta_path.write_text("[]\n", encoding="utf-8")
            with mock.patch.object(
                VolumeMultiDataset,
                "vdb_ext_get",
                return_value={"category": "Alpha"},
            ), self.assertRaisesRegex(MetadataLoadError, "must contain an object"):
                dataset._get_core(0)

    def test_return_meta_requires_json_and_does_not_silently_skip_samples(self) -> None:
        from tile_dataloader import MetadataLoadError, build_dataset_splits

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vdb_path = root / "Alpha/0/sample.vdb"
            vdb_path.parent.mkdir(parents=True)
            vdb_path.write_bytes(b"fixture")
            write_json(
                root / "Alpha/category_index.json",
                {
                    "samples": [
                        {
                            "id": "sample",
                            "vdb_path": "0/sample.vdb",
                            "meta_path": "index/sample.json",
                        }
                    ]
                },
            )

            with self.assertRaisesRegex(MetadataLoadError, "sample JSON is missing"):
                build_dataset_splits(
                    str(root),
                    categories=["Alpha"],
                    return_meta=True,
                    balance_train=False,
                )

    def test_metadata_failure_is_not_hidden_by_getitem_fallback(self) -> None:
        from tile_dataloader import MetadataLoadError, VolumeMultiDataset

        dataset = object.__new__(VolumeMultiDataset)
        dataset.samples = [object(), object()]
        dataset._get_core = mock.Mock(side_effect=MetadataLoadError("broken requested JSON"))
        with self.assertRaisesRegex(MetadataLoadError, "broken requested JSON"):
            dataset[0]
        dataset._get_core.assert_called_once_with(0)

    def test_metadata_batch_collation_accepts_different_json_keys(self) -> None:
        import torch

        from tile_dataloader import collate_volume_batch

        batch = [
            {"volume": torch.zeros(1, 2, 2, 2), "category_id": 0, "meta": {"a": 1}},
            {"volume": torch.ones(1, 2, 2, 2), "category_id": 1, "meta": {"b": 2}},
        ]
        collated = collate_volume_batch(batch)
        self.assertEqual(tuple(collated["volume"].shape), (2, 1, 2, 2, 2))
        self.assertEqual(collated["meta"], [{"a": 1}, {"b": 2}])

    def test_frame_bboxes_are_unioned_for_sequence_sampling(self) -> None:
        import numpy as np
        import tile_dataloader as loader

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = []
            for frame, bmin, bmax in (
                (1, [-2, 0, -1], [3, 4, 5]),
                (2, [-7, -1, 0], [8, 2, 9]),
            ):
                meta = root / f"index/sample__n{frame:04d}.json"
                write_json(meta, {"bbox_min": bmin, "bbox_max": bmax})
                records.append(
                    loader.SampleRecord(
                        id=f"sample-{frame}",
                        category="Alpha",
                        data_path=f"0/sample__n{frame:04d}.vdb",
                        meta_path=f"index/{meta.name}",
                        abs_data_path=str(root / f"0/sample__n{frame:04d}.vdb"),
                        abs_meta_path=str(meta),
                        abs_cat_dir=str(root),
                    )
                )
            with mock.patch.object(loader, "HAS_VDB_EXT", True):
                dataset = loader.VolumeMultiDataset(
                    records,
                    {"Alpha": 0},
                    prev_k=1,
                    prev_bbox_mode="seq",
                    transform_included=False,
                )
            bmin, bmax = dataset._seq_bbox_minmax["Alpha/0"]
            np.testing.assert_array_equal(bmin, [-7, -1, -1])
            np.testing.assert_array_equal(bmax, [8, 4, 9])


if __name__ == "__main__":
    unittest.main()
