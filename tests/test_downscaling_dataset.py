import pytest
import torch
import yaml
from pathlib import Path

from credit.datasets.downscaling_dataset import DownscalingDataset


@pytest.fixture
def config():
    """Load test configuration from YAML.

    Paths in the YAML (``rootpath``) are written relative to the repository
    root, so they are resolved to absolute paths here. This lets the tests run
    regardless of the directory pytest is invoked from.
    """
    repo_root = Path(__file__).resolve().parent.parent
    config_path = Path(__file__).parent / "data" / "downscaling" / "dataset.yml"
    with open(config_path) as f:
        conf = yaml.safe_load(f)

    def _absolutize_rootpaths(node):
        """Recursively rewrite relative ``rootpath`` values to absolute paths."""
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "rootpath" and isinstance(value, str) and not Path(value).is_absolute():
                    node[key] = str(repo_root / value)
                else:
                    _absolutize_rootpaths(value)
        elif isinstance(node, list):
            for item in node:
                _absolutize_rootpaths(item)

    _absolutize_rootpaths(conf["data"])
    return conf


@pytest.fixture
def dataset(config):
    """Create dataset instance for testing."""
    return DownscalingDataset(**config["data"])


class TestDownscalingDataset:
    """Integration smoke tests for DownscalingDataset."""

    def test_initialization(self, config):
        """Test dataset can be created without errors."""
        dataset = DownscalingDataset(**config["data"])
        assert dataset is not None
        assert len(dataset) > 0

    def test_dataset_length(self, dataset):
        """Test dataset length is reasonable."""
        assert len(dataset) > 0
        # Based on config: 22 hours of data, history=2, forecast=1
        # Should have fewer samples than total timesteps
        assert len(dataset) <= 22

    def test_getitem_tensor_mode(self, dataset):
        """Test __getitem__ returns proper tensor format."""
        dataset.output = "tensor"
        sample = dataset[0]

        # Check structure
        assert isinstance(sample, dict)
        assert "input" in sample or "x" in sample
        assert "target" in sample or "y" in sample
        assert "dates" in sample

        # Check tensor properties
        if "x" in sample and sample["x"] is not None:
            assert isinstance(sample["x"], torch.Tensor)
            assert sample["x"].dim() == 5  # [batch, var, time, height, width]

        if "y" in sample and sample["y"] is not None:
            assert isinstance(sample["y"], torch.Tensor)
            assert sample["y"].dim() == 5

    def test_getitem_by_dset_mode(self, dataset):
        """Test __getitem__ with by_dset output format."""
        dataset.output = "by_dset"
        sample = dataset[0]

        assert isinstance(sample, dict)
        # Should have nested structure [dataset][usage][variable]
        for dset_name in dataset.datasets:
            if dset_name in sample:
                assert isinstance(sample[dset_name], dict)

    def test_getitem_by_io_mode(self, dataset):
        """Test __getitem__ with by_io output format."""
        dataset.output = "by_io"
        sample = dataset[0]

        assert isinstance(sample, dict)
        assert "input" in sample
        assert "target" in sample

    def test_mode_switching(self, dataset):
        """Test switching between train/init/infer modes."""
        original_mode = dataset.mode

        for mode in ["train", "init", "infer"]:
            dataset.mode = mode
            assert dataset.mode == mode

            # Should be able to get sample in any mode
            sample = dataset[0]
            assert sample is not None

        # Reset to original
        dataset.mode = original_mode

    def test_invalid_mode(self, dataset):
        """Test invalid mode raises error."""
        with pytest.raises(ValueError):
            dataset.mode = "invalid_mode"

    def test_invalid_output(self, dataset):
        """Test invalid output format raises error."""
        with pytest.raises(ValueError):
            dataset.output = "invalid_output"

    def test_arrangement_property(self, dataset):
        """Test arrangement dataframe exists and has expected structure."""
        assert hasattr(dataset, "arrangement")
        df = dataset.arrangement

        # Check required columns exist
        expected_cols = ["dataset", "dim", "var", "usage", "name"]
        for col in expected_cols:
            assert col in df.columns

        # Check non-empty
        assert len(df) > 0

    def test_tnames_property(self, dataset):
        """Test tnames list exists for tensor channel naming."""
        assert hasattr(dataset, "tnames")
        assert isinstance(dataset.tnames, list)
        assert len(dataset.tnames) > 0

    def test_sample_consistency(self, dataset):
        """Test multiple samples from same index are consistent."""
        dataset.output = "tensor"

        sample1 = dataset[0]
        sample2 = dataset[0]

        # Dates should be identical
        assert sample1["dates"] == sample2["dates"]

        # Tensor shapes should match
        if sample1.get("x") is not None and sample2.get("x") is not None:
            assert sample1["x"].shape == sample2["x"].shape

        if sample1.get("y") is not None and sample2.get("y") is not None:
            assert sample1["y"].shape == sample2["y"].shape


if __name__ == "__main__":
    pytest.main([__file__])
