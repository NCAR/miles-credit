"""Unit tests for credit.registry and the register_* decorators."""

import logging
import sys

import pytest
import torch
import torch.nn as nn

from credit.models.base_model import BaseModel
from credit.preblock.base import BasePreblock
from credit.postblock.base import BasePostblock
from credit.datasets.base_dataset import BaseDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_installable_module(tmp_path, module_name, source):
    """Write source to tmp_path/<module_name>.py and add tmp_path to sys.path."""
    (tmp_path / f"{module_name}.py").write_text(source)
    if str(tmp_path) not in sys.path:
        sys.path.insert(0, str(tmp_path))
    return module_name


# ---------------------------------------------------------------------------
# credit.registry.load_custom_objects
# ---------------------------------------------------------------------------


class TestLoadCustomObjects:
    def test_does_nothing_when_absent(self):
        """load_custom_objects does nothing when the custom_objects key is missing."""
        from credit.registry import load_custom_objects

        load_custom_objects({})

    def test_registers_dataset(self, tmp_path):
        """A dataset entry with correct base class is imported and registered."""
        _make_installable_module(
            tmp_path,
            "reg_test_dataset_pkg",
            "from credit.datasets.base_dataset import BaseDataset\n"
            "class RegTestDataset(BaseDataset):\n"
            "    def __init__(self, conf, rank=0, world_size=1, is_train=True): pass\n"
            "    def __len__(self): return 0\n"
            "    def __getitem__(self, i): return {}\n",
        )

        from credit.registry import load_custom_objects
        from credit.datasets import _DATASET_REGISTRY

        conf = {
            "custom_objects": {
                "RegTestDataset": {
                    "object_type": "dataset",
                    "module_path": "reg_test_dataset_pkg",
                    "module_name": "RegTestDataset",
                }
            }
        }
        load_custom_objects(conf)
        assert "RegTestDataset" in _DATASET_REGISTRY

    def test_registers_preblock(self, tmp_path):
        """A preblock entry with correct base class is imported and registered."""
        _make_installable_module(
            tmp_path,
            "reg_test_preblock_pkg",
            "from credit.preblock.base import BasePreblock\n"
            "class RegTestPreBlock(BasePreblock):\n"
            "    def forward(self, batch): return batch\n",
        )

        from credit.registry import load_custom_objects
        from credit.preblock import _PREBLOCK_REGISTRY

        conf = {
            "custom_objects": {
                "RegTestPreBlock": {
                    "object_type": "preblock",
                    "module_path": "reg_test_preblock_pkg",
                    "module_name": "RegTestPreBlock",
                }
            }
        }
        load_custom_objects(conf)
        assert "RegTestPreBlock" in _PREBLOCK_REGISTRY

    def test_registers_model(self, tmp_path):
        """A model entry with correct base class is imported and registered."""
        _make_installable_module(
            tmp_path,
            "reg_test_model_pkg",
            "from credit.models.base_model import BaseModel\n"
            "class RegTestModel(BaseModel):\n"
            "    def __init__(self): super().__init__()\n"
            "    def forward(self, x): return x\n",
        )

        from credit.registry import load_custom_objects
        from credit.models import _MODEL_REGISTRY

        conf = {
            "custom_objects": {
                "RegTestModel": {
                    "object_type": "model",
                    "module_path": "reg_test_model_pkg",
                    "module_name": "RegTestModel",
                }
            }
        }
        load_custom_objects(conf)
        assert "RegTestModel" in _MODEL_REGISTRY

    def test_registers_postblock(self, tmp_path):
        """A postblock entry with correct base class is imported and registered."""
        _make_installable_module(
            tmp_path,
            "reg_test_postblock_pkg",
            "from credit.postblock.base import BasePostblock\n"
            "class RegTestPostBlock(BasePostblock):\n"
            "    def forward(self, batch): return batch\n",
        )

        from credit.registry import load_custom_objects
        from credit.postblock import _POSTBLOCK_REGISTRY

        conf = {
            "custom_objects": {
                "RegTestPostBlock": {
                    "object_type": "postblock",
                    "module_path": "reg_test_postblock_pkg",
                    "module_name": "RegTestPostBlock",
                }
            }
        }
        load_custom_objects(conf)
        assert "RegTestPostBlock" in _POSTBLOCK_REGISTRY

    def test_registers_loss(self, tmp_path):
        """A loss entry is imported and registered in the loss registry."""
        _make_installable_module(
            tmp_path,
            "reg_test_loss_pkg",
            "import torch.nn as nn\n"
            "class RegTestLoss(nn.Module):\n"
            "    def __init__(self, reduction='mean'): super().__init__()\n"
            "    def forward(self, x, y): return (x - y).abs().mean()\n",
        )

        from credit.registry import load_custom_objects
        from credit.losses import _LOSS_REGISTRY

        conf = {
            "custom_objects": {
                "RegTestLoss": {
                    "object_type": "loss",
                    "module_path": "reg_test_loss_pkg",
                    "module_name": "RegTestLoss",
                }
            }
        }
        load_custom_objects(conf)
        assert "RegTestLoss" in _LOSS_REGISTRY

    def test_yaml_key_is_registry_key(self, tmp_path):
        """The YAML key, not the class name, is used as the registry key.

        This allows stable config aliases (e.g. 'mymodel') that are independent
        of the Python class name (e.g. 'MyModel'), and lets two packages with
        identically-named classes coexist without collision.
        """
        _make_installable_module(
            tmp_path,
            "reg_test_alias_pkg",
            "from credit.models.base_model import BaseModel\n"
            "class MyModel(BaseModel):\n"
            "    def __init__(self): super().__init__()\n"
            "    def forward(self, x): return x\n",
        )

        from credit.registry import load_custom_objects
        from credit.models import _MODEL_REGISTRY

        conf = {
            "custom_objects": {
                "mymodel": {  # registry key differs from class name
                    "object_type": "model",
                    "module_path": "reg_test_alias_pkg",
                    "module_name": "MyModel",
                }
            }
        }
        load_custom_objects(conf)
        assert "mymodel" in _MODEL_REGISTRY
        assert "MyModel" not in _MODEL_REGISTRY

    def test_module_name_optional(self, tmp_path):
        """module_name defaults to the YAML key when omitted.

        Saves repetition when the config key already matches the class name.
        """
        _make_installable_module(
            tmp_path,
            "reg_test_noname_pkg",
            "from credit.models.base_model import BaseModel\n"
            "class MyShortModel(BaseModel):\n"
            "    def __init__(self): super().__init__()\n"
            "    def forward(self, x): return x\n",
        )

        from credit.registry import load_custom_objects
        from credit.models import _MODEL_REGISTRY

        conf = {
            "custom_objects": {
                "MyShortModel": {  # no module_name → defaults to "MyShortModel"
                    "object_type": "model",
                    "module_path": "reg_test_noname_pkg",
                }
            }
        }
        load_custom_objects(conf)
        assert "MyShortModel" in _MODEL_REGISTRY

    def test_invalid_object_type_raises(self):
        from credit.registry import load_custom_objects

        conf = {
            "custom_objects": {
                "Nothing": {
                    "object_type": "scheduler",
                    "module_path": "nowhere",
                    "module_name": "Nothing",
                }
            }
        }
        with pytest.raises(ValueError, match="unsupported object_type"):
            load_custom_objects(conf)

    def test_missing_module_raises(self):
        from credit.registry import load_custom_objects

        conf = {
            "custom_objects": {
                "SomeClass": {
                    "object_type": "model",
                    "module_path": "this.module.does.not.exist",
                    "module_name": "SomeClass",
                }
            }
        }
        with pytest.raises(ImportError, match="this.module.does.not.exist"):
            load_custom_objects(conf)

    def test_missing_class_raises(self, tmp_path):
        _make_installable_module(tmp_path, "reg_test_empty_pkg", "# empty module\n")

        from credit.registry import load_custom_objects

        conf = {
            "custom_objects": {
                "NonExistentClass": {
                    "object_type": "model",
                    "module_path": "reg_test_empty_pkg",
                    "module_name": "NonExistentClass",
                }
            }
        }
        with pytest.raises(AttributeError, match="NonExistentClass"):
            load_custom_objects(conf)

    def test_wrong_base_raises(self, tmp_path):
        """TypeError from register_* propagates through load_custom_objects (tested via dataset)."""
        _make_installable_module(
            tmp_path,
            "reg_test_bad_dataset_pkg",
            "from torch.utils.data import Dataset\n"
            "class BadDataset(Dataset):\n"
            "    def __len__(self): return 0\n"
            "    def __getitem__(self, i): return {}\n",
        )

        from credit.registry import load_custom_objects

        conf = {
            "custom_objects": {
                "BadDataset": {
                    "object_type": "dataset",
                    "module_path": "reg_test_bad_dataset_pkg",
                    "module_name": "BadDataset",
                }
            }
        }
        with pytest.raises(TypeError, match="BaseDataset"):
            load_custom_objects(conf)


# ---------------------------------------------------------------------------
# register_dataset
# ---------------------------------------------------------------------------


class TestRegisterDataset:
    def test_decorator_adds_to_registry(self):
        from credit.datasets import register_dataset, _DATASET_REGISTRY

        @register_dataset("unit_test_dataset")
        class UnitTestDataset(BaseDataset):
            def __init__(self, conf, rank=0, world_size=1, is_train=True):
                pass

            def __len__(self):
                return 0

            def __getitem__(self, i):
                return {}

        assert "unit_test_dataset" in _DATASET_REGISTRY
        assert _DATASET_REGISTRY["unit_test_dataset"] is UnitTestDataset

    def test_wrong_base_raises(self):
        """register_dataset raises TypeError when class does not inherit BaseDataset."""
        from credit.datasets import register_dataset

        with pytest.raises(TypeError, match="BaseDataset"):

            @register_dataset("bad_dataset_direct")
            class BadDataset(torch.utils.data.Dataset):
                def __len__(self):
                    return 0

                def __getitem__(self, i):
                    return {}


# ---------------------------------------------------------------------------
# register_preblock
# ---------------------------------------------------------------------------


class TestRegisterPreblock:
    def test_decorator_adds_to_registry(self):
        from credit.preblock import register_preblock, _PREBLOCK_REGISTRY

        @register_preblock("unit_test_preblock")
        class UnitTestPreBlock(BasePreblock):
            def forward(self, batch):
                return batch

        assert "unit_test_preblock" in _PREBLOCK_REGISTRY
        assert _PREBLOCK_REGISTRY["unit_test_preblock"] is UnitTestPreBlock

    def test_wrong_base_raises(self):
        """register_preblock raises TypeError when class does not inherit BasePreblock."""
        from credit.preblock import register_preblock

        with pytest.raises(TypeError, match="BasePreblock"):

            @register_preblock("bad_preblock_direct")
            class BadPreBlock(nn.Module):
                def forward(self, batch):
                    return batch

    def test_build_preblocks_with_custom(self):
        """build_preblocks loads custom objects from conf and uses the registry."""
        from credit.preblock import register_preblock, build_preblocks

        @register_preblock("unit_test_pb_build")
        class UTPBBuild(BasePreblock):
            def forward(self, batch):
                return batch

        conf = {"preblocks": {"per_step": {"my_block": {"type": "unit_test_pb_build"}}}}
        modules = build_preblocks(conf, phase="per_step")
        assert "my_block" in modules
        assert isinstance(modules["my_block"], UTPBBuild)

    def test_unknown_preblock_type_friendly_error(self):
        """build_preblocks raises ValueError with a helpful message for unknown types."""
        from credit.preblock import build_preblocks

        conf = {"preblocks": {"per_step": {"bad_block": {"type": "this_type_does_not_exist"}}}}
        with pytest.raises(ValueError, match="this_type_does_not_exist"):
            build_preblocks(conf, phase="per_step")


# ---------------------------------------------------------------------------
# register_model
# ---------------------------------------------------------------------------


class TestRegisterModel:
    def test_decorator_registers_and_loads(self):
        from credit.models import register_model, load_model

        @register_model("unit_test_model")
        class UnitTestModel(BaseModel):
            def __init__(self, size=4):
                super().__init__()
                self.fc = nn.Linear(size, size)

            def forward(self, x):
                return self.fc(x)

        conf = {"model": {"type": "unit_test_model", "size": 8}, "save_loc": "/tmp"}
        model = load_model(conf)
        assert isinstance(model, UnitTestModel)
        assert model.fc.in_features == 8

    def test_overwrite_warns(self, caplog):
        from credit.models import register_model

        @register_model("unit_test_overwrite")
        class V1(BaseModel):
            pass

        with caplog.at_level(logging.WARNING, logger="credit.models"):

            @register_model("unit_test_overwrite")
            class V2(BaseModel):
                pass

        assert any("unit_test_overwrite" in m for m in caplog.messages)

    def test_wrong_base_raises(self):
        """register_model raises TypeError when class does not inherit BaseModel."""
        from credit.models import register_model

        with pytest.raises(TypeError, match="BaseModel"):

            @register_model("bad_model_direct")
            class BadModel(nn.Module):
                def forward(self, x):
                    return x


# ---------------------------------------------------------------------------
# register_postblock
# ---------------------------------------------------------------------------


class TestRegisterPostblock:
    def test_decorator_adds_to_registry(self):
        from credit.postblock import register_postblock, _POSTBLOCK_REGISTRY

        @register_postblock("unit_test_postblock")
        class UnitTestPostBlock(BasePostblock):
            def forward(self, batch):
                return batch

        assert "unit_test_postblock" in _POSTBLOCK_REGISTRY
        assert _POSTBLOCK_REGISTRY["unit_test_postblock"] is UnitTestPostBlock

    def test_wrong_base_raises(self):
        """register_postblock raises TypeError when class does not inherit BasePostblock."""
        from credit.postblock import register_postblock

        with pytest.raises(TypeError, match="BasePostblock"):

            @register_postblock("bad_postblock_direct")
            class BadPostBlock(nn.Module):
                def forward(self, batch):
                    return batch

    def test_build_postblocks_with_custom(self):
        """build_postblocks loads custom objects from conf and uses the registry."""
        from credit.postblock import register_postblock, build_postblocks

        @register_postblock("unit_test_postb_build")
        class UTPoBBuild(BasePostblock):
            def forward(self, batch):
                return batch

        conf = {"postblocks": {"per_step": {"my_block": {"type": "unit_test_postb_build"}}}}
        modules = build_postblocks(conf, phase="per_step")
        assert "my_block" in modules
        assert isinstance(modules["my_block"], UTPoBBuild)

    def test_unknown_postblock_type_friendly_error(self):
        """build_postblocks raises ValueError with a helpful message for unknown types."""
        from credit.postblock import build_postblocks

        conf = {"postblocks": {"per_step": {"bad_block": {"type": "this_type_does_not_exist"}}}}
        with pytest.raises(ValueError, match="this_type_does_not_exist"):
            build_postblocks(conf, phase="per_step")


# ---------------------------------------------------------------------------
# register_loss
# ---------------------------------------------------------------------------


class TestRegisterLoss:
    def test_decorator_adds_to_registry(self):
        from credit.losses import register_loss, _LOSS_REGISTRY

        @register_loss("unit_test_loss")
        class UnitTestLoss(nn.Module):
            def __init__(self, reduction="mean"):
                super().__init__()

            def forward(self, x, y):
                return (x - y).pow(2).mean()

        assert "unit_test_loss" in _LOSS_REGISTRY
        assert _LOSS_REGISTRY["unit_test_loss"] is UnitTestLoss

    def test_custom_loss_loaded_by_instantiate_loss(self):
        """_instantiate_loss() can instantiate a registered custom loss."""
        from credit.losses import register_loss, _instantiate_loss

        @register_loss("unit_test_loss_load")
        class LoadableLoss(nn.MSELoss):
            pass

        conf = {
            "loss": {
                "training_loss": "unit_test_loss_load",
                "training_loss_parameters": {"reduction": "mean"},
            }
        }
        loss_fn = _instantiate_loss(conf)
        assert isinstance(loss_fn, LoadableLoss)

    def test_wrong_base_raises(self):
        """register_loss raises TypeError when class does not inherit nn.Module."""
        from credit.losses import register_loss

        with pytest.raises(TypeError, match="nn.Module"):

            @register_loss("bad_loss_direct")
            class BadLoss:
                pass

    def test_unsupported_loss_raises(self):
        """_instantiate_loss() raises ValueError for an unknown loss name."""
        from credit.losses import _instantiate_loss

        conf = {"loss": {"training_loss": "this_loss_does_not_exist"}}
        with pytest.raises(ValueError, match="not supported"):
            _instantiate_loss(conf)


# ---------------------------------------------------------------------------
# load_custom_objects idempotency
# ---------------------------------------------------------------------------


class TestLoadCustomObjectsIdempotency:
    def test_double_call_no_overwrite_warning(self, tmp_path, caplog):
        """Calling load_custom_objects twice with identical conf produces no overwrite warnings."""
        _make_installable_module(
            tmp_path,
            "idempotency_test_pkg",
            "from credit.models.base_model import BaseModel\n"
            "class IdempotencyModel(BaseModel):\n"
            "    def __init__(self): super().__init__()\n"
            "    def forward(self, x): return x\n",
        )
        from credit.registry import load_custom_objects

        conf = {
            "custom_objects": {
                "IdempotencyModel": {
                    "object_type": "model",
                    "module_path": "idempotency_test_pkg",
                    "module_name": "IdempotencyModel",
                }
            }
        }
        load_custom_objects(conf)  # first call registers

        with caplog.at_level(logging.WARNING):
            load_custom_objects(conf)  # second call must do nothing

        overwrite_msgs = [m for m in caplog.messages if "overwriting" in m.lower() and "IdempotencyModel" in m]
        assert not overwrite_msgs


# ---------------------------------------------------------------------------
# load_dataloader custom BaseDataset fallback
# ---------------------------------------------------------------------------


class TestLoadDataloaderCustomDataset:
    def test_unknown_dataset_type_still_raises(self):
        """load_dataloader still raises ValueError for non-BaseDataset objects."""
        from credit.datasets.load_dataset_and_dataloader import load_dataloader

        conf = {
            "seed": 42,
            "trainer": {
                "train_batch_size": 1,
                "valid_batch_size": 1,
                "thread_workers": 0,
                "valid_thread_workers": 0,
                "prefetch_factor": 2,
                "type": "era5",
            },
            "data": {"forecast_len": 0, "valid_forecast_len": 0},
            "loss": {"training_loss": "mse"},
        }

        class NotADataset:
            pass

        with pytest.raises(ValueError, match="Unsupported dataset type"):
            load_dataloader(conf, NotADataset(), is_train=True)
