import importlib
import logging

logger = logging.getLogger(__name__)

# Maps each object_type to (module_path, register_fn_name).
# module_path: where the register_* function lives (each package's __init__.py).
#   Imported only when load_custom_objects is called, not when this file is imported,
#   so that importing credit.registry does not automatically load all five packages.
# register_fn_name: the decorator that validates the base class and stores the class in the registry.
# Adding a new type here is all that's needed — validation and dispatch both use this dict.
_REGISTRY_MAP = {
    "dataset": ("credit.datasets", "register_dataset"),
    "preblock": ("credit.preblock", "register_preblock"),
    "model": ("credit.models", "register_model"),
    "postblock": ("credit.postblock", "register_postblock"),
    "loss": ("credit.losses", "register_loss"),
}

# Tracks what has already been registered so repeated calls with the same conf
# (e.g. from build_preblocks and build_postblocks in the same training run) do nothing.
_REGISTERED_ENTRIES: set = set()


def load_custom_objects(conf):
    """Load and register custom classes listed under ``custom_objects`` in the config.

    Each entry names a Python class to import (via ``module_path`` and
    ``module_name``) and which registry to add it to (via ``object_type``).
    Once registered, the class is available in the rest of the config under the
    dict key, exactly like any built-in type.

    ``module_path`` must be a dotted Python import path (e.g. ``mypackage.models``).
    The package must be importable from your Python environment — run
    ``pip install -e .`` in the directory containing your custom code if it is
    not already installed.

    ``module_name`` is the Python class name inside the module.  It defaults to
    the dict key, so you only need it when the registry key and the class name
    differ (e.g. key ``mymodel``, class ``MyModel``).

    This function is called internally by ``load_model``, ``build_preblocks``,
    ``build_postblocks``, and others, so it runs multiple times per training run.
    Already-registered entries are silently skipped — idempotency is tracked via
    an internal set keyed on ``(object_type, module_path, registry_name)``.

    ``_REGISTRY_MAP`` is the single source of truth for supported object types.
    It maps each ``object_type`` to the module and ``register_*`` function that
    handles it.  Adding a new type requires only one line there — validation and
    dispatch both use the same dict.

    Each custom class must inherit from the corresponding CREDIT base class,
    or registration will raise a ``TypeError``:

    - ``credit.datasets.gen_2.base_dataset.BaseDataset`` for datasets
    - ``credit.preblock.base.BasePreblock`` for preblocks
    - ``credit.models.base_model.BaseModel`` for models
    - ``credit.postblock.base.BasePostblock`` for postblocks
    - ``torch.nn.Module`` for losses

    Note: CREDIT's built-in types use snake_case (``unet``, ``log_transform``).
    Use a distinct key to avoid accidentally overwriting a built-in.

    Full example — all five object types::

        # ---- Register custom classes ----
        custom_objects:

          MyDataset:              # key == class name, so module_name can be omitted
            object_type: dataset
            module_path: mypackage.data   # dotted Python import path, not a file path

          MyPreBlock:
            object_type: preblock
            module_path: mypackage.preblock

          mymodel:                # key differs from class name → module_name required
            object_type: model
            module_path: mypackage.models
            module_name: MyModel

          MyPostBlock:
            object_type: postblock
            module_path: mypackage.postblock

          MyLoss:
            object_type: loss
            module_path: mypackage.losses

        # ---- Use the registered classes elsewhere in the config ----

        # dataset: referenced under data.source.<name>.dataset_type
        # Must match the custom_objects key exactly (case-sensitive), same as built-in
        # source types (e.g. "local", "arco_era5").
        data:
          source:
            MySource:
              dataset_type: MyDataset
              # ... other data config ...

        # preblock: placed inside preblocks.ic_only or preblocks.per_step
        preblocks:
          per_step:
            my_norm:
              type: MyPreBlock
              args:
                fill_value: 0.0

        # model: referenced as model.type
        model:
          type: mymodel

        # postblock: placed inside postblocks.per_step or postblocks.post_rollout
        postblocks:
          per_step:
            my_post:
              type: MyPostBlock

        # loss: referenced as loss.training_loss
        loss:
          training_loss: MyLoss

    Args:
        conf (dict): Top-level config dict. If ``custom_objects`` is absent or
            empty this function does nothing.

    Raises:
        KeyError: If a required field (``object_type`` or ``module_path``) is
            missing from an entry.
        ValueError: If ``object_type`` is not one of the supported values.
        ImportError: If the module at ``module_path`` cannot be imported.
        AttributeError: If the class named ``module_name`` is not in the module.
        TypeError: If the class does not inherit from the required base class.
    """
    # conf["custom_objects"] may be absent, None, or an empty dict — all are skipped.
    entries = conf.get("custom_objects") or {}
    if not entries:
        return

    for registry_name, entry in entries.items():
        object_type = entry["object_type"]
        module_path = entry["module_path"]
        # module_name is the Python class name; defaults to the dict key when omitted.
        module_name = entry.get("module_name") or registry_name

        # Skip entries we have already successfully registered in this process lifetime.
        _key = (object_type, module_path, registry_name)
        if _key in _REGISTERED_ENTRIES:
            continue

        # Validate object_type before doing any I/O.
        if object_type not in _REGISTRY_MAP:
            raise ValueError(
                f"custom_objects: unsupported object_type {object_type!r}. Must be one of: {sorted(_REGISTRY_MAP)}"
            )

        # Dynamically import the user's module by its dotted Python path.
        # The package must be importable, e.g. via `pip install -e .` in the
        # directory containing the custom code.
        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:
            raise ImportError(
                f"custom_objects: cannot import module {module_path!r} "
                f"for {object_type} {registry_name!r}. "
                "Make sure the package is installed in your Python environment."
            ) from exc

        # Pull the class out of the module by its Python class name.
        try:
            cls = getattr(module, module_name)
        except AttributeError as exc:
            raise AttributeError(f"custom_objects: class {module_name!r} not found in module {module_path!r}.") from exc

        # Look up and call the correct register_* function from the dispatch map.
        # The register_* decorator validates that cls inherits the required base class.
        # Show the class name in the log only when it differs from the registry key.
        label = registry_name if registry_name == module_name else f"{registry_name} ({module_name})"
        reg_module_path, reg_func_name = _REGISTRY_MAP[object_type]
        register_fn = getattr(importlib.import_module(reg_module_path), reg_func_name)
        register_fn(registry_name)(cls)
        _REGISTERED_ENTRIES.add(_key)
        logger.info(f"Registered custom {object_type} {label!r}")
