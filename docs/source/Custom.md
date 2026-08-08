# Custom Objects

CREDIT is built around a **registry pattern**: models, datasets, preblocks,
postblocks, losses, and metrics are all selected purely by a string key in the
config (`model.type`, `data.source.<name>.dataset_type`, a preblock's `type`, and
so on). This same mechanism lets you plug in **your own** classes without editing
or forking CREDIT's source code — you write a class, point the config at it, and
it behaves exactly like a built-in type.

This page explains how that works and how to add your own objects.

**AutoAPI:** {py:obj}`credit.registry.load_custom_objects`

## When to use custom objects

Reach for a custom object whenever the behavior you need isn't covered by a
built-in type — a dataset for a data source CREDIT doesn't ship, a new model
architecture, a bespoke preprocessing step, a physics constraint, or a special
loss or metric. You get the full CREDIT training/inference/checkpointing pipeline
for free; you only supply the one piece that is new.

There are two ways a class enters a registry:

1. **Inside the CREDIT repo** — decorate the class with the matching
   `@register_*` decorator. Use this when you are contributing a new built-in
   type back to CREDIT.
2. **From your own external package** — list the class under `custom_objects:`
   in your config. Use this when you want to keep your code in your own project
   and never touch CREDIT's source. **This is the common case and the focus of
   this guide.**

## The six object types

Every custom class must subclass the CREDIT base class for its type. The base
class fixes the method contract the rest of the pipeline relies on.

| `object_type` | Base class to subclass | Contract |
|---|---|---|
| `dataset`   | {py:obj}`credit.datasets.gen_2.base_dataset.BaseDataset` | yields the nested `input`/`target`/`metadata` sample dict |
| `preblock`  | {py:obj}`credit.preblock.base.BasePreblock`   | `forward(batch: dict) -> dict` |
| `model`     | {py:obj}`credit.models.base_model.BaseModel`  | `forward(x) -> prediction` |
| `postblock` | {py:obj}`credit.postblock.base.BasePostblock` | `forward(batch: dict) -> dict` |
| `loss`      | `torch.nn.Module` | `forward(pred, target) -> loss` |
| `metric`    | {py:obj}`credit.metrics.base.BaseVariableMetric` (recommended) | called as `metric(full_data_dict)` |

Registration **validates the base class** and raises `TypeError` if it does not
match — so a wrong base class fails loudly at startup, not deep in training.

For losses and metrics the hard requirement is only `torch.nn.Module`, but a
metric is invoked as `metric(full_data_dict)`; subclass `BaseVariableMetric`
unless your class handles that calling convention itself.

## Adding a custom object from your own package

### 1. Write the class

Subclass the appropriate base class in your own installable package:

```python
# mypackage/preblock.py
from credit.preblock.base import BasePreblock

class MyPreBlock(BasePreblock):
    def __init__(self, fill_value: float = 0.0):
        super().__init__()
        self.fill_value = fill_value

    def forward(self, batch: dict) -> dict:
        batch = self._copy_batch(batch)   # never mutate the caller's dict
        # ... your transform ...
        return batch
```

### 2. Make it importable

`module_path` in the config is a **dotted Python import path**
(`mypackage.preblock`), *not* a file path. The package must import cleanly from
your Python environment. If it isn't installed yet, run this from the directory
containing your code:

```bash
pip install -e .
```

### 3. Declare it under `custom_objects:`

Add a `custom_objects:` block at the top level of your config. Each entry maps a
**registry key** (the dict key) to the class to import:

```yaml
custom_objects:
  MyPreBlock:                    # registry key — how you'll refer to it below
    object_type: preblock        # which registry to add it to
    module_path: mypackage.preblock   # dotted import path (not a file path)
    # module_name: MyPreBlock    # optional; defaults to the key above
```

Fields:

- **`object_type`** *(required)* — one of `dataset`, `preblock`, `model`,
  `postblock`, `loss`, `metric`.
- **`module_path`** *(required)* — the dotted Python module to import the class
  from.
- **`module_name`** *(optional)* — the Python class name. It **defaults to the
  registry key**, so you only need it when the key and the class name differ
  (e.g. key `mymodel`, class `MyModel`).

### 4. Reference it by key elsewhere in the config

Once registered, the key works exactly like a built-in type. Keys are
**case-sensitive** and must match exactly:

```yaml
preblocks:
  per_step:
    my_norm:
      type: MyPreBlock           # the custom_objects key
      args:
        fill_value: 0.0          # passed to MyPreBlock.__init__
```

## Where each type is referenced

The registry key is used in a different place depending on the object type:

```yaml
# dataset → data.source.<name>.dataset_type
data:
  source:
    MySource:
      dataset_type: MyDataset

# model → model.type
model:
  type: mymodel

# preblock → inside preblocks.ic_only or preblocks.per_step
preblocks:
  per_step:
    my_norm:
      type: MyPreBlock

# postblock → inside postblocks.per_step or postblocks.post_rollout
postblocks:
  per_step:
    my_post:
      type: MyPostBlock

# loss → an entire custom loss replaces "base" in loss.type ...
loss:
  type: MyLoss

# ... or a custom *univariate* loss (applied per variable by BaseLoss) goes in args
loss:
  type: base
  args:
    training_loss: MyLoss        # same for validation_loss

# metric → one entry in a combined group, or on its own
metrics:
  type: combined
  args:
    metrics: {rmse: {}, MyMetric: {}}
```

## Full example — all six types

```yaml
custom_objects:

  MyDataset:                 # key == class name → module_name omitted
    object_type: dataset
    module_path: mypackage.data

  MyPreBlock:
    object_type: preblock
    module_path: mypackage.preblock

  mymodel:                   # key differs from class name → module_name required
    object_type: model
    module_path: mypackage.models
    module_name: MyModel

  MyPostBlock:
    object_type: postblock
    module_path: mypackage.postblock

  MyLoss:
    object_type: loss
    module_path: mypackage.losses

  MyMetric:
    object_type: metric
    module_path: mypackage.metrics
```

## Validate before you train

`credit check -c config.yml` imports and registers every `custom_objects` entry
first, then resolves all the type keys — so a misspelled key, an uninstalled
package, a wrong `module_name`, or a base-class mismatch is reported up front
instead of failing minutes into a run. Always run it after editing
`custom_objects`:

```bash
credit check -c config.yml
```

## Gotchas

- **Don't shadow built-ins.** CREDIT's built-in types use snake_case (`unet`,
  `log_transform`, `local`). Registering a key that already exists overwrites the
  built-in (with a warning). Pick a distinct key.
- **`module_path` is an import path, not a file path.** Use `mypackage.models`,
  not `mypackage/models.py` or `/abs/path/models.py`.
- **Keys are case-sensitive.** `MyPreBlock` and `mypreblock` are different keys;
  the reference elsewhere in the config must match exactly.
- **The package must be installed** in the same environment that runs `credit`
  (`pip install -e .`), or the import fails.
- **Subclass the right base class.** Registration enforces it; a mismatch raises
  `TypeError` at startup.

## Contributing a new built-in type

If instead you are adding a type directly to the CREDIT source, register it with
the matching decorator rather than `custom_objects:` — the base-class validation
is identical:

```python
from credit.preblock import register_preblock
from credit.preblock.base import BasePreblock

@register_preblock("my_preblock")
class MyPreBlock(BasePreblock):
    def forward(self, batch: dict) -> dict:
        ...
```

The equivalent decorators are `register_dataset`, `register_model`,
`register_postblock`, `register_loss`, and `register_metric`, each importable
from its package (`credit.datasets`, `credit.models`, …). When you add a
built-in type, also update the matching validation in `credit check` and the
relevant doc page.
