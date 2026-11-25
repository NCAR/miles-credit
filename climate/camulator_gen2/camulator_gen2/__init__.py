"""CREDIT custom objects that reproduce the Hugging Face CAMulator in the gen2 pipeline.

The released checkpoints (``huggingface.co/willychap/camulator``) were trained
and run with the gen1 code on the ``WillyChap/miles-credit@camulator_huggingface``
branch. The gen2 pipeline already covers most of that setup; these objects add
the parts it does not have, without changing CREDIT itself:

- :class:`~camulator_gen2.postblocks.HFTracerFixer`: the gen1 tracer fixer, in
  normalized space and in the same place in the chain, optionally keeping its
  level-0 Qtot statistics bug.
- :class:`~camulator_gen2.postblocks.HFWaterFixer`: the gen2 water fixer that
  also returns the gen1 trainer's water-conservation penalty.
- :class:`~camulator_gen2.losses.HFLoss`: the gen1 latitude-weighted MSE,
  scored where the gen1 trainer scored it, plus the penalty.

Register them through the config's ``custom_objects`` block; see
``configs/camulator_hf.yml`` and ``README.md``.
"""

__version__ = "0.1.0"
