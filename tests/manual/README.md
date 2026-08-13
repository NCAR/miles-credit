# Manual tests

Tests in this directory are **not** collected or run by GitHub CI. They require
resources the CI runners do not have: multiple GPUs, MPI/NCCL, HPC schedulers
(PBS on Casper/Derecho), or large shared datasets on `/glade`.

CI excludes this tree via `norecursedirs` in `pyproject.toml`
(`[tool.pytest.ini_options]`). Run these by hand on the appropriate cluster.

## Layout

- `gen2_parallelism/` — gen2 (V2) parallelism smoke matrix: FSDP2, domain, tensor,
  DDP and their combinations. Submitted with `qsub` on Derecho. See its own README.
