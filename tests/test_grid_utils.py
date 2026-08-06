"""Tests for credit.datasets.gen_2.grid_utils.GridSchema persistence.

Focus: concurrent writes of the same schema file. Every rank and every
DataLoader worker that resolves a grid persists it (see
``write_source_grid_schema_if_missing``), so the same path is routinely written
by several processes at once.
"""

import multiprocessing as mp
import os

import numpy as np
import pytest
import xarray as xr

from credit.datasets.gen_2.grid_utils import (
    SOURCE_GRID_SCHEMA_FILENAME,
    GridSchema,
    write_source_grid_schema_if_missing,
)

LAT = np.linspace(-90.0, 90.0, 19)
LON = np.linspace(0.0, 350.0, 36)


def _schema():
    return GridSchema("rectilinear", LAT, LON)


def _tmp_leftovers(directory):
    return [n for n in os.listdir(directory) if ".tmp" in n]


class TestGridSchemaSave:
    def test_round_trip(self, tmp_path):
        path = str(tmp_path / "ERA5_grid_schema.nc")
        _schema().save(path)

        loaded = GridSchema.load(path)
        assert loaded.grid_type == "rectilinear"
        np.testing.assert_allclose(loaded.lat, LAT)
        np.testing.assert_allclose(loaded.lon, LON)
        assert _tmp_leftovers(tmp_path) == []

    def test_staging_file_is_per_process(self, tmp_path, monkeypatch):
        """The temp name must be unique per process, not a shared ``<path>.tmp``.

        A shared name made concurrent writers collide inside HDF5, which
        surfaced as ``[Errno 13] Permission denied`` for every writer but the
        first. Here the shared name is occupied by a directory (unwritable as a
        file) — the save must not touch it.
        """
        path = str(tmp_path / "ERA5_grid_schema.nc")
        os.mkdir(path + ".tmp")

        _schema().save(path)

        assert os.path.isfile(path)
        assert os.path.isdir(path + ".tmp")  # untouched
        assert GridSchema.load(path).grid_type == "rectilinear"

    def test_failed_write_leaves_no_temp_file(self, tmp_path, monkeypatch):
        path = str(tmp_path / "ERA5_grid_schema.nc")

        def boom(*args, **kwargs):
            raise RuntimeError("disk on fire")

        monkeypatch.setattr(xr.Dataset, "to_netcdf", boom)
        with pytest.raises(RuntimeError, match="disk on fire"):
            _schema().save(path)

        assert not os.path.exists(path)
        assert _tmp_leftovers(tmp_path) == []


def _worker(save_loc, barrier, errors):
    """Persist the same source grid from a separate process, at the same time."""
    try:
        barrier.wait(timeout=30)
        write_source_grid_schema_if_missing("ERA5", {"grid_type": "rectilinear", "lat": LAT, "lon": LON}, save_loc)
        # write_source_grid_schema_if_missing swallows failures by design, so
        # assert the post-condition it is supposed to have reached.
        path = os.path.join(save_loc, SOURCE_GRID_SCHEMA_FILENAME.format(source="ERA5"))
        if not os.path.isfile(path):
            errors.append(f"pid {os.getpid()}: no schema at {path}")
    except Exception as exc:  # pragma: no cover - reported through `errors`
        errors.append(f"pid {os.getpid()}: {exc!r}")


class TestConcurrentSave:
    def test_four_processes_write_the_same_schema(self, tmp_path):
        """Mirrors 4 DDP ranks fitting scalers: all must succeed, file stays valid."""
        ctx = mp.get_context("fork")
        n = 4
        barrier = ctx.Barrier(n)
        with ctx.Manager() as manager:
            errors = manager.list()
            procs = [ctx.Process(target=_worker, args=(str(tmp_path), barrier, errors)) for _ in range(n)]
            for p in procs:
                p.start()
            for p in procs:
                p.join(timeout=120)

            assert all(p.exitcode == 0 for p in procs), [p.exitcode for p in procs]
            assert list(errors) == []

        path = str(tmp_path / "ERA5_grid_schema.nc")
        loaded = GridSchema.load(path)
        np.testing.assert_allclose(loaded.lat, LAT)
        np.testing.assert_allclose(loaded.lon, LON)
        assert _tmp_leftovers(tmp_path) == []
