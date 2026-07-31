"""Tests for the pbsdsh-based multi-node launcher.

Two layers are covered:

* ``credit.pbs._build_pbsdsh_script`` / ``_pbsdsh_launch_block`` -- the standalone
  full-script builder and the shared launch block it delegates to.
* ``credit submit --launcher pbsdsh`` -- the integrated derecho path in
  ``credit.cli._submit``, checked side-by-side against the default mpiexec launcher.

Every test generates a script by string-building only (no filesystem or qsub side
effects), so no cluster is needed. These check the generated script has the shape we
intend; the launch mechanism itself (pbsdsh rendezvous + NCCL over libfabric) is
validated separately on real Derecho hardware.
"""

import argparse

from credit.cli import _build_pbs_script
from credit.pbs import PBSDSH_DERECHO_MODULES, _build_pbsdsh_script, _pbsdsh_launch_block

FAKE_SCRIPT_PATH = "/glade/work/user/miles-credit/credit/applications/train_gen2.py"
FAKE_CONFIG_SAVE_PATH = "/glade/derecho/scratch/user/CREDIT_runs/my_run/model.yml"


def _pbs_options(**kw):
    defaults = dict(
        nodes=4,
        ngpus=4,
        ncpus=64,
        mem="480GB",
        conda="credit-derecho",
        project="NAML0001",
        job_name="test_pbsdsh",
        walltime="02:00:00",
        queue="main",
    )
    defaults.update(kw)
    return defaults


def _script(**kw):
    return _build_pbsdsh_script(_pbs_options(**kw), FAKE_SCRIPT_PATH, FAKE_CONFIG_SAVE_PATH)


class TestPbsdshLauncher:
    def test_no_mpiexec(self):
        assert "mpiexec" not in _script()

    def test_uses_pbsdsh(self):
        assert 'pbsdsh -v -n "$i" -- bash "${node_script}"' in _script()

    def test_static_rdzv_backend(self):
        assert "--rdzv-backend=static" in _script()

    def test_no_c10d_rdzv_backend(self):
        # Static backend only -- c10d rendezvous would race across independently
        # launched pbsdsh tasks since there's no shared arrival-order negotiation.
        assert "--rdzv-backend=c10d" not in _script()

    def test_node_rank_uses_loop_index(self):
        assert "--node-rank=${i}" in _script()

    def test_master_addr_and_port_flags_present(self):
        script = _script()
        assert "--master-addr=${MASTER_ADDR}" in script
        assert "--master-port=${MASTER_PORT}" in script

    def test_master_addr_resolved_locally_not_via_ssh(self):
        # pbsdsh index 0 == mother superior == the host running this script, so MASTER_ADDR
        # must be this host (resolved locally), NOT a possibly-reordered nodefile entry via
        # ssh -- otherwise rank 0's rendezvous store binds on a host rank 0 doesn't run on.
        script = _script()
        assert "MASTER_ADDR=$(hostname -i | awk '{print $1}')" in script
        assert "ssh" not in script

    def test_node_count_from_nodefile(self):
        # Node count is derived from the deduplicated nodefile at runtime, not hardcoded.
        script = _script()
        assert 'NUM_NODES=$(sort -u "$PBS_NODEFILE" | wc -l)' in script
        assert "--nnodes=${NUM_NODES}" in script

    def test_select_line_correct(self):
        script = _script(nodes=8, ngpus=4, ncpus=64, mem="480GB")
        assert "select=8:ncpus=64:ngpus=4:mem=480GB" in script

    def test_nproc_per_node_matches_gpus(self):
        assert "--nproc-per-node=2" in _script(ngpus=2)

    def test_cuda_visible_devices_matches_gpu_count(self):
        assert "CUDA_VISIBLE_DEVICES=0,1,2" in _script(ngpus=3)

    def test_account_in_header(self):
        assert "#PBS -A NAML0001" in _script(project="NAML0001")

    def test_walltime_in_header(self):
        assert "#PBS -l walltime=03:00:00" in _script(walltime="03:00:00")

    def test_queue_in_header(self):
        assert "#PBS -q main" in _script(queue="main")

    # -- NCCL over libfabric --------------------------------------------------

    def test_libfabric_module_loaded(self):
        # The user requirement: NCCL runs over the libfabric module. Its version is matched
        # to the aws-ofi-nccl plugin's build (Cray libfabric 1.15.2.0).
        script = _script()
        assert "libfabric/1.15.2.0" in script
        assert "libfabric/1.15.2.0" in PBSDSH_DERECHO_MODULES

    def test_cray_mpich_still_loaded_for_libtorch_runtime(self):
        # cray-mpich is dropped from the *launcher* (no mpiexec) but its runtime libs are
        # still needed: libtorch links Cray's libmpi_gnu_123.so. Dropping it breaks import.
        assert "cray-mpich/8.1.29" in _script()

    def test_nccl_debug_surfaces_transport(self):
        # NCCL_DEBUG=INFO so the run log confirms libfabric/OFI rather than a silent socket
        # fallback.
        assert "export NCCL_DEBUG=INFO" in _script()

    # -- environment baking ---------------------------------------------------

    def test_conda_activated_once_in_header(self):
        # conda activate runs once, in the outer (mother-superior) shell; the per-node
        # scripts inherit its result via baked PATH/LD_LIBRARY_PATH, not by re-activating.
        script = _script(conda="credit-derecho")
        assert script.count("conda activate credit-derecho") == 1

    def test_per_node_script_bakes_resolved_env(self):
        # pbsdsh's spawned shell doesn't inherit modules/conda, so the mother superior's
        # fully-resolved PATH/LD_LIBRARY_PATH are captured and baked into each per-node script.
        script = _script()
        assert 'RESOLVED_PATH="$PATH"' in script
        assert 'RESOLVED_LD="${LD_LIBRARY_PATH:-}"' in script
        assert 'export PATH="${RESOLVED_PATH}"' in script
        assert 'export LD_LIBRARY_PATH="${RESOLVED_LD}"' in script

    def test_no_module_or_conda_inside_per_node_heredoc(self):
        # A live attempt found `module`/`conda` both unavailable in pbsdsh's spawned shell.
        # The per-node script (everything inside the heredoc) must not depend on either.
        script = _script()
        _, _, after = script.partition('cat > "${node_script}" <<EOF')
        node_body, _, _ = after.partition("\nEOF\n")
        assert "conda activate" not in node_body
        assert "module load" not in node_body

    # -- torchrun resolution --------------------------------------------------

    def test_torchrun_resolved_from_conda_env_path(self):
        # Absolute conda path -> explicit torchrun path baked in.
        script = _script(conda="/glade/work/user/conda-envs/credit-derecho")
        assert "/glade/work/user/conda-envs/credit-derecho/bin/torchrun" in script

    def test_torchrun_from_path_when_conda_name(self):
        # Name-based conda -> bare `torchrun` (found via the baked PATH after activation).
        script = _script(conda="credit-derecho")
        assert "\ntorchrun \\" in script or "\ntorchrun " in script

    def test_config_save_path_in_script(self):
        assert FAKE_CONFIG_SAVE_PATH in _script()

    def test_training_script_path_in_script(self):
        assert FAKE_SCRIPT_PATH in _script()

    # -- robustness / regression guards --------------------------------------

    def test_per_node_command_is_a_written_script_file_not_inline_dash_c(self):
        # A first live attempt passed a multi-line `bash -l -c "<...>"` string to pbsdsh;
        # every node exited status 0 with zero output because tm_spawn dropped everything
        # after the first line. Each node's command must be written to its own file.
        script = _script()
        assert "-- bash -l -c" not in script
        assert 'cat > "${node_script}" <<EOF' in script
        assert 'chmod +x "${node_script}"' in script

    def test_per_node_script_heredoc_is_unquoted_for_immediate_expansion(self):
        # Unquoted <<EOF (not <<'EOF') so the outer script bakes fully-resolved literal
        # values (RESOLVED_PATH, MASTER_ADDR, node-rank, ...) into each per-node script.
        script = _script()
        assert "<<EOF" in script
        assert "<<'EOF'" not in script

    def test_waits_on_all_pids_and_propagates_failure(self):
        script = _script()
        assert "pids+=($!)" in script
        assert 'wait "$pid" || status=1' in script
        assert "exit $status" in script

    def test_per_node_logs_written_and_dumped(self):
        # torchrun output is redirected to node_<i>.out by the remote node itself (PBS
        # routes a pbsdsh task's stdout to the job stream, not the launcher's redirect), and
        # pbsdsh's -v status goes to node_<i>.pbsdsh. Both land on disk and are dumped at end.
        script = _script()
        assert "LOGDIR=" in script
        assert 'mkdir -p "$LOGDIR"' in script
        assert '> "${LOGDIR}/node_${i}.out" 2>&1' in script
        assert '> "${LOGDIR}/node_${i}.pbsdsh" 2>&1 &' in script
        assert 'cat "${LOGDIR}/node_${i}.out"' in script

    def test_defaults_when_pbs_options_sparse(self):
        # Only nodes/ngpus provided -- everything else should fall back to defaults.
        script = _build_pbsdsh_script({"nodes": 1, "ngpus": 4}, FAKE_SCRIPT_PATH, FAKE_CONFIG_SAVE_PATH)
        assert "#PBS -A NAML0001" in script
        assert "#PBS -N credit_v2_pbsdsh" in script
        assert "#PBS -l walltime=01:00:00" in script
        assert "select=1:ncpus=64:ngpus=4:mem=480GB" in script
        assert "conda activate credit" in script

    def test_launch_block_terminator_at_column_zero(self):
        # The per-node heredoc terminator must not be indented, or the heredoc never closes.
        block = _pbsdsh_launch_block("torchrun", FAKE_SCRIPT_PATH, FAKE_CONFIG_SAVE_PATH, 4, "/tmp")
        assert "\nEOF\n" in block


# ===========================================================================
# credit submit --launcher pbsdsh  vs  --launcher mpiexec  (derecho, multi-node)
# ===========================================================================

FAKE_CONFIG = "/glade/work/user/my_run/config.yml"
FAKE_REPO = "/glade/work/user/miles-credit"
FAKE_ACCOUNT = "NAML0001"


def _derecho_args(launcher="mpiexec", nodes=2, gpus=4, **kw):
    defaults = dict(
        cluster="derecho",
        nodes=nodes,
        gpus=gpus,
        cpus=None,
        mem=None,
        walltime="12:00:00",
        queue=None,
        conda_env=None,
        launcher=launcher,
    )
    defaults.update(kw)
    return argparse.Namespace(**defaults)


def _submit_script(launcher, nodes=2, **kw):
    return _build_pbs_script(_derecho_args(launcher=launcher, nodes=nodes, **kw), FAKE_CONFIG, FAKE_REPO, FAKE_ACCOUNT)


class TestSubmitLauncherSelection:
    def test_pbsdsh_selected_for_multinode(self):
        script = _submit_script("pbsdsh", nodes=2)
        assert 'pbsdsh -v -n "$i"' in script
        assert "mpiexec" not in script

    def test_mpiexec_is_default_for_multinode(self):
        script = _submit_script("mpiexec", nodes=2)
        assert "mpiexec" in script
        assert "pbsdsh" not in script

    def test_single_node_ignores_launcher_and_uses_standalone(self):
        # pbsdsh only changes the multi-node path; single-node always uses --standalone.
        for launcher in ("pbsdsh", "mpiexec"):
            script = _submit_script(launcher, nodes=1)
            assert "--standalone" in script
            assert "pbsdsh" not in script
            assert "mpiexec" not in script

    def test_both_launchers_target_train_entrypoint(self):
        for launcher in ("pbsdsh", "mpiexec"):
            assert "credit/applications/train_gen2.py" in _submit_script(launcher)

    def test_both_launchers_share_select_line(self):
        pbsdsh = _submit_script("pbsdsh", nodes=2, gpus=4)
        mpiexec = _submit_script("mpiexec", nodes=2, gpus=4)
        assert "select=2:" in pbsdsh
        assert "select=2:" in mpiexec

    def test_pbsdsh_path_loads_libfabric_mpiexec_does_not(self):
        # The launcher-neutral mpiexec path relies on cray-mpich's bundled libfabric; the
        # pbsdsh path loads the libfabric module explicitly (the user requirement).
        assert "libfabric" in _submit_script("pbsdsh", nodes=2)
        assert "libfabric" not in _submit_script("mpiexec", nodes=2)

    def test_pbsdsh_uses_torchrun_mpiexec_uses_python(self):
        assert "torchrun" in _submit_script("pbsdsh", nodes=2)
        assert "mpiexec" in _submit_script("mpiexec", nodes=2)
