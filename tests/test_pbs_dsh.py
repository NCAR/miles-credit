"""Tests for the pbsdsh-based multi-node launcher prototype in credit/pbs.py.

Every test generates a script via _build_pbsdsh_script directly (a pure string builder,
no filesystem or qsub side effects), so no cluster is needed.

This launcher is a prototype meant to be tried alongside the existing mpiexec-based
launch_script_torchrun on real Derecho hardware — see _build_pbsdsh_script's docstring
for the assumptions that still need to be confirmed there. These tests only check that
the generated script has the shape we intend, not that it actually works on a cluster.
"""

from credit.pbs import _build_pbsdsh_script


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
        assert "pbsdsh -v -n" in _script()

    def test_static_rdzv_backend(self):
        assert "--rdzv-backend=static" in _script()

    def test_no_c10d_rdzv_backend(self):
        # Static backend only — c10d rendezvous would race across independently
        # launched pbsdsh tasks since there's no shared arrival-order negotiation.
        assert "--rdzv-backend=c10d" not in _script()

    def test_node_rank_uses_loop_index(self):
        assert "--node-rank=$i" in _script()

    def test_master_addr_and_port_flags_present(self):
        script = _script()
        assert "--master-addr=${MASTER_ADDR}" in script
        assert "--master-port=${MASTER_PORT}" in script

    def test_head_node_ip_lookup(self):
        assert "hostname -i" in _script()

    def test_select_line_correct(self):
        script = _script(nodes=8, ngpus=4, ncpus=64, mem="480GB")
        assert "select=8:ncpus=64:ngpus=4:mem=480GB" in script

    def test_nnodes_matches_config(self):
        assert "NUM_NODES=8" in _script(nodes=8)

    def test_nproc_per_node_matches_gpus(self):
        assert "NUM_GPUS=2" in _script(ngpus=2)

    def test_cuda_visible_devices_matches_gpu_count(self):
        script = _script(ngpus=3)
        assert "CUDA_VISIBLE_DEVICES=0,1,2" in script

    def test_account_in_header(self):
        assert "#PBS -A NAML0001" in _script(project="NAML0001")

    def test_walltime_in_header(self):
        assert "#PBS -l walltime=03:00:00" in _script(walltime="03:00:00")

    def test_queue_in_header(self):
        assert "#PBS -q main" in _script(queue="main")

    def test_nccl_net_ofi_set(self):
        # NCCL communicates over libfabric/CXI regardless of launcher; this is set
        # explicitly so removing mpiexec doesn't accidentally change NCCL's transport.
        assert "NCCL_NET=OFI" in _script()

    def test_conda_activate_in_header_and_per_node_task(self):
        script = _script(conda="credit-derecho")
        # Once for the job script itself, once inside each pbsdsh-spawned task as a
        # safeguard in case pbsdsh doesn't inherit the job script's environment.
        assert script.count("conda activate credit-derecho") == 2

    def test_module_load_repeated_inside_per_node_script_for_name_based_conda(self):
        # Name-based pbs.conda (no "/") falls back to module+activate inside the per-node
        # script too, appearing once for the outer script and once inside the per-node
        # heredoc. NOTE: a live attempt found pbsdsh's spawned shell has neither 'module'
        # nor 'conda' available at all, so this fallback is not expected to actually work —
        # see test_path_based_conda_bypasses_module_system below for the fix that does.
        script = _script(conda="credit-derecho")
        assert script.count("module load ncarenv/24.12 nvhpc cuda/12.3.2 conda") == 2

    def test_path_based_conda_bypasses_module_system(self):
        # Regression guard: two live attempts confirmed 'module' and 'conda' are both
        # unavailable in pbsdsh's spawned login shell ("command not found" for each), so
        # nothing depending on the module system can work there. When pbs.conda is an
        # absolute path, PATH/LD_LIBRARY_PATH must be set directly instead — this is what
        # 'conda activate' would have done anyway, and needs no module/conda command at all.
        conda_path = "/glade/campaign/cisl/aiml/credit/conda_envs/credit-derecho"
        script = _script(conda=conda_path)
        assert f'export PATH="{conda_path}/bin:$PATH"' in script
        assert f'export LD_LIBRARY_PATH="{conda_path}/lib:$LD_LIBRARY_PATH"' in script
        # Only the outer script's own module load remains; the per-node script skips it.
        assert script.count("module load ncarenv/24.12 nvhpc cuda/12.3.2 conda") == 1

    def test_torchrun_resolved_from_conda_env_path(self):
        script = _script(conda="/glade/work/user/conda-envs/credit-derecho")
        assert "TORCHRUN=/glade/work/user/conda-envs/credit-derecho/bin/torchrun" in script

    def test_torchrun_resolved_from_conda_env_name(self):
        script = _script(conda="credit-derecho")
        assert "TORCHRUN=$(conda info --base)/envs/credit-derecho/bin/torchrun" in script

    def test_config_save_path_in_script(self):
        assert FAKE_CONFIG_SAVE_PATH in _script()

    def test_training_script_path_in_script(self):
        assert FAKE_SCRIPT_PATH in _script()

    def test_per_node_command_is_a_written_script_file_not_inline_dash_c(self):
        # Regression guard: a first live attempt passed a multi-line `bash -l -c "<...>"`
        # string directly to pbsdsh. Every node exited status 0 with zero output — pbsdsh's
        # tm_spawn does not reliably preserve a single quoted argument containing literal
        # newlines, so only the first line silently ran. Each node's command must instead be
        # written to its own plain script file and executed by path.
        script = _script()
        assert "-- bash -l -c" not in script
        assert 'cat > "${node_script}" <<EOF' in script
        assert 'chmod +x "${node_script}"' in script
        assert 'pbsdsh -v -n "$i" -- bash "${node_script}"' in script

    def test_per_node_script_heredoc_is_unquoted_for_immediate_expansion(self):
        # The heredoc delimiter must be unquoted (<<EOF, not <<'EOF') so the outer script
        # bakes in fully-resolved literal values (MASTER_ADDR, TORCHRUN, node-rank, ...)
        # rather than leaving variable references that depend on pbsdsh propagating shell
        # variables (as opposed to exported env vars) to the remote task.
        script = _script()
        assert "<<EOF" in script
        assert "<<'EOF'" not in script
        assert "--node-rank=$i" in script

    def test_waits_on_all_pids_and_propagates_failure(self):
        script = _script()
        assert "pids+=($!)" in script
        assert 'wait "$pid" || status=1' in script
        assert "exit $status" in script

    def test_per_node_logs_written_and_dumped(self):
        # Diagnostic output must land on disk independently of the job's own stdout stream,
        # so a job that gets killed abruptly (as the first live attempt was) still leaves logs.
        script = _script()
        assert "LOGDIR=" in script
        assert 'mkdir -p "${LOGDIR}"' in script
        assert '> "${LOGDIR}/node_${i}.log" 2>&1 &' in script
        assert 'cat "${LOGDIR}/node_${i}.log"' in script

    def test_defaults_when_pbs_options_sparse(self):
        # Only nodes/ngpus provided — everything else should fall back to defaults
        # rather than raising a KeyError.
        script = _build_pbsdsh_script({"nodes": 1, "ngpus": 4}, FAKE_SCRIPT_PATH, FAKE_CONFIG_SAVE_PATH)
        assert "#PBS -A NAML0001" in script
        assert "#PBS -N credit_v2_pbsdsh" in script
        assert "#PBS -l walltime=01:00:00" in script
        assert "select=1:ncpus=64:ngpus=4:mem=480GB" in script
        assert "conda activate credit" in script
