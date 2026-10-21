# HPC batch execution (optional)

For large samples the nested-sampling fits can be dispatched as a SLURM array
job. **The general user does not need anything in this directory** — `granalyse
run config.yaml` performs the same fits locally.

Workflow (run from a directory of your choice on the cluster, with the
Granalyse venv, e.g. `gvenv`, active):

1. **Generate** the SLURM array job from your config, giving a rough estimate
   of how long a single model fit takes:

   ```bash
   python hpc/create_jobs.py --config config.yaml --hours-per-fit 3
   ```

2. **Submit**:

   ```bash
   sbatch dispatch_fits.job
   ```

That's it. `create_jobs.py` writes four files in the current directory:

| file | purpose |
| --- | --- |
| `star_list.txt` | one star per line: `name <tab> models <tab> PDS path` |
| `granalyse_src.tar.gz` | the granalyse source, staged to node-local scratch |
| `run_node.sh` | stages a star to scratch, fits it, rsyncs results back |
| `dispatch_fits.job` | the `sbatch` script (SLURM directives + dispatch) |

Everything else uses sensible defaults. The per-task walltime is sized from
`--hours-per-fit`; partition, memory, CPUs and array limits are set by the
constants at the top of `create_jobs.py` (or just edit the generated
`dispatch_fits.job` before submitting).

### What happens on the node

Each array task ships `granalyse_src.tar.gz` to node-local `/scratch`, unpacks
it, and imports granalyse **from the node** (via `PYTHONPATH`) — so the heavy
package is not read back from the shared filesystem on every call. The venv in
the Granalyse checkout is activated automatically; no alias setup is needed.

Each star is fit entirely on node-local scratch, and its results are rsynced
back into `Results/<star>/` **after every individual model**, so an interrupted
task keeps every model it had already finished. Scratch is purged
automatically when the job ends.

### Resuming / topping up

Re-running `create_jobs.py` diagnoses what already has results and regenerates
the job for exactly the work that is left: fully finished stars are dropped,
and partly finished stars are listed with only their remaining models. If
nothing is pending it says so and writes no job. This makes it safe to re-run
after an interrupted batch, or after adding stars to the config.

Radius inference is cheap and is normally done afterwards on the collected
results (see `granalyse.radius.infer_radius`), or simply by running
`granalyse run config.yaml` locally with `output_dir` pointing at the
collected `Results/` tree — completed fits are loaded, not re-fitted.
