# HPC batch execution (optional)

For large samples the nested-sampling fits can be dispatched as a SLURM array
job. **The general user does not need anything in this directory** — `granalyse
run config.yaml` performs the same fits locally.

Workflow (run from a directory of your choice on the cluster):

1. **Prepare** the fit cases (fast, local or on the login node):

   ```bash
   python hpc/prepare_fitcases.py config.yaml --outdir Fitcases
   ```

2. **Generate** the SLURM array job (adjust the options to your cluster):

   ```bash
   python hpc/create_jobs.py --fitcases Fitcases \
       --venv /path/to/venv/bin/activate \
       --partition q36,q48 --walltime 15:00:00 --cpus 4 --mem-per-cpu 8G
   ```

3. **Submit**:

   ```bash
   sbatch dispatch_fits.job
   ```

Each array task runs one star x model fit on node-local scratch and rsyncs
the results back into `Results/<star>/`. Re-running `create_jobs.py` skips
fit cases that already have results, so interrupted batches can be resumed.

Radius inference is cheap and is normally done afterwards on the collected
results (see `granalyse.radius.infer_radius`), or simply by running
`granalyse run config.yaml` locally with `output_dir` pointing at the
collected `Results/` tree — completed fits are loaded, not re-fitted.
