# Running on a cluster (optional)

Only step 2a (window features) is slow enough to need a cluster. Everything here depends on your institution, so treat the scripts as templates. Lines that need editing are marked `CHANGE ME`.

## 1. Put the code and data on the cluster

```bash
git clone <your github url> SPTnano-core        # or copy the folder
mamba env create -f SPTnano-core/environment.yml -n SPTnano_HPC
```

Copy the experiment's `saved_data/instant_df.csv` (from notebook 01) to scratch.

## 2. Window features

Edit `REPO_DIR` and `EXPERIMENT_DIR` in `slurm_compute_windows.sh`, then run:

```bash
sbatch hpc/slurm_compute_windows.sh
squeue -u $USER                     # check it is running
```

Copy `instant_df_processed.csv` and `time_windowed_df_processed.csv` back into the experiment's `saved_data/` folder.

## 3. Jupyter on a big node (tunnel)

```bash
sbatch hpc/slurm_jupyter.sh
cat sptnano_jupyter_<jobid>.out     # shows the node name and the ssh command
```

On your own computer, run the printed command, for example:

```bash
ssh -N -L 8888:<node>:8888 <user>@<login node>
```

Then open the `http://127.0.0.1:8888/lab?token=...` link from the `.out` file in your browser. Keep the ssh window open while you work.
