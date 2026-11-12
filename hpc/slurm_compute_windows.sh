#!/bin/bash
# Step 2a on a cluster: window features for one experiment. Submit with:  sbatch hpc/slurm_compute_windows.sh
# Lines marked CHANGE ME are specific to your cluster account.
#SBATCH --job-name=sptnano_parallel
#SBATCH --account=briv_hotel_bank          # CHANGE ME
#SBATCH --partition=hpc_bigmem_a           # CHANGE ME
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --mem=512G
#SBATCH --time=1-0:0:0
#SBATCH --output=sptnano_parallel_%j.out
#SBATCH --error=sptnano_parallel_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=<your email>           # CHANGE ME

echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST  CPUs: $SLURM_CPUS_PER_TASK  Start: $(date)"

source "$HOME/miniforge3/etc/profile.d/conda.sh"                        # CHANGE ME
eval "$(mamba shell hook --shell bash)"
mamba activate SPTnano_HPC                                               # CHANGE ME (env from environment.yml)

export REPO_DIR="/lustre/fs4/briv_lab/store/mshannon/SPTnano-core"                         # CHANGE ME
export EXPERIMENT_DIR="/lustre/fs4/briv_lab/scratch/mshannon/<experiment>_analyze"         # CHANGE ME
export INPUT_FILE="$EXPERIMENT_DIR/saved_data/instant_df.csv"
export OUTPUT_DIR="$EXPERIMENT_DIR/saved_data"

if [ ! -f "$INPUT_FILE" ]; then
    echo "Input file not found: $INPUT_FILE"
    exit 1
fi
mkdir -p "$OUTPUT_DIR"
cd "$REPO_DIR"

python scripts/compute_windows.py \
    --input_file "$INPUT_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --n_jobs "$SLURM_CPUS_PER_TASK" \
    --chunk_size 2000 \
    --time_between_frames 0.01 || { echo "Processing failed"; exit 1; }

echo "Done: $OUTPUT_DIR  End: $(date)"
