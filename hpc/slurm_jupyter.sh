#!/bin/bash -l
# A big-memory node running Jupyter, for the notebooks that need lots of RAM.
# Submit with:  sbatch hpc/slurm_jupyter.sh   then see hpc/README.md for the tunnel.
#SBATCH --job-name=sptnano_jupyter
#SBATCH --partition=hpc_bigmem_a     # CHANGE ME
#SBATCH -c 192
#SBATCH --mem 1028G
#SBATCH --time 1-0:0:0
#SBATCH --output=sptnano_jupyter_%j.out

micromamba activate sptnano_10       # CHANGE ME

PORT=8888
echo "Node: $(hostname)  Port: $PORT"
echo "On your own computer run:  ssh -N -L $PORT:$(hostname):$PORT <user>@<login node>"
jupyter lab --no-browser --ip=0.0.0.0 --port=$PORT
