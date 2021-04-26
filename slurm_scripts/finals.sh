#!/usr/bin/env bash
#SBATCH --job-name=finals    # create a short name for your job

#SBATCH --partition=batch_default   # use batch_default, or wacc for quick (< 30 min) ones

# Node configs
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core
#SBATCH --time=12:00:00          # total run time limit (HH:MM:SS)
#SBATCH --gres=gpu:rtx2080ti:1     # GPU needed ##SBATCH --array=0-1

# Mailing stuff
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=vsundar4@wisc.edu
#SBATCH --output=slurm_outputs/log-%x.%A_%a.out

# Job info
echo "== Starting run at $(date)"
echo "== Job ID: ${SLURM_JOB_ID}"
echo "== Node list: ${SLURM_NODELIST}"
echo "== Submit dir: ${SLURM_SUBMIT_DIR}"
echo "== Time limit: ${SBATCH_TIMELIMIT}"

nvidia-smi

# Conda stuff
module load cuda/10.2 anaconda/wml
source ~/.zshrc
conda activate torch1.7_py38

# NVIDIA SMI monitoring
if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
  while true
   do
       nvidia-smi | cat >"slurm_outputs/nvidia-smi-${SLURM_JOB_NAME}.${SLURM_JOB_ID}.out"
       sleep 0.1
  done &
fi

# Start Job here

# Options: ${1}
# flower
# building
# bridge

density_ll=(
    0.01
    0.02
    0.05
    0.10
    0.20
    0.25
    0.30
    0.35
    0.40
    0.45
    0.50
    0.55
    0.60
    0.65
    0.70
    0.75
    0.80
    0.85
    0.90
    0.95
)

for i in "${density_ll[@]}"; do
    if [ ${1} == "building" ] ||  [ ${1} == "bridge" ]; then
       make finals.compress.${1} DENSITY=$i KWARGS="mlp.hidden_size=182 quant.bits=9"
#    elif [ ${1} == "text_3" ]; then
#       make finals.compress.${1} DENSITY=$i KWARGS="mlp.hidden_size=64"
    else
       make finals.compress.${1} DENSITY=$i
    fi
done
