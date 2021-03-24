#!/usr/bin/env bash
#SBATCH --job-name=large-sparse    # create a short name for your job

#SBATCH --partition=batch_default   # use batch_default, or wacc for quick (< 30 min) ones

# Node configs
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core
#SBATCH --time=6:00:00          # total run time limit (HH:MM:SS)
#SBATCH --gres=gpu:gtx1080:1     # GPU needed ##SBATCH --array=0-1

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

#Options: ${1} ${2}

# ${2}
# flower_16bit
# building_16bit
# bridge_16bit

if [ ${1} == 0.5 ]; then
  python main.py \
    exp_name='${img.name}_w_${mlp.hidden_size}_${masking.name}_${masking.density}_train_${train.multiplier}x' \
    img=${2} \
    +masking=RigL masking.density=${1} \
    wandb.project=large-sparse \
    train.multiplier=5 -m
fi

if [ ${1} == 0.25 ]; then
  python main.py \
    exp_name='${img.name}_w_${mlp.hidden_size}_${masking.name}_${masking.density}_train_${train.multiplier}x' \
    img=${2} \
    +masking=RigL masking.density=${1} \
    wandb.project=large-sparse \
    mlp.hidden_size=182 train.multiplier=5 -m
fi

if [ ${1} == 0.125 ]; then
  python main.py \
    exp_name='${img.name}_w_${mlp.hidden_size}_${masking.name}_${masking.density}_train_${train.multiplier}x' \
    img=${2} \
    +masking=RigL masking.density=${1} \
    wandb.project=large-sparse \
    mlp.hidden_size=256 train.multiplier=5 -m
fi
