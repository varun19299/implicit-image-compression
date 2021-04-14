#!/usr/bin/env bash
#SBATCH --job-name=image-fit    # create a short name for your job

#SBATCH --partition=batch_default   # use batch_default, or wacc for quick (< 30 min) ones

# Node configs
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core
#SBATCH --time=2:00:00          # total run time limit (HH:MM:SS)
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
#make siren

#python compress.py \
#exp_name='siren_depth_${mlp.depth}_amp_quant_density_${masking.density}_trainx_${train.multiplier}' \
#mlp.depth=4 train.mixed_precision=True \
#mlp.simulate_quantization=True \
#+masking=RigL masking.density=0.5 \
#train.multiplier=5 train.save_weights=True

# RigL
python main.py \
exp_name='${mlp.name}-${mlp.hidden_size}_${img.name}_train_${train.multiplier}x' \
img=flower_16bit,bridge_16bit,building_16bit \
+masking=RigL masking.density=0.75,0.5,0.2,0.1,0.05 \
mlp.hidden_size=128 train.multiplier=5 -m

# Pruning

# Small Dense

# FeatherMap


