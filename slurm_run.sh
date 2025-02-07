#!/bin/bash
#SBATCH --job-name=HRC_normalized_slurm_2
#SBATCH --output=output_slurm/eval_log_0.txt
#SBATCH --error=output_slurm/eval_error_0txt
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8g
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
##SBATCH --partition=debug
#SBATCH --time=18:30:00
#SBATCH --account=engin1
##### END preamble

my_job_header
module load python3.10-anaconda
module load cuda/11.8.0
module load cudnn/11.8-v8.7.0
module load cupti/11.8.0
module load python/3.10.4
module load pytorch/2.0.1
module list

conda activate meta-q-learning
mkdir output_slurm

project_name=$SLURM_JOB_NAME
#API_KEY="API_KEY"
python -u model_based_rl.py \
--wandb_project $project_name \
--wandb_mode "online" \
--slurm_id ${SLURM_JOB_ID} > "output_slurm/eval_${SLURM_JOB_ID}_output.out"

#--debug_mode \
#--wandb_api_key $API_KEY \
