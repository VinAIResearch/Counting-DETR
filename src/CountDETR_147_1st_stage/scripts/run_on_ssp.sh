#!/bin/bash -e
#SBATCH --job-name=container-job
#SBATCH --output=/lustre/scratch/client/vinai/users/thanhnv57/Task_Count_And_Detect/LearnToCountEveryThing/slurm_%A.out
#SBATCH --error=/lustre/scratch/client/vinai/users/thanhnv57/Task_Count_And_Detect/LearnToCountEveryThing/slurm_%A.err

#SBATCH --gpus=1

#SBATCH --nodes=1

#SBATCH --mem-per-gpu=36G

#SBATCH --cpus-per-gpu=8

#SBATCH --partition=research

#SBATCH --mail-type=all 
#SBATCH --mail-user=v.thanhnv57@vinai.io 
srun --container-image="harbor.vinai-systems.com#research/thanhnv57:pytorch18_cuda102_detectron204" \
--container-mounts=/lustre/scratch/client/vinai/users/thanhnv57/Task_Count_And_Detect/LearnToCountEveryThing/:/workspace/ \
python3 -W ignore train.py
# dgxuser@sdc2-hpc-login-mgmt001:~$ sbatch container.sh
