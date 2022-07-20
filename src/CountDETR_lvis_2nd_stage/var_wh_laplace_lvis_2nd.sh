#!/bin/bash -e
#SBATCH --job-name=anchor_lvis
#SBATCH --output=/lustre/scratch/client/vinai/users/chauph12/LVIS_exps/FewShotDETR_var_wh_laplace_lvis_2nd_stage/slurm_%A.out
#SBATCH --error=/lustre/scratch/client/vinai/users/chauph12/LVIS_exps/FewShotDETR_var_wh_laplace_lvis_2nd_stage/slurm_%A.err

#SBATCH --gpus=1

#SBATCH --nodes=1

#SBATCH --mem-per-gpu=36G

#SBATCH --cpus-per-gpu=8

#SBATCH --partition=research

#SBATCH --mail-type=all 
#SBATCH --mail-user=v.chauph12@vinai.io 
srun --container-image="harbor.vinai-systems.com#research/thanhnv57:pytorch18_cuda102_detectron206" \
--container-mounts=/lustre/scratch/client/vinai/users/chauph12/LVIS_exps/FewShotDETR_var_wh_laplace_lvis_2nd_stage/:/workspace/ \
 sh ./scripts/var_wh_laplace_lvis_2nd.sh
# dgxuser@sdc2-hpc-login-mgmt001:~$ sbatch container.sh
#  sbatch container.sh

