#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=23:00:00
#SBATCH --job-name 1-np-dql
#SBATCH --output=results.txt
#SBATCH --mail-user=nparv038@uottawa.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

module load anaconda3/2021.05 cuda/11.2.2 gcc
source activate env

cd /scratch/b/bkantarc/nahidp/ns-allinone-3.35/ns-3.35/scratch/ACDQL 
echo "Current working directory: `pwd`"
echo "Starting run at: `date`"

python3 Agent-ACDQL.py

echo 'Agent-DQL-New is called.'
wait

# ---------------------------------------------------------------------
echo "Job finished with exit code $? at: `date`"
# ---------------------------------------------------------------------

conda deactivate
