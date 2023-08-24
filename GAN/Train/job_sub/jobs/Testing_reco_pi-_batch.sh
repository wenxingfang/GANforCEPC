#! /bin/bash
######## Part 1 #########
# Script parameters     #
#########################
  
# Specify the partition name from which resources will be allocated, mandatory option
#SBATCH --partition=gpu
  
# Specify the QOS, mandatory option
#SBATCH --qos=normal
  
# Specify which group you belong to, mandatory option
# This is for the accounting, so if you belong to many group,
#SBATCH --account=junogpu
  
# Specify your job name, optional option, but strongly recommand to specify some name
#SBATCH --job-name=train_1
  
# Specify how many cores you will need, default is one if not specified
#SBATCH --ntasks=1
  
# Specify the output file path of your job
# Attention!! Your afs account must have write access to the path
# Or the job will be FAILED!
#SBATCH --output=/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/job_sub/jobs/testing_reco_pi-.out
#SBATCH --error=/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/job_sub/jobs/testing_reco_pi-.err
  
# Specify memory to use, or slurm will allocate all available memory in MB
#SBATCH --mem-per-cpu=30720
#SBATCH --cpus-per-task=2  
#
# Specify how many GPU cards to use
#SBATCH --gres=gpu:v100:1 
######## Part 2 ######
# Script workload    #
######################
  
# Replace the following lines with your real workload
  
# list the allocated hosts
echo CUDA_VISIBLE_DEVICES $CUDA_VISIBLE_DEVICES
hostname
df -h
cd /hpcfs/juno/junogpu/fangwx
source /hpcfs/juno/junogpu/fangwx/setup.sh

/hpcfs/juno/junogpu/fangwx/python/Python-3.6.6/python /hpcfs/juno/junogpu/fangwx/FastSim/CEPC/reco/reco_test_v4.py --datafile /hpcfs/juno/junogpu/fangwx/FastSim/CEPC/data/HCAL/pi-/Digi_sim_2_20_pionm_95.h5 --model-in '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/reco/pi-/reco_model_0217.h5' --output /hpcfs/juno/junogpu/fangwx/FastSim/CEPC/reco/HCAL//Digi_sim_2_20_pionm_95_pred.h5


/hpcfs/juno/junogpu/fangwx/python/Python-3.6.6/python /hpcfs/juno/junogpu/fangwx/FastSim/CEPC/reco/reco_test_v4.py --datafile /hpcfs/juno/junogpu/fangwx/FastSim/CEPC/data/HCAL/pi-/Digi_sim_2_20_pionm_96.h5 --model-in '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/reco/pi-/reco_model_0217.h5' --output /hpcfs/juno/junogpu/fangwx/FastSim/CEPC/reco/HCAL//Digi_sim_2_20_pionm_96_pred.h5


/hpcfs/juno/junogpu/fangwx/python/Python-3.6.6/python /hpcfs/juno/junogpu/fangwx/FastSim/CEPC/reco/reco_test_v4.py --datafile /hpcfs/juno/junogpu/fangwx/FastSim/CEPC/data/HCAL/pi-/Digi_sim_2_20_pionm_97.h5 --model-in '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/reco/pi-/reco_model_0217.h5' --output /hpcfs/juno/junogpu/fangwx/FastSim/CEPC/reco/HCAL//Digi_sim_2_20_pionm_97_pred.h5


/hpcfs/juno/junogpu/fangwx/python/Python-3.6.6/python /hpcfs/juno/junogpu/fangwx/FastSim/CEPC/reco/reco_test_v4.py --datafile /hpcfs/juno/junogpu/fangwx/FastSim/CEPC/data/HCAL/pi-/Digi_sim_2_20_pionm_98.h5 --model-in '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/reco/pi-/reco_model_0217.h5' --output /hpcfs/juno/junogpu/fangwx/FastSim/CEPC/reco/HCAL//Digi_sim_2_20_pionm_98_pred.h5


/hpcfs/juno/junogpu/fangwx/python/Python-3.6.6/python /hpcfs/juno/junogpu/fangwx/FastSim/CEPC/reco/reco_test_v4.py --datafile /hpcfs/juno/junogpu/fangwx/FastSim/CEPC/data/HCAL/pi-/Digi_sim_2_20_pionm_99.h5 --model-in '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/reco/pi-/reco_model_0217.h5' --output /hpcfs/juno/junogpu/fangwx/FastSim/CEPC/reco/HCAL//Digi_sim_2_20_pionm_99_pred.h5


/hpcfs/juno/junogpu/fangwx/python/Python-3.6.6/python /hpcfs/juno/junogpu/fangwx/FastSim/CEPC/reco/reco_test_v4.py --datafile /hpcfs/juno/junogpu/fangwx/FastSim/CEPC/data/HCAL/pi-/Digi_sim_2_20_pionm_9.h5 --model-in '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/reco/pi-/reco_model_0217.h5' --output /hpcfs/juno/junogpu/fangwx/FastSim/CEPC/reco/HCAL//Digi_sim_2_20_pionm_9_pred.h5

