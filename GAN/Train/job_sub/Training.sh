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
#SBATCH --output=/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/job_sub/training.out
#SBATCH --error=/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/job_sub/training.err
  
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
#################/hpcfs/juno/junogpu/fangwx/python/Python-3.6.6/python /hpcfs/juno/junogpu/fangwx/FastSim/CEPC/models/train_v1.py --dataset /hpcfs/juno/junogpu/fangwx/FastSim/CEPC/data/Hit_Barrel_e.h5 --nb-epochs 100 --batch-size 128 --latent-size 512 --gen-model-out '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/gen_model.yaml' --gen-weight-out '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/gen_weight.h5' --dis-model-out '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/dis_model.yaml' --dis-weight-out '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/dis_weight.h5' --comb-model-out '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/comb_model.yaml' --comb-weight-out '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/comb_weight.h5'   --reg-model-in '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/reco_model_0p04.yaml' --reg-weight-in '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/reco_weight_0p04.h5' --gen-out '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/gen_model_all.h5' --comb-out '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/comb_model_all.h5'
#############33/hpcfs/juno/junogpu/fangwx/python/Python-3.6.6/python /hpcfs/juno/junogpu/fangwx/FastSim/CEPC/models/train_v1.py --dataset /hpcfs/juno/junogpu/fangwx/FastSim/CEPC/data/Hit_Barrel_e.h5 --nb-epochs 100 --batch-size 128 --latent-size 512 --gen-model-out '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/gen_model_woBN.yaml' --reg-model-in '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/reco_model_woBN.h5' --gen-out '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/gen_model_disv1.h5' --comb-out '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/comb_model_disv1.h5' --dis-out '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/dis_model_disv1.h5'
#################/hpcfs/juno/junogpu/fangwx/python/Python-3.6.6/python /hpcfs/juno/junogpu/fangwx/FastSim/CEPC/models/train_v2.py --dataset /hpcfs/juno/junogpu/fangwx/FastSim/CEPC/data/Hit_Barrel_e.h5 --nb-epochs 100 --batch-size 128 --latent-size 512 --reg-model-in '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/reco_model_woBN.h5' --gen-out '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/gen_model_disv2.h5' --comb-out '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/comb_model_disv2.h5' --dis-out '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/dis_model_disv2.h5'
################/hpcfs/juno/junogpu/fangwx/python/Python-3.6.6/python /hpcfs/juno/junogpu/fangwx/FastSim/CEPC/models/train_v3.py --datafile /hpcfs/juno/junogpu/fangwx/FastSim/CEPC/data/dataset.txt --nb-epochs 100 --batch-size 128 --latent-size 512 --reg-model-in '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/reco_model_0921.h5' --gen-out '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/gen_model_0923.h5' --comb-out '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/comb_model_0923.h5' --dis-out '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/dis_model_0923.h5'
/hpcfs/juno/junogpu/fangwx/python/Python-3.6.6/python /hpcfs/juno/junogpu/fangwx/FastSim/CEPC/models/train_v4.py --datafile /hpcfs/juno/junogpu/fangwx/FastSim/CEPC/data/dataset.txt --nb-epochs 500 --batch-size 128 --latent-size 512 --reg-model-in '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/reco_model_1001.h5' --gen-out '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/gen_model_1003.h5' --comb-out '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/comb_model_1003.h5' --dis-out '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/dis_model_1003.h5'
