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
#SBATCH --output=/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/job_sub/testing.out
#SBATCH --error=/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/job_sub/testing.err
  
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
##################/hpcfs/juno/junogpu/fangwx/python/Python-3.6.6/python /hpcfs/juno/junogpu/fangwx/FastSim/CEPC/generator/generator_v3.py --latent-size 512 --nb-events 5000 --gen-model-in "/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/gen_model_disv2.h5" --comb-model-in "/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/comb_model_disv2.h5"   --exact-model True --exact-list '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/generator/exact_input.txt'  --output "/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/generator/Gen0915.h5" --check-dis-real True --dis-model-in '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/dis_model_disv2.h5' --real-data '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/data/Hit_Barrel_e_ext1_10000.h5'

####################/hpcfs/juno/junogpu/fangwx/python/Python-3.6.6/python /hpcfs/juno/junogpu/fangwx/FastSim/CEPC/generator/generator_v4.py --latent-size 512 --nb-events 5000 --gen-model-in "/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/gen_model_1003.h5" --comb-model-in "/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/comb_model_1003.h5"   --exact-model True --exact-list '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/generator/exact_input_v1.txt'  --output "/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/generator/Gen1007.h5" --check-dis-real True --dis-model-in '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/dis_model_1003.h5' --real-data '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/data/Dg_10_40_em_0.h5'


###################### generate for mc reco ##################
/hpcfs/juno/junogpu/fangwx/python/Python-3.6.6/python /hpcfs/juno/junogpu/fangwx/FastSim/CEPC/generator/generator_v4.py --latent-size 512 --nb-events 1 --gen-model-in "/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/gen_model_1003.h5" --comb-model-in "/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/comb_model_1003.h5" --for-mc-reco True --info-mc-reco '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/data/mc_info/mc_info_for_reco.h5'  --output "/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/generator/Gen1008_mc_reco.h5" --exact-model True --exact-list '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/generator/exact_input_mc.txt'        --check-dis-real True --dis-model-in '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/dis_model_1003.h5' --real-data '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/data/mc_info/mc_info_and_cell_ID_final_orgin.h5' 
