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
#SBATCH --output=/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/job_sub/jobs/testing_gamma.out
#SBATCH --error=/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/job_sub/jobs/testing_gamma.err
  
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


###################### generate for mc reco ##################

############3/hpcfs/juno/junogpu/fangwx/python/Python-3.6.6/python /hpcfs/juno/junogpu/fangwx/FastSim/CEPC/generator/generator_v6.py --latent-size 512 --nb-events 4200 --gen-model-in "/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/gan/gamma/gen_model_1105_epoch71.h5" --comb-model-in "/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/gan/gamma/comb_model_1105_epoch71.h5" --for-mc-reco False  --output "/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/generator/Gen_gamma_1105_epoch71.h5" --exact-model False --exact-list '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/generator/exact_input_gamma.txt'        --check-dis-real True --dis-model-in '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/gan/gamma/dis_model_1105_epoch71.h5' --real-data '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/data/gamma/gamma_ext9.h5' --convert2pb False --path_pb "/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/generator/" --name_pb "model_gamma_epoch71.pb" --SavedModel False 
/hpcfs/juno/junogpu/fangwx/python/Python-3.6.6/python /hpcfs/juno/junogpu/fangwx/FastSim/CEPC/generator/generator_v6.py --latent-size 512 --nb-events 500 --gen-model-in "/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/gan/gamma/gen_model_1105_epoch71.h5" --comb-model-in "/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/gan/gamma/comb_model_1105_epoch71.h5"  --output "/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/generator/Gen_gamma_1105_epoch71_nnHaa.h5" --exact-model False --exact-list '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/generator/exact_input_gamma.txt'        --check-dis-real True --dis-model-in '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/gan/gamma/dis_model_1105_epoch71.h5' --real-data '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/data/gamma/check/test_nnhaa_gamma.h5' --convert2pb False --path_pb "/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/generator/" --name_pb "model_gamma_epoch71.pb" --SavedModel False 

