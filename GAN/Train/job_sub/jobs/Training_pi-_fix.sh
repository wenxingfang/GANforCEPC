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
#SBATCH --output=/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/job_sub/jobs/training_pi-_fix_v2.out
#SBATCH --error=/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/job_sub/jobs/training_pi-_fix_v2.err
  
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
##/hpcfs/juno/junogpu/fangwx/python/Python-3.6.6/python /hpcfs/juno/junogpu/fangwx/FastSim/CEPC/models/train_lsgan_v0_fix.py --datafile /hpcfs/juno/junogpu/fangwx/FastSim/CEPC/data/HCAL/pi-_fix/data_train.txt --nb-epochs 100 --batch-size 256 --latent-size 512 --reg-model-in '' --gen-out '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/gan/pi-_fix/gen_model_0224.h5' --comb-out '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/gan/pi-_fix/comb_model_0224.h5' --dis-out '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/gan/pi-_fix/dis_model_0224.h5'
###/hpcfs/juno/junogpu/fangwx/python/Python-3.6.6/python /hpcfs/juno/junogpu/fangwx/FastSim/CEPC/models/train_lsgan_v1_fix.py --datafile /hpcfs/juno/junogpu/fangwx/FastSim/CEPC/data/HCAL/pi-_fix/data_train.txt --nb-epochs 100 --batch-size 256 --latent-size 512 --reg-model-in '' --gen-out '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/gan/pi-_fix/gen_model_0225.h5' --comb-out '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/gan/pi-_fix/comb_model_0225.h5' --dis-out '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/gan/pi-_fix/dis_model_0225.h5'
##/hpcfs/juno/junogpu/fangwx/python/Python-3.6.6/python /hpcfs/juno/junogpu/fangwx/FastSim/CEPC/models/train_lsgan_v1_fix.py --datafile /hpcfs/juno/junogpu/fangwx/FastSim/CEPC/data/HCAL/pi-_fix/HoEless2/data_train.txt --nb-epochs 100 --batch-size 256 --latent-size 512 --reg-model-in '' --gen-out '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/gan/pi-_fix/gen_model_0226.h5' --comb-out '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/gan/pi-_fix/comb_model_0226.h5' --dis-out '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/gan/pi-_fix/dis_model_0226.h5'
###/hpcfs/juno/junogpu/fangwx/python/Python-3.6.6/python /hpcfs/juno/junogpu/fangwx/FastSim/CEPC/models/train_lsgan_v1_fix_EcalOnly.py --datafile /hpcfs/juno/junogpu/fangwx/FastSim/CEPC/data/HCAL/pi-_fix/HoEless2/data_train.txt --nb-epochs 100 --batch-size 256 --latent-size 512 --reg-model-in '' --gen-out '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/gan/pi-_fix/gen_model_0226v2.h5' --comb-out '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/gan/pi-_fix/comb_model_0226v2.h5' --dis-out '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/gan/pi-_fix/dis_model_0226v2.h5'


##/hpcfs/juno/junogpu/fangwx/python/Python-3.6.6/python /hpcfs/juno/junogpu/fangwx/FastSim/CEPC/models/train_lsgan_v2_fix_EcalOnly.py --datafile /hpcfs/juno/junogpu/fangwx/FastSim/CEPC/data/HCAL/pi-_fix/HoEless2/data_train.txt --nb-epochs 100 --batch-size 128 --latent-size 512 --reg-model-in '' --restore True --restore_gen '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/gan/pi-_fix/saved/gen_model_0228_epoch98.h5' --gen-out '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/gan/pi-_fix/gen_model_0304.h5' --comb-out '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/gan/pi-_fix/comb_model_0304.h5' --dis-out '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/gan/pi-_fix/dis_model_0304.h5'
##/hpcfs/juno/junogpu/fangwx/python/Python-3.6.6/python /hpcfs/juno/junogpu/fangwx/FastSim/CEPC/models/train_lsgan_v2_fix_EcalOnly.py --datafile /hpcfs/juno/junogpu/fangwx/FastSim/CEPC/data/HCAL/pi-_fix/HoEless2/data_train.txt --nb-epochs 100 --batch-size 128 --latent-size 512 --reg-model-in '' --restore False --restore_gen '' --gen-out '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/gan/pi-_fix/gen_model_0306.h5' --comb-out '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/gan/pi-_fix/comb_model_0306.h5' --dis-out '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/gan/pi-_fix/dis_model_0306.h5'
/hpcfs/juno/junogpu/fangwx/python/Python-3.6.6/python /hpcfs/juno/junogpu/fangwx/FastSim/CEPC/models/train_lsgan_v2_fix_EcalOnly.py --datafile /hpcfs/juno/junogpu/fangwx/FastSim/CEPC/data/HCAL/pi-_fix/HoEless2/data_train.txt --nb-epochs 100 --batch-size 256 --latent-size 512 --reg-model-in '' --restore False --restore_gen '' --gen-out '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/gan/pi-_fix/gen_model_0310.h5' --comb-out '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/gan/pi-_fix/comb_model_0310.h5' --dis-out '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/gan/pi-_fix/dis_model_0310.h5'
