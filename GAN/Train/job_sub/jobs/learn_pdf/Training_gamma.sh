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
#SBATCH --output=/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/job_sub/jobs/learn_pdf/training_gamma.out
#SBATCH --error=/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/job_sub/jobs/learn_pdf/training_gamma.err
  
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
/hpcfs/juno/junogpu/fangwx/python/Python-3.6.6/python /hpcfs/juno/junogpu/fangwx/FastSim/CEPC/models/LearnPDF/learn_pdf_v1.py --doTraining True --datafile /hpcfs/juno/junogpu/fangwx/FastSim/CEPC/data/learn_pdf/gamma/dataset_train.txt --doValid True --validation_file /hpcfs/juno/junogpu/fangwx/FastSim/CEPC/data/learn_pdf/gamma/dataset_valid.txt --doTest True --test_file /hpcfs/juno/junogpu/fangwx/FastSim/CEPC/data/learn_pdf/gamma/gamma_test.h5  --nb-epochs 1000 --batch-size 128 --Restore False --restored_model 'model.h5' --outFilePath '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/generator/learn_pdf/pred_0212.h5' --model_out '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/generator/learn_pdf/gamma_model_0211.h5' --N_units 1000 --N_hidden 3 --act_mode 2 --lr 1.3996e-05
