#!/bin/bash
#BSUB -J tribig 					# job name
#BSUB -W 00:10 						# walltime in HH:MM or MMMM
#BSUB -n 96 							# num CPU slots
#BSUB -M 64000 						# memory limit in MB
#BSUB -e ~/logs/posetail/%J.err 	# error file
#BSUB -o ~/logs/posetail/%J.out 	# out file
#BSUB -R "span[hosts=1]" 			# number of nodes spanned by gpus
#BSUB -q gpu_h200_parallel   		# type of gpu
#BSUB -gpu "num=8" 					# number of gpus

# define model params
CONFIG_PATH=${1:-"configs/config_default_3d.toml"}
NUM_NODES=${2:-1}
PRECISION=${3:-"32"}
STRATEGY=${4:-"ddp"}
echo "using config $CONFIG_PATH"
echo "pwd: $PWD"

source ~/.setup_env_vars.sh

# gpu specs
nvidia-smi

# pixi environment 
export PATH="$HOME/.pixi/bin:$PATH"
export PYTHONPATH="/groups/karashchuk/home/karashchukl/builds/vjepa2:$PYTHONPATH"
pixi --version

pixi install 
echo "pixi env setup complete"

# wandb login 
wandb login $WANDB_API_KEY

# run training script 
echo "starting training..."
pixi run python train.py --config-path "${CONFIG_PATH}" \
                         --num-nodes ${NUM_NODES} \
                         --precision ${PRECISION} \
                         --strategy ${STRATEGY}
echo "done!"
