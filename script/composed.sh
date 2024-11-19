#!/bin/bash

# List of parameter values to use

# Activate the Anaconda environment
source ~/anaconda3/bin/activate RFgame

# Loop through each value and run the Python script with that value
echo "run DDPG 1"
python BiomeTrainer_Bash.py --render_mode "None" --use_ddpg --ddpg_lr 0.00003 --max_iter 1000 --num_episodes 500
echo "run DDPG 2"
python BiomeTrainer_Bash.py --render_mode "None" --use_ddpg --ddpg_lr 0.00001 --max_iter 1000 --num_episodes 500
echo "run DDPG 3"
python BiomeTrainer_Bash.py --render_mode "None" --use_ddpg --ddpg_lr 0.0001 --max_iter 1000 --num_episodes 500
echo "run som"
python BiomeTrainer_Bash.py --render_mode "None" --use_som  --max_iter 1000 --num_episodes 500
echo "run dqn"
python BiomeTrainer_Bash.py --render_mode "None" --use_som  --som_lr 0 --max_iter 1000 --num_episodes 500
echo "all done"

# Deactivate the Anaconda environment
conda deactivate