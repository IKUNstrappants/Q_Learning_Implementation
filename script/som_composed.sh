#!/bin/bash
# List of parameter values to use

# Loop through each value and run the Python script with that value

# python BiomeTrainer_Bash.py --render_mode "None" --use_ddpg --ddpg_lr 1e-4 --max_iter 1000 --num_episodes 500

python BiomeTrainer_Bash.py --render_mode "None" --use_som  --max_iter 1000 --num_episodes 300 --som_epsilon 2.0 --som_lr 1

python BiomeTrainer_Bash.py --render_mode "None" --use_som  --max_iter 1000 --num_episodes 300 --som_epsilon 1.0 --som_lr 1

python BiomeTrainer_Bash.py --render_mode "None" --use_som  --max_iter 1000 --num_episodes 300 --som_epsilon 0.5 --som_lr 1

python BiomeTrainer_Bash.py --render_mode "None" --use_som  --max_iter 1000 --num_episodes 300 --som_epsilon 1.0 --som_lr 1e-1

python BiomeTrainer_Bash.py --render_mode "None" --use_som  --max_iter 1000 --num_episodes 300 --som_epsilon 1.0 --som_lr 1e-2

# python BiomeTrainer_Bash.py --render_mode "None" --use_som  --som_lr 0 --max_iter 1000 --num_episodes 300


