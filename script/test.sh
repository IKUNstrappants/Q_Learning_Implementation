#!/bin/bash
# List of parameter values to use

# Loop through each value and run the Python script with that value
echo "run test"
python BiomeTrainer_Bash.py --render_mode "None" --use_ddpg --ddpg_lr 3e-5 --max_iter 1000 --num_episodes 1
echo "all done"
