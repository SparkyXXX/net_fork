python ./code/val.py config.yaml --log_to_file --delete_past_log
tensorboard --logdir=./results/tb/val

python ./code/train.py config.yaml --log_to_file --delete_past_log
tensorboard --logdir=./results/tb/train
