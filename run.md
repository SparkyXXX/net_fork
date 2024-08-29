python ./code/val.py config.yaml --delete_log --log_to_file
tensorboard --logdir=./results/tb/val

python ./code/train.py config.yaml --tensorboard --log_to_file --delete_model --delete_log
tensorboard --logdir=./results/tb/train
