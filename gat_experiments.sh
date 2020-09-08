python train.py --data_dir tacred/data/json/ --vocab_dir vocab --id 10 --info "GAT" --optim adagrad --gat --seed 1234 > log_10 &
python train.py --data_dir tacred/data/json/ --vocab_dir vocab --id 11 --info "GAT, entity average" --optim adagrad --gat --ee --seed 1234 > log_11 &
python train.py --data_dir tacred/data/json/ --vocab_dir vocab --id 12 --info "GAT, entity attention" --optim adagrad --gat --ee --e_attn --seed 1234 > log_12 &
wait
python train.py --data_dir tacred/data/json/ --vocab_dir vocab --id 13 --info "GAT" --optim adagrad --gat --seed 0 > log_13 &
python train.py --data_dir tacred/data/json/ --vocab_dir vocab --id 14 --info "GAT, entity average" --optim adagrad --gat --ee --seed 0 > log_14 &
python train.py --data_dir tacred/data/json/ --vocab_dir vocab --id 15 --info "GAT, entity attention" --optim adagrad --gat --ee --e_attn --seed 0 > log_15 &
wait
python train.py --data_dir tacred/data/json/ --vocab_dir vocab --id 16 --info "GAT" --optim adagrad --gat --seed 9999 > log_16 &
python train.py --data_dir tacred/data/json/ --vocab_dir vocab --id 17 --info "GAT, entity average" --optim adagrad --gat --ee --seed 9999 > log_17 &
python train.py --data_dir tacred/data/json/ --vocab_dir vocab --id 18 --info "GAT, entity attention" --optim adagrad --gat --ee --e_attn --seed 9999 > log_18 &