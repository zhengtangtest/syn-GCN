python train.py --data_dir tacred/data/json/ --vocab_dir vocab --id 20 --info "S GCN" --optim adagrad --sgcn --seed $1 > log_10 &
python train.py --data_dir tacred/data/json/ --vocab_dir vocab --id 21 --info "S GCN, entity average" --optim adagrad --sgcn --ee --seed $1 > log_11 &
python train.py --data_dir tacred/data/json/ --vocab_dir vocab --id 22 --info "S GCN, entity attention" --optim adagrad --sgcn --ee --e_attn --seed $1 > log_12 &
wait
python train.py --data_dir tacred/data/json/ --vocab_dir vocab --id 23 --info "R GCN" --optim adagrad --rgcn --seed $1 > log_13 &
python train.py --data_dir tacred/data/json/ --vocab_dir vocab --id 24 --info "R GCN, entity average" --optim adagrad --rgcn --ee --seed $1 > log_14 &
python train.py --data_dir tacred/data/json/ --vocab_dir vocab --id 25 --info "R GCN, entity attention" --optim adagrad --rgcn --ee --e_attn --seed $1 > log_15 &
wait
python train.py --data_dir tacred/data/json/ --vocab_dir vocab --id 26 --info "Vanilla GCN" --optim adagrad --gcn --seed $1 > log_16 &
python train.py --data_dir tacred/data/json/ --vocab_dir vocab --id 27 --info "Vanilla GCN, entity average" --optim adagrad --gcn --ee --seed $1 > log_17 &
python train.py --data_dir tacred/data/json/ --vocab_dir vocab --id 28 --info "Vanilla GCN, entity attention" --optim adagrad --gcn --ee --e_attn --seed $1 > log_18 &
wait
python train.py --data_dir tacred/data/json/ --vocab_dir vocab --id 29 --info "LSTM" --optim adagrad --seed $1 > log_19 &
python train.py --data_dir tacred/data/json/ --vocab_dir vocab --id 210 --info "LSTM, entity average" --optim adagrad --ee --seed $1 > log_110 &
python train.py --data_dir tacred/data/json/ --vocab_dir vocab --id 211 --info "LSTM, entity attention" --optim adagrad --ee --e_attn --seed $1 > log_111 &