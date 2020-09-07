python train.py --data_dir tacred/data/json/ --vocab_dir vocab --id 10 --info "S GCN" --optim adagrad --sgcn --seed $1 > log_10 &
python train.py --data_dir tacred/data/json/ --vocab_dir vocab --id 11 --info "S GCN, entity average" --optim adagrad --sgcn --ee --seed $1 > log_11 &
python train.py --data_dir tacred/data/json/ --vocab_dir vocab --id 12 --info "S GCN, entity attention" --optim adagrad --sgcn --ee --e_attn --seed $1 > log_12 &
wait
python train.py --data_dir tacred/data/json/ --vocab_dir vocab --id 13 --info "R GCN" --optim adagrad --rgcn --seed $1 > log_13 &
python train.py --data_dir tacred/data/json/ --vocab_dir vocab --id 14 --info "R GCN, entity average" --optim adagrad --rgcn --ee --seed $1 > log_14 &
python train.py --data_dir tacred/data/json/ --vocab_dir vocab --id 15 --info "R GCN, entity attention" --optim adagrad --rgcn --ee --e_attn --seed $1 > log_15 &
wait
python train.py --data_dir tacred/data/json/ --vocab_dir vocab --id 16 --info "Vanilla GCN" --optim adagrad --gcn --seed $1 > log_16 &
python train.py --data_dir tacred/data/json/ --vocab_dir vocab --id 17 --info "Vanilla GCN, entity average" --optim adagrad --gcn --ee --seed $1 > log_17 &
python train.py --data_dir tacred/data/json/ --vocab_dir vocab --id 18 --info "Vanilla GCN, entity attention" --optim adagrad --gcn --ee --e_attn --seed $1 > log_18 &
wait
python train.py --data_dir tacred/data/json/ --vocab_dir vocab --id 19 --info "LSTM" --optim adagrad --seed $1 > log_19 &
python train.py --data_dir tacred/data/json/ --vocab_dir vocab --id 110 --info "LSTM, entity average" --optim adagrad --ee --seed $1 > log_110 &
python train.py --data_dir tacred/data/json/ --vocab_dir vocab --id 111 --info "LSTM, entity attention" --optim adagrad --ee --e_attn --seed $1 > log_111 &