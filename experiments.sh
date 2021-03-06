python train.py --data_dir tacred/data/json/ --vocab_dir vocab --id 00 --info "gcn, no label" --lr 0.3 --gcn --batch_size 50 --pooling_l2 0.003 --prune_k 1 --mlp_layers 2 --seed 1234 --save_dir ./new_models2 > log00 &
python train.py --data_dir tacred/data/json/ --vocab_dir vocab --id 01 --info "gcn, no label" --lr 0.3 --gcn --batch_size 50 --pooling_l2 0.003 --prune_k 1 --mlp_layers 2 --seed 0 --save_dir ./new_models2 > log01 &
python train.py --data_dir tacred/data/json/ --vocab_dir vocab --id 02 --info "gcn, no label" --lr 0.3 --gcn --batch_size 50 --pooling_l2 0.003 --prune_k 1 --mlp_layers 2 --seed 1 --save_dir ./new_models2 > log02 &
python train.py --data_dir tacred/data/json/ --vocab_dir vocab --id 03 --info "gcn, no label" --lr 0.3 --gcn --batch_size 50 --pooling_l2 0.003 --prune_k 1 --mlp_layers 2 --seed 2 --save_dir ./new_models2 > log03 &
python train.py --data_dir tacred/data/json/ --vocab_dir vocab --id 04 --info "gcn, no label" --lr 0.3 --gcn --batch_size 50 --pooling_l2 0.003 --prune_k 1 --mlp_layers 2 --seed 9999 --save_dir ./new_models2 > log04 &
wait
python train.py --data_dir tacred/data/json/ --vocab_dir vocab --id 10 --info "gcn, with label" --lr 0.3 --sgcn --batch_size 50 --pooling_l2 0.003 --prune_k 1 --mlp_layers 2 --seed 1234 --save_dir ./new_models2 > log10 &
python train.py --data_dir tacred/data/json/ --vocab_dir vocab --id 11 --info "gcn, with label" --lr 0.3 --sgcn --batch_size 50 --pooling_l2 0.003 --prune_k 1 --mlp_layers 2 --seed 0 --save_dir ./new_models2 > log11 &
python train.py --data_dir tacred/data/json/ --vocab_dir vocab --id 12 --info "gcn, with label" --lr 0.3 --sgcn --batch_size 50 --pooling_l2 0.003 --prune_k 1 --mlp_layers 2 --seed 1 --save_dir ./new_models2 > log12 &
python train.py --data_dir tacred/data/json/ --vocab_dir vocab --id 13 --info "gcn, with label" --lr 0.3 --sgcn --batch_size 50 --pooling_l2 0.003 --prune_k 1 --mlp_layers 2 --seed 2 --save_dir ./new_models2 > log13 &
python train.py --data_dir tacred/data/json/ --vocab_dir vocab --id 14 --info "gcn, with label" --lr 0.3 --sgcn --batch_size 50 --pooling_l2 0.003 --prune_k 1 --mlp_layers 2 --seed 9999 --save_dir ./new_models2 > log14 &
wait
python train.py --data_dir tacred/data/json/ --vocab_dir vocab --id 20 --info "gat, with label" --lr 0.3 --sgcn --batch_size 50 --pooling_l2 0.003 --mlp_layers 2 --seed 1234 --save_dir ./new_models2 > log20 &
python train.py --data_dir tacred/data/json/ --vocab_dir vocab --id 21 --info "gat, with label" --lr 0.3 --sgcn --batch_size 50 --pooling_l2 0.003 --mlp_layers 2 --seed 0 --save_dir ./new_models2 > log21 &
python train.py --data_dir tacred/data/json/ --vocab_dir vocab --id 22 --info "gat, with label" --lr 0.3 --sgcn --batch_size 50 --pooling_l2 0.003 --mlp_layers 2 --seed 1 --save_dir ./new_models2 > log22 &
python train.py --data_dir tacred/data/json/ --vocab_dir vocab --id 23 --info "gat, with label" --lr 0.3 --sgcn --batch_size 50 --pooling_l2 0.003 --mlp_layers 2 --seed 2 --save_dir ./new_models2 > log23 &
python train.py --data_dir tacred/data/json/ --vocab_dir vocab --id 24 --info "gat, with label" --lr 0.3 --sgcn --batch_size 50 --pooling_l2 0.003 --mlp_layers 2 --seed 9999 --save_dir ./new_models2 > log24 &
