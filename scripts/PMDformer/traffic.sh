model_name=PMDformer

root_path_name=../dataset/traffic
data_path_name=traffic.csv
model_id_name=traffic
data_name=custom

seq_len=720

mkdir -p logs/PMDformer/traffic


pred_len=96
log_file="logs/PMDformer/traffic/${seq_len}_${pred_len}.log"
echo "Training seq_len=${seq_len}, pred_len=${pred_len} | Logging to: $log_file"

python -u run.py \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data $data_name \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 862 \
  --e_layers 3 \
  --v_layers 1 \
  --train_epochs 50 \
  --patience 5 \
  --n_heads 8 \
  --patch_size 72 \
  --d_model 512 \
  --d_ff 512 \
  --lradj type3 \
  --dropout 0.1 \
  --itr 1 --batch_size 32 --learning_rate 0.001 > $log_file 2>&1

pred_len=192
log_file="logs/PMDformer/traffic/${seq_len}_${pred_len}.log"
echo "Training seq_len=${seq_len}, pred_len=${pred_len} | Logging to: $log_file"

python -u run.py \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data $data_name \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 862 \
  --e_layers 3 \
  --v_layers 1 \
  --train_epochs 50 \
  --patience 5 \
  --n_heads 16 \
  --patch_size 72 \
  --d_model 256 \
  --d_ff 512 \
  --lradj type3 \
  --dropout 0.1 \
  --itr 1 --batch_size 16 --learning_rate 0.001 > $log_file 2>&1


pred_len=336
log_file="logs/PMDformer/traffic/${seq_len}_${pred_len}.log"
echo "Training seq_len=${seq_len}, pred_len=${pred_len} | Logging to: $log_file"

python -u run.py \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data $data_name \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 862 \
  --e_layers 3 \
  --v_layers 1 \
  --train_epochs 50 \
  --patience 5 \
  --n_heads 8 \
  --patch_size 72 \
  --d_model 512 \
  --d_ff 512 \
  --lradj type3 \
  --dropout 0.1 \
  --itr 1 --batch_size 16 --learning_rate 0.0005 > $log_file 2>&1


pred_len=720
log_file="logs/PMDformer/traffic/${seq_len}_${pred_len}.log"
echo "Training seq_len=${seq_len}, pred_len=${pred_len} | Logging to: $log_file"

python -u run.py \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data $data_name \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 862 \
  --e_layers 3 \
  --v_layers 1 \
  --train_epochs 50 \
  --patience 5 \
  --n_heads 8 \
  --patch_size 72 \
  --d_model 512 \
  --d_ff 512 \
  --lradj type3 \
  --dropout 0.1 \
  --itr 1 --batch_size 16 --learning_rate 0.0005 > $log_file 2>&1
