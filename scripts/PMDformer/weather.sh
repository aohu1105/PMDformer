model_name=PMDformer

root_path_name=../dataset/weather
data_path_name=weather.csv
model_id_name=weather
data_name=custom

seq_len=720

mkdir -p logs/PMDformer/Weather

pred_len=96
log_file="logs/PMDformer/Weather/${seq_len}_${pred_len}.log"
echo "Saving log to: $log_file"

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
  --enc_in 21 \
  --train_epochs 10 \
  --e_layers 2 \
  --v_layers 1 \
  --patience 5 \
  --dropout 0.1 \
  --n_heads 8 \
  --patch_size 48 \
  --d_model 128 \
  --d_ff 128 \
  --use_norm 0 \
  --lradj type1 \
  --itr 1 --batch_size 32 --learning_rate 0.0005 > $log_file 2>&1

pred_len=192
log_file="logs/PMDformer/Weather/${seq_len}_${pred_len}.log"
echo "Saving log to: $log_file"

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
  --enc_in 21 \
  --train_epochs 10 \
  --e_layers 2 \
  --v_layers 1 \
  --patience 5 \
  --dropout 0.1 \
  --n_heads 8 \
  --patch_size 48 \
  --d_model 32 \
  --d_ff 32 \
  --use_norm 0 \
  --lradj type1 \
  --itr 1 --batch_size 32 --learning_rate 0.001 > $log_file 2>&1

pred_len=336
log_file="logs/PMDformer/Weather/${seq_len}_${pred_len}.log"
echo "Saving log to: $log_file"

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
  --enc_in 21 \
  --train_epochs 10 \
  --e_layers 2 \
  --v_layers 1 \
  --patience 5 \
  --dropout 0.1 \
  --n_heads 8 \
  --patch_size 48 \
  --d_model 32 \
  --d_ff 64 \
  --use_norm 0 \
  --lradj type1 \
  --itr 1 --batch_size 32 --learning_rate 0.0002 > $log_file 2>&1

pred_len=720
log_file="logs/PMDformer/Weather/${seq_len}_${pred_len}.log"
echo "Saving log to: $log_file"

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
  --enc_in 21 \
  --train_epochs 10 \
  --e_layers 2 \
  --v_layers 1 \
  --patience 5 \
  --dropout 0.1 \
  --n_heads 8 \
  --patch_size 48 \
  --d_model 32 \
  --d_ff 32 \
  --use_norm 0 \
  --lradj type1 \
  --itr 1 --batch_size 32 --learning_rate 0.001 > $log_file 2>&1