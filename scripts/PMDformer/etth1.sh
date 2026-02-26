model_name=PMDformer

root_path_name=../dataset/ETT-small
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1

seq_len=720

mkdir -p logs/PMDformer/ETTh1

pred_len=96
log_file="logs/PMDformer/ETTh1/${seq_len}_${pred_len}.log"
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
  --enc_in 7 \
  --train_epochs 30 \
  --e_layers 1 \
  --v_layers 1 \
  --patience 5 \
  --dropout 0.3 \
  --n_heads 4 \
  --patch_size 72 \
  --d_model 64 \
  --d_ff 128 \
  --lradj type3 \
  --itr 1 --batch_size 128 --learning_rate 0.0005 > $log_file 2>&1

pred_len=192
log_file="logs/PMDformer/ETTh1/${seq_len}_${pred_len}.log"
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
  --enc_in 7 \
  --train_epochs 30 \
  --e_layers 1 \
  --v_layers 1 \
  --patience 5 \
  --dropout 0.3 \
  --n_heads 4 \
  --patch_size 72 \
  --d_model 32 \
  --d_ff 32 \
  --lradj type3 \
  --itr 1 --batch_size 128 --learning_rate 0.0005 > $log_file 2>&1

pred_len=336
log_file="logs/PMDformer/ETTh1/${seq_len}_${pred_len}.log"
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
  --enc_in 7 \
  --train_epochs 30 \
  --e_layers 1 \
  --v_layers 1 \
  --patience 5 \
  --dropout 0.3 \
  --n_heads 4 \
  --patch_size 72 \
  --d_model 32 \
  --d_ff 32 \
  --lradj type3 \
  --itr 1 --batch_size 128 --learning_rate 0.0005 > $log_file 2>&1

pred_len=720
log_file="logs/PMDformer/ETTh1/${seq_len}_${pred_len}.log"
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
  --enc_in 7 \
  --train_epochs 30 \
  --e_layers 1 \
  --v_layers 1 \
  --patience 5 \
  --dropout 0.3 \
  --n_heads 4 \
  --patch_size 72 \
  --d_model 32 \
  --d_ff 32 \
  --lradj type3 \
  --itr 1 --batch_size 128 --learning_rate 0.0005 > $log_file 2>&1

