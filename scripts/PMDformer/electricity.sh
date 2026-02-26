model_name=PMDformer

root_path_name=../dataset/electricity
data_path_name=electricity.csv
model_id_name=electricity
data_name=custom

seq_len=720

mkdir -p logs/PMDformer/ECL

for pred_len in 96 192 336 720
do
    log_file="logs/PMDformer/ECL/${seq_len}_${pred_len}.log"
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
      --enc_in 321 \
      --train_epochs 30 \
      --e_layers 2 \
      --v_layers 2 \
      --patience 5 \
      --dropout 0.1 \
      --n_heads 8 \
      --patch_size 72 \
      --d_model 256 \
      --d_ff 512 \
      --use_norm 0 \
      --lradj type1 \
      --itr 1 --batch_size 16 --learning_rate 0.0005 > $log_file 2>&1
done
