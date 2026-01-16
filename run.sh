BASE_MODEL="your_base_model"
BASE_OUTPUT_DIR="your_base_output_dir"
DATASET="shadow_2k"
LEARNING_RATE=0.00001 
prob_threshold="0.1"
threshold_direction="higher"
OUTPUT_DIR="${BASE_OUTPUT_DIR}/full-${DATASET}-lr${LEARNING_RATE}"
LOG_FILE="${OUTPUT_DIR}/training_log.log"
mkdir -p "$OUTPUT_DIR"

llamafactory-cli train \
    --model_name_or_path "$BASE_MODEL" \
    --stage sft \
    --do_train true \
    --finetuning_type full \
    --prob_threshold $prob_threshold \
    --threshold_direction "$threshold_direction" \
    --dataset "$DATASET" \
    --cutoff_len 8192 \
    --max_samples 4000 \
    --output_dir "$OUTPUT_DIR" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate "$LEARNING_RATE" \
    --num_train_epochs 1 \
    --logging_steps 1 \
    --save_steps 200 \
    --save_only_model True \
    --plot_loss true \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --bf16 true \
    --per_device_eval_batch_size 1 \
    --trust_remote_code True \
    --flash_attn fa2 \
    --overwrite_cache false 2>&1 | tee "$LOG_FILE"