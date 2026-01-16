
# ProFit

Official code for paper "ProFit: Leveraging High-Value Signals in SFT via Probability-Guided Token Selection"

  <a href="https://arxiv.org/abs/2601.09195"><b>[📜 Paper]</b></a> •
  <a href="https://github.com/Utaotao/ProFit"><b>[🐱 GitHub]</b></a>

This repo contains the code for our paper: <a href="https://arxiv.org/abs/2601.09195" target="_blank">ProFit: Leveraging High-Value Signals in SFT via Probability-Guided Token Selection</a>.


## Quick Start

The training codes are built on <a href="https://github.com/hiyouga/LLaMA-Factory" target="_blank">LLaMA-Factory</a>.
We employ <a href="https://github.com/open-compass/opencompass" target="_blank">OpenCompass</a> for evaluation.
Both are *Tremendous* projects, and you can find nearly everything there, thanks to their great framework and beautiful code!

### Environment

```bash
git clone https://github.com/Utaotao/ProFit
cd ProFit
pip install -e ".[torch,metrics]"
pip install torch==2.9.1 transformers==4.57.1 deepspeed==0.16.9
```

Please refer to [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory) for more details.

### Training Data

We use the Shadow 2K dataset and save it at `data/Shadow_2k.parquet`.
You can download via this [link](https://github.com/wutaiqiang/Shadow-FT/blob/main/data/Shadow_2k.parquet).

For custom datasets, remember to add information at `data/dataset_info.json`.

### For Train

Set `BASE_MODEL`, `BASE_OUTPUT_DIR`, and other parameters in `run.sh`, then:

```bash
bash run.sh
```

Set `BASE_MODEL=""` to download the model from Huggingface, rather than using a local file.

The training script will automatically:
- Create output directories
- Start training with ProFit loss function
- Log training progress to `training_log.log`
- Save model checkpoints at specified intervals

### Training Parameters

#### Core ProFit Parameters

- `prob_threshold`: Probability threshold(s) for sample selection
  - Single value: `0.1` (for `higher`, `lower`, `random` strategies)
  - Two values: `[0.3, 0.7]` (for `middle` strategy)

- `threshold_direction`: Sample selection strategy
  - `"higher"`: Train on tokens with prediction probability > threshold (core expression tokens)
  - `"lower"`: Train on tokens with prediction probability < threshold (non-core expression tokens)  
  - `"middle"`: Train on tokens with probability within threshold range
  - `"random"`: Randomly select tokens for training

#### Example Configurations

```bash
# Train on core expression tokens (recommended)
--prob_threshold 0.1 --threshold_direction "higher"

# Train on non-core expression tokens
--prob_threshold 0.8 --threshold_direction "lower"

# Train on medium probability tokens
--prob_threshold [0.3,0.7] --threshold_direction "middle"

# Random sampling (30% of tokens)
--prob_threshold 0.3 --threshold_direction "random"
```

### Training Script Example

```bash
BASE_MODEL="Qwen/Qwen3-0.6B-Base"
BASE_OUTPUT_DIR="./output"
DATASET="shadow_2k"
LEARNING_RATE=0.00001
prob_threshold="0.1"
threshold_direction="higher"

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
    --bf16 true \
    --flash_attn fa2
```

### For Evaluation

Please refer to <a href="https://github.com/open-compass/opencompass" target="_blank">OpenCompass</a> for evaluation.
You may find more details at this [repo](https://github.com/wutaiqiang/MI?tab=readme-ov-file#evaluation).

### Future Plan

- [ ] Introduce evaluation scripts in this repo.
- [ ] Add more threshold strategies.
- [ ] Support for multi-modal models.


## License

We use the Apache‑2.0 license. Please also comply with the licenses of any upstream models and datasets.

## ☕️ Citation

If you find this repository helpful, please consider citing our paper:

```bibtex
@article{profit2026,
  title={ProFit: Leveraging High-Value Signals in SFT via Probability-Guided Token Selection},
  author={Tao Liu and Taiqiang Wu and Runming Yang and Shaoning Sun and Junjie Wang and Yujiu Yang},
  journal={arXiv preprint arXiv:2601.09195},
  year={2026}
}
```

For any questions, feel free to pull an issue or email at `liu-t25@mails.tsinghua.edu.cn`
