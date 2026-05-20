# Student-Training Hyperparameters

The student model is a LoRA fine-tune of `Qwen/Qwen2.5-7B-Instruct`. The
default hyperparameters used in the paper are pinned in two places and
are reproduced here for convenience.

## Architecture-level defaults (`configs/base.yaml -> models.student`)

| Setting        | Value                                                                |
| -------------- | -------------------------------------------------------------------- |
| `lora_r`       | 64                                                                   |
| `lora_alpha`   | 128                                                                  |
| `lora_dropout` | 0.05                                                                 |
| target modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`      |

## Training-loop defaults (`configs/base.yaml -> training`)

| Setting                     | Value     |
| --------------------------- | --------- |
| `epochs`                    | 3         |
| `lr`                        | 2.0e-4    |
| `batch_per_gpu`             | 8         |
| `grad_accum`                | 4         |
| `cosine_warmup`             | 0.03      |
| `faithfulness_loss_weight`  | 0.5       |
| `symmetry_loss_weight`      | 0.3       |
| DPO `beta`                  | 0.1       |
| DPO `prm_weight_exponent`   | 1.0       |
| DPO `mirror_pair_ratio`     | 0.5       |
| classifier head `dropout`   | 0.1       |
| classifier `temperature_init` | 1.0     |

## Launcher overrides

Every value above can be overridden through environment variables to the
example launchers under `scripts/cluster_examples/`. See
`scripts/cluster_examples/run_phase_c_sft.sh` and
`scripts/cluster_examples/run_phase_c_dpo.sh` for the full list.
