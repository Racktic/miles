# `run_alchemy_qwen3.5_4B.sh` 环境变量速查

全部可选,不设就用默认。用法:`VAR=值 ... apptainer exec ... bash run_alchemy_qwen3.5_4B.sh`

## 资源 / 并行
| 变量 | 默认 | 含义 |
|---|---|---|
| `ALCHEMY_NGPU` | 本机卡数 | 用几张卡 |
| `ALCHEMY_TP` | =NGPU | 张量并行度 |
| `CUDA_VISIBLE_DEVICES` | `0..NGPU-1` | 指定可见卡 |

## 采样规模(调实验主要动这里)
| 变量 | 默认 | 含义 |
|---|---|---|
| `ALCHEMY_NUM_ROLLOUT` | 1 | 外层训练步数(主循环迭代次数) |
| `ALCHEMY_ROLLOUT_BATCH_SIZE` | 2 | 每步取几个 episode |
| `ALCHEMY_N_SAMPLES` | 8 | 每个 episode 采几条 sibling(ACT 组大小) |
| `ALCHEMY_GLOBAL_BATCH_SIZE` | 16 | 每个优化步的样本数 |
| `ALCHEMY_MAX_RESP` | 2560 | 单次生成 token 上限 |
| `ALCHEMY_MAX_TOK_PER_GPU` | 12288 | dynamic batch 每卡 token 上限 |

⚠️ **硬约束**:`GLOBAL_BATCH ≤ NUM_ROLLOUT × ROLLOUT_BATCH × N_SAMPLES`,否则 `train_iters` 取整为 0 → 启动断言失败。

## RL 超参
| 变量 | 默认 | 含义 |
|---|---|---|
| `ALCHEMY_KL` | 0.01 | KL-to-ref 系数(防 collapse) |
| `ALCHEMY_LR` | 1e-6 | 学习率 |

## 日志 / 落盘
| 变量 | 默认 | 含义 |
|---|---|---|
| `ALCHEMY_RUN_ID` | 时间戳 | 本次 run 名,决定 `logs/<id>/` 目录 |
| `ALCHEMY_TRAJ_DIR` | `logs/<id>/traj` | 每个 episode 的轨迹 json 落点 |
| `ALCHEMY_USE_WANDB` | (空=关) | 设 `1` 才开 wandb(还需有 key) |
| `WANDB_API_KEY` | 取 `~/.wandb.env` | wandb key |
| `WANDB_PROJECT` | miles-alchemy | wandb 项目 |
| `WANDB_GROUP` | qwen3.5-4B | wandb 分组 |

## 其它
| 变量 | 默认 | 含义 |
|---|---|---|
| `ALCHEMY_NO_DRYFAST` | (空=开dryfast) | 设非空则**捕获 cuda-graph**(解码快、启动慢);smoke 默认跳过 |
| `RAY_DASH_PORT` / `RAY_GCS_PORT` | 8265 / 6379 | ray 端口 |
| `MASTER_ADDR` | 127.0.0.1 | ray head 地址 |

## 非环境变量(在 `alchemy_config.yaml` 里改)
`max_turns`(200)、`arc_enable_thinking`(false)、`wm_gprime`(4,WRITE 候选数)、`wm_fk_cap`(12,F_k 上限)、`wm_min_fk`(3,F_k 太少则丢)、`wm_gen_max_tokens`(384)。

## 当前 smoke 示例
```bash
SIF=/data/user_data/qixinx/images/miles_dev-202606081341.sif
ALCHEMY_N_SAMPLES=8 ALCHEMY_NUM_ROLLOUT=1 ALCHEMY_ROLLOUT_BATCH_SIZE=1 ALCHEMY_GLOBAL_BATCH_SIZE=8 \
apptainer exec --nv --bind /data,/home/qixinx "$SIF" \
  bash /home/qixinx/miles/examples/alchemy/run_alchemy_qwen3.5_4B.sh
```
