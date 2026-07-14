# `run_alchemy_qwen3-4B.sh` 环境变量速查

全部可选,不设就用默认。用法:`VAR=值 ... apptainer exec --nv --bind /data,/home/qixinx "$SIF" bash run_alchemy_qwen3-4B.sh`
(`--load==--save==ALCHEMY_SAVE_DIR`:目录不存在→从 `ref-load` 冷启;已存在→自动 resume 到 latest。)

## 资源 / 并行
| 变量 | 默认 | 含义 |
|---|---|---|
| `ALCHEMY_NGPU` | 本机卡数(`nvidia-smi -L`) | 用几张卡 |
| `ALCHEMY_TP` | =NGPU | 张量并行度(实跑 8 卡常用 TP=2 → 4 路 DP) |
| `ALCHEMY_CP` | 1 | context parallel |
| `ALCHEMY_SEQ_LEN` | 32768 | seq-length / max-position-embeddings |
| `ALCHEMY_MAX_TOK_PER_GPU` | 20480 | dynamic batch 每卡 token 上限 |
| `ALCHEMY_LOGP_CHUNK` | 512 | log-probs chunk |
| `ALCHEMY_DIST_TIMEOUT_MIN` | 120 | 分布式超时(分钟) |
| `ALCHEMY_RAY_NUM_CPUS` | (空) | ray `--num-cpus`(限制 ray 占的 CPU) |

## 采样规模(调实验主要动这里)
| 变量 | 默认 | 含义 |
|---|---|---|
| `ALCHEMY_NUM_ROLLOUT` | 1 | 外层训练步数(= train_iters,主循环迭代次数) |
| `ALCHEMY_ROLLOUT_BATCH_SIZE` | 2 | 每步取几个 episode |
| `ALCHEMY_N_SAMPLES` | 8 | 每 episode 采几条 sibling(ACT 组大小) |
| `ALCHEMY_GLOBAL_BATCH_SIZE` | 16 | 每个优化步样本数(用 dynamic-global-batch,会按实际放大) |
| `ALCHEMY_MAX_RESP` | 2560 | 单次生成 token 上限 |

⚠️ **硬约束**:`GLOBAL_BATCH ≤ NUM_ROLLOUT × ROLLOUT_BATCH × N_SAMPLES`,否则 `train_iters` 取整为 0 → 启动断言失败。

## RL 超参
| 变量 | 默认 | 含义 |
|---|---|---|
| `ALCHEMY_KL` | 0.01 | KL-to-ref 系数(`--use-kl-loss --kl-loss-coef`,防 collapse) |
| `ALCHEMY_LR` | 1e-6 | 学习率(constant) |

固定在脚本里(非 env):`--eps-clip 0.2 --eps-clip-high 0.28`、`--advantage-estimator grpo`、adam β 0.9/0.98、weight-decay 0.1、optimizer-cpu-offload。

## 训练流 / memory(两流 GRPO 的开关)
| 变量 | 默认 | 含义 |
|---|---|---|
| `ALCHEMY_TRAIN_ACT` | 1 | 是否训练 ACT 流(用 memory 做决策)。`0`=只训 WRITE |
| `ALCHEMY_TRAIN_WRITE` | 1 | 是否训练 WRITE 流(写 memory)。`0`=ACT-only(memory 仍在线生成供 ACT 用) |
| `ALCHEMY_MEMORY_WINDOW_SIZE` | 1 | WRITE 上下文窗口。`1`=旧行为(上一条 memory + 上一 trial);`>1`=喂最近 K 条 memory + K 个 trial transcript |
| `ALCHEMY_FREEFORM` | 0 | `1`=自由格式 memory(去掉固定两段式结构) |
| `ALCHEMY_NO_MEMORY` | 0 | `1`=无 memory 全历史 baseline(整 episode 一条样本,按 trial 段给 advantage) |
| `ALCHEMY_NUM_TRIALS_CAP` | 0 | `>0`=把 episode 截断到前 N 个 trial(ablation) |

## WRITE reward 信号
| 变量 | 默认 | 取值 |
|---|---|---|
| `ALCHEMY_WRITE_SIGNAL` | `transition_acc` | `transition_acc`(候选 memory 在 F_k 上的预测准确率)/ `downstream`(下一 trial 的 ACT 得分,按 `(group,k+1)` 白化)/ `downstream_improve`(`r_{k+1}-r_k`)/ `downstream_norm_improve`(`r_{k+1}/oracle_{k+1} - r_k/oracle_k`) |
| `ALCHEMY_WRITE_IMPROVE_K` | 1 | 仅作用于 `downstream_norm_improve`:trial 差分窗口,`R(M_k) = mean(norm[k+1..k+K]) − mean(norm[k−K+1..k])`(窗口按 episode 裁剪、oracle≤0 的 trial 剔除)。`1` = 原单 trial 差分,逐字节等价;白化 key(`downstream_trial_pos=k+1`)不变。降方差用,建议与 memory window 对齐(如 3) |
| `ALCHEMY_WRITE_K0_MODE` | improve | 仅作用于 `downstream_norm_improve` 的 **k=0**(第一条记忆):`improve`=现行为 `mean(norm[1..K])−norm0`;`downstream`=去掉 `−norm0` 项(拆除"为烂第一局付费"通道);`skip`=k=0 不训 WRITE(M_0 照常写入/使用)。其余 k 一律不变 |

## ACT exploration reward(memory-delta judge,新)
| 变量 | 默认 | 含义 |
|---|---|---|
| `ALCHEMY_ACT_EXPLORE_BETA` | 0 | `>0` 开启。`act_adv += β·explore_adv`,explore_adv = judge(M_{k-1}→M_k) 的 explore_score 在 `(group,trial_pos)` 组内标准化。`0`=完全关闭(不调 judge、行为不变) |
| `ALCHEMY_JUDGE_MODEL` | deepseek-chat | judge 模型(走 deepseek API) |
| `ALCHEMY_JUDGE_CONCURRENCY` | 64 | judge 并发上限(进程内 semaphore) |
| `ALCHEMY_JUDGE_TIMEOUT` | 60 | 单次 judge 超时(秒);失败→该 trial explore_score 缺省、不阻塞训练 |
| `DEEPSEEK_API_KEY` / `DEEPSEEK_API_BASE` | (从 `.env`) | judge API key / base(`https://api.deepseek.com`) |

## 数据 / 课程
| 变量 | 默认 | 含义 |
|---|---|---|
| `ALCHEMY_PROMPT_DATA` | `data/alchemy_episodes.jsonl` | 训练数据(curriculum 实验用 `data/alchemy_train_950_curriculum.jsonl`) |
| `ALCHEMY_SHUFFLE` | **1** | `1`=`--rollout-shuffle`(打乱);**`0`=按文件顺序喂 = curriculum**。⚠️ curriculum 实验必须显式设 `0`,否则默认打乱 |

## Eval
| 变量 | 默认 | 含义 |
|---|---|---|
| `ALCHEMY_EVAL_INTERVAL` | (空=不 eval) | 每多少步在 eval 集上评一次 |
| `ALCHEMY_EVAL_PROMPT_DATA` | `data/hard_set_20_eval.jsonl` | eval 集(hard20) |
| `ALCHEMY_N_EVAL_SAMPLES` | 1 | 每个 eval prompt 采几条 |

## 落盘 / checkpoint
| 变量 | 默认 | 含义 |
|---|---|---|
| `ALCHEMY_RUN_ID` | 时间戳 | run 名,决定 `logs/<id>/` 与默认 ckpt 目录 |
| `ALCHEMY_SAVE_DIR` | `/data/user_data/qixinx/alchemy_runs/<id>/ckpt` | ckpt 落点(`--load==--save`,torch_dist 格式) |
| `ALCHEMY_SAVE_INTERVAL` | 10 | 每多少步存一次 ckpt(每个 ~53G) |
| `ALCHEMY_TRAJ_DIR` | `logs/<id>/traj` | 每 episode 轨迹 json 落点(在 /home,不写 /data) |

⚠️ 多组并行时磁盘易爆:用 `logs/ckpt_janitor_user_data.sh`(supervisor 看护)滚动把 `<latest` 的 ckpt 转 HF→上传→删本地,只留 resume 点。

## 日志 / wandb
| 变量 | 默认 | 含义 |
|---|---|---|
| `ALCHEMY_USE_WANDB` | (空=关) | 设 `1` 且有 key 才开 |
| `WANDB_API_KEY` | 取 `~/.wandb.env` 或 `.env` | wandb key(不泄进日志) |
| `WANDB_PROJECT` | **miles-alchemy** | wandb 项目(同 project 内任意 run 可叠加对比) |
| `WANDB_GROUP` | **qwen3-4B** | wandb 分组(仅侧边栏组织;对比不受限于 group)。同一对照建议设同一个 group |
| `ALCHEMY_WANDB_RUN_ID` | (空) | wandb run 名(一般 = `ALCHEMY_RUN_ID`) |

## 启动模式 / ray
| 变量 | 默认 | 含义 |
|---|---|---|
| `ALCHEMY_DRYFAST` | (空) | **空=cuda-graph ON**(正常/真跑:启动慢、解码快);**`1`=cuda-graph OFF**(启动快、解码慢,smoke 用) |
| `ALCHEMY_CUSTOM_CONFIG_PATH` | `examples/alchemy/alchemy_config.yaml` | merge 到 args 的 yaml |
| `RAY_DASH_PORT` / `RAY_GCS_PORT` | 8265 / 6379 | ray 端口 |
| `MASTER_ADDR` | 127.0.0.1 | ray head 地址 |

> 传给 ray worker(rollout 用)的 env:`ALCHEMY_{TRAIN_ACT,TRAIN_WRITE,FREEFORM,NO_MEMORY,NUM_TRIALS_CAP,WRITE_SIGNAL,MEMORY_WINDOW_SIZE,ACT_EXPLORE_BETA,JUDGE_*,TRAJ_DIR}` + `DEEPSEEK_API_*` + `WANDB_API_KEY`。其余只在 host 侧拼 train.py args。

## 非环境变量(在 `alchemy_config.yaml` 里改;env 同名会覆盖)
| 键 | 默认 | 含义 |
|---|---|---|
| `max_turns` | 200 | 每 episode 最大决策步(10 trial × 20 步) |
| `arc_enable_thinking` | false | 关长 CoT(ACT/WRITE 短输出可解析) |
| `wm_gprime` | 4 | transition_acc 时每个 REWRITE 点的 G' 候选数 |
| `wm_fk_cap` | 12 | F_k(每候选打分的 transition 数)上限 |
| `wm_min_fk` | 3 | F_k 少于此则该 boundary 丢弃 |
| `wm_rewrite_keep` | 3 | DEFER+K:每 episode 只在 K 个随机有效 boundary 上训 WRITE |
| `wm_gen_max_tokens` | 384 | reason-then-answer 打分生成上限 |
| `wm_summary_mode` | replace | memory 是唯一跨 trial 载体(每 trial 重置 history,对齐 eval) |
| `wm_memory_window_size` | 1 | 同 `ALCHEMY_MEMORY_WINDOW_SIZE` |
| `write_reward_signal` | transition_acc | 同 `ALCHEMY_WRITE_SIGNAL` |
| `write_improve_k` | 1 | 同 `ALCHEMY_WRITE_IMPROVE_K` |
| `write_k0_mode` | improve | 同 `ALCHEMY_WRITE_K0_MODE` |
| `act_explore_beta` | 0.0 | 同 `ALCHEMY_ACT_EXPLORE_BETA` |
| `train_act` / `train_write` | true / true | 同 `ALCHEMY_TRAIN_ACT/WRITE` |

## 真跑示例(curriculum + window3 + explore β0.3,8 卡)
```bash
SIF=/data/user_data/qixinx/images/miles_dev-202606081341.sif
set -a; source /home/qixinx/miles/.env; set +a          # DEEPSEEK_API_KEY + WANDB_API_KEY
export ALCHEMY_RUN_ID=qwen3-4b-curr950-sig4norm-w3-expl03-r120-e10-XXXX
export ALCHEMY_SHUFFLE=0 ALCHEMY_MEMORY_WINDOW_SIZE=3 ALCHEMY_TP=2
export ALCHEMY_TRAIN_ACT=1 ALCHEMY_TRAIN_WRITE=1 ALCHEMY_WRITE_SIGNAL=downstream_norm_improve
export ALCHEMY_ACT_EXPLORE_BETA=0.3
export ALCHEMY_NUM_ROLLOUT=120 ALCHEMY_ROLLOUT_BATCH_SIZE=8 ALCHEMY_N_SAMPLES=8 ALCHEMY_GLOBAL_BATCH_SIZE=64
export ALCHEMY_SAVE_INTERVAL=20 ALCHEMY_EVAL_INTERVAL=10 ALCHEMY_KL=0.01
export ALCHEMY_PROMPT_DATA=/home/qixinx/miles/examples/alchemy/data/alchemy_train_950_curriculum.jsonl
export ALCHEMY_USE_WANDB=1 WANDB_PROJECT=miles-alchemy WANDB_GROUP=qwen3-4B-curr950-sig4norm-w3-rb8-n8-r120-e10
export ALCHEMY_DRYFAST=1
apptainer exec --nv --bind /data,/home/qixinx "$SIF" bash run_alchemy_qwen3-4B.sh
```
