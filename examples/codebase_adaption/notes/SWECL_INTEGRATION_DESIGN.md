# SWE-Bench-CL → miles `codebase_adaption` 训练接入设计

> 目标:把我们在 continual-learning-bench 构造的 **swe_bench_cl 训练池(232 题)** 接进 miles 的
> `codebase_adaption`(ACT/WRITE co-evolving 记忆协同训练)作**训练集**,原来的 **19 道
> codebase_adaptation 题作测试集(heldout)**。本文是开发前的设计与改动清单。
>
> 关联:算法参考实现见 `../alchemy/`;数据构造/清洗见
> `/home/qixinx/continual-learning-bench/notes/swe-bench-cl-*.md`。

---

## 0. 范围与不改的东西

- **改**:数据来源(train 换成 swe_bench_cl 232 池)、task 选择(train 用 swe_bench_cl task)、
  episode 排布(9+10 pair 覆盖)、baseline artifact(覆盖新题)、run 脚本/config 指向。
- **不改**:ACT/WRITE 双流 GRPO 算法、rollout 主循环、advantage、prompts、metrics——已核实**接线正确、无致命 bug**(WRITE 训练样本在 `codebase_rollout.py:641-648` 正确打了
  `downstream_trial_pos`,分组不塌缩;`notify_change` 由 clbench `task.py:1046-1057` gate 在
  `changed`,只在 repo 切换那题提示一次)。

---

## 1. 背景:现有数据流(一句话)

- miles `data/{train,heldout}_episodes.jsonl` 只是 **split 触发器**(占位 prompt + `metadata.split`);
  真正的 19 题 id **硬编码在 `schedule.py:18-48 DEFAULT_STAGE_IDS`**(tablib 9 + tenacity 10)。
- `codebase_rollout.py:346-369 _make_task()` 运行时 import `CodebaseAdaptationTask`,
  `dataset_path = <clbench_root>/data/codebase_adaptation/final-dataset.jsonl`,
  把 `EpisodeOrder` 的 `instance_ids` 塞进 `task._schedule_instance_ids`,用 clbench 当环境/判分。
- `split_stage_ids()`(`schedule.py:76-95`)按 `DEFAULT_TRAIN_COUNTS={tablib:6,tenacity:7}`
  在**每段内部**切:train=前 13,heldout=后 6。
- **ACT reward** = `1−regret/40`(clbench `task.py:630-660`);**WRITE reward** = 下游 gain 改善
  `R(M_k)=mean(gain[k+1..k+K])−mean(gain[k−K+1..k])`,`gain=reward−baseline`,baseline 从
  `codebase_baseline_artifact` 按 instance_id 读(`schedule.py:185-220`,缺 id **静默返回 0**,
  见 `codebase_rollout.py:547`)。
- **GRPO 关键不变量**:`n_samples_per_prompt=8` 个 sibling 共享 `group_index` →
  `shuffle_seed=offset+group_index` 相同 → **同一题序**;同一下游位置面对同一道题,
  题目难度作为公共项被组内均值抵消。**接数据时必须保持"8 个 sibling 重玩同一 episode"。**

---

## 2. 关键设计决策

### 2.1 双 task 切分:train=swe_bench_cl,eval=codebase(**必须**)

**为什么不能把 232 题塞进 codebase 的 dataset**:swe_bench_cl 里 django/sympy 走**官方 SWE-bench
判分**(`SweBenchCLTask._evaluate_submission`,swe_bench_cl `task.py:117-133`),而
`CodebaseAdaptationTask` 只用 pytest returncode 判分——会把 django/sympy(232 池里占 ~94 题)判错。

**做法**:`_make_task` 按 split 选 task 类 + dataset。`SweBenchCLTask` **继承**
`CodebaseAdaptationTask`,`reset/step/_next_query/instance_outcome/reward` 全通用,只
`_evaluate_submission` 不同 → **干净 drop-in**。

```python
# codebase_rollout.py _make_task(args, split, instance_ids, stage_lookup)
if split == "train":
    from src.tasks.swe_bench_cl.task import SweBenchCLTask
    dataset_path = root / "data/swe_bench_cl/train_pool_avg0.6.jsonl"   # 232 池(已过滤简单题)
    task = SweBenchCLTask(dataset_path=str(dataset_path), schedule="all",
                          max_steps_per_issue=..., seed=...)
else:  # heldout / eval
    from src.tasks.codebase_adaptation.task import CodebaseAdaptationTask
    dataset_path = root / "data/codebase_adaptation/final-dataset.jsonl"  # 19 题
    task = CodebaseAdaptationTask(dataset_path=str(dataset_path), schedule="default",
                                  max_steps_per_issue=..., seed=...)
task._schedule_instance_ids = list(instance_ids)
task._schedule_stage_lookup = stage_lookup
```

### 2.2 train episode 来源:预生成 9+10 episode 池(alchemy 风格)

现在"1 个 train 种子 + shuffle 固定 13 题池"多样性太差。改成 **alchemy 那种"每行一个预建
episode"**(alchemy 的 `data/alchemy_episodes.jsonl` 就是每行一个固定 episode)。

- 生成 `data/swecl_train_episodes.jsonl`,每行 = 一个 **9+10 episode**:
  ```json
  {"prompt":"", "metadata":{"split":"train","episode_index":0,
     "repoA":"astropy","repoB":"pydata",
     "instance_ids":["astropy__...",...9 个..., "pydata__...",...10 个...],
     "stage_labels":["astropy",...×9,"pydata",...×10]}}
  ```
- 生成逻辑**复用** `continual-learning-bench/scripts/swecl/build_sequences.py` 的选题(wrand 分层随机,
  难度来自 `baseline_qwen35_9b.avg4`)+ 段内难度优先拓扑,从 **232 池** 按 repo pair 采样。
- **pair 覆盖**:先覆盖 9 对中强共享知识对(见 sequence-design.md §1),双向 + 每对多次不同采样,
  可再掺少量负对照对增强鲁棒。episode 数目 N 作参数(初期 ~50–100)。
- **GRPO 不变量**:每行是一个 prompt,`n_samples_per_prompt=8` 个 sibling **重玩同一 episode**
  (同 instance_ids 同序)→ 分组正确。段内是否再洗序 = 可选(默认保留 curriculum 序)。
- `schedule.py` 的 train 分支从 **seed 的 `metadata.instance_ids/stage_labels`** 直接构造
  `EpisodeOrder`(不再走 `DEFAULT_STAGE_IDS`);heldout 分支保持现状(codebase 19 题 + order_rank)。

> 备选(不推荐):保留 seed+shuffle 机制,把 `DEFAULT_STAGE_IDS` 换成 swe_bench_cl 单一 pair。
> 缺点:训练只见一对 repo,多样性差。故采用预生成 episode 池。

### 2.3 合并 baseline artifact

`gain=reward−baseline`,baseline 缺 id 会静默取 0(`codebase_rollout.py:547`),悄悄改 WRITE 口径。
- 造 `data/baseline_merged.json`,`instance_outcomes=[{instance_id, reward}]` 覆盖
  **232(swe_bench_cl)+19(codebase)= 251 题**(id 不冲突)。
- swe_bench_cl 侧 reward 用我们
  `continual-learning-bench/results/swe_bench_cl/qwen35_9b_baseline/passk_detail.jsonl` 的
  **`avg_reward`(avg@4)**;codebase 侧沿用现有 baseline.json。
- config `codebase_baseline_artifact` 指向合并后文件。(train/eval 共用一个文件即可,因 id 不冲突。)

### 2.4 容器镜像

swe_bench_cl 各题的 `image_name`(SWE-bench 官方镜像)要在 miles 集群以 apptainer/singularity SIF
可用(类比 alchemy 的 `upload_swebench_sifs.sh`)。判分环境:`CLBENCH_CONTAINER_BACKEND=singularity`、
`MSWEA_SINGULARITY_EXECUTABLE=apptainer`,与我们跑 baseline 时一致。**这是 infra 前置,需先盘点/准备。**

---

## 3. 逐文件改动清单(dev plan)

| 文件 | 改动 |
|---|---|
| `codebase_rollout.py` | `_make_task` 增 `split` 参数,按 2.1 选 task+dataset;train 分支 import `SweBenchCLTask` |
| `schedule.py` | 加 `build_episode_order_from_meta(instance_ids, stage_labels)`(train 走 seed metadata);`stage_lookup_for` 里 `REPO_BY_STAGE` 扩到 8 个 swe_bench_cl repo;heldout 逻辑不动 |
| `data/swecl_train_episodes.jsonl` | **新增**,预生成的 9+10 episode 池(2.2) |
| `data/baseline_merged.json` | **新增**,251 题合并 baseline(2.3) |
| `codebase_config.yaml` | `codebase_baseline_artifact` → 合并文件;新增 `codebase_train_episodes`(路径)、可选 `codebase_write_improve_k` 调大 |
| `run_codebase_adaption_qwen3.5_4B.sh` | `--prompt-data` train 指向 `swecl_train_episodes.jsonl`;heldout 不变;确认 clbench_root/backend 环境变量 |
| (continual-learning-bench) `scripts/swecl/gen_train_episodes.py` | **新增**,从 232 池生成 episode 池 + 合并 baseline(复用 build_sequences 逻辑) |

---

## 4. 配置改动(`codebase_config.yaml`)

```yaml
codebase_clbench_root: /home/qixinx/continual-learning-bench
codebase_train_dataset: data/swe_bench_cl/train_pool_avg0.6.jsonl   # 新:train 用
codebase_train_episodes: examples/codebase_adaption/data/swecl_train_episodes.jsonl  # 新
codebase_baseline_artifact: examples/codebase_adaption/data/baseline_merged.json     # 改:251 题
codebase_write_improve_k: 2        # 建议:SWE 域单题 reward 噪声大, K 从 1 提到 2~3
codebase_write_k0_mode: improve
codebase_max_steps_per_issue: 40
codebase_memory_max_tokens: 2048
```

---

## 5. 不变量与坑(开发时必须守住)

1. **GRPO sibling 同 episode**:8 个 sibling 必须重玩同一 episode(同 instance_ids)。预生成 episode
   池天然满足(一行一个 prompt)。**别在 sibling 间引入不同题序**,否则 WRITE 下游分组与 ACT trial_pos
   分组都会错位。
2. **baseline 全覆盖**:合并 artifact 必须含全部 251 id,否则缺 id gain=reward,WRITE 口径漂移。
   开发后加一条断言:episode 里每个 instance_id 都能在 baseline map 命中,缺失就报错(不静默 0)。
3. **grading 路由**:train 必须是 `SweBenchCLTask`(django/sympy 官方判分),eval 是
   `CodebaseAdaptationTask`。别混。
4. **环境 parity**:swe_bench_cl 用它自己的 singularity args(见 continual-learning-bench
   `swe_bench_cl/task.py`),与我们跑 baseline/ICL 时一致,保证 reward 口径可比。
5. **WRITE 末题无下游**:最后一题的 memory 无下游、不训练(`codebase_advantage.py:37 range(len-1)`),
   正常。
6. **简单题已过滤**:train 用 `train_pool_avg0.6.jsonl`(avg@4>0.6 的 8 道已去掉),这些贴天花板的题
   gain≈0、`gain[k+1]−gain[k]` 全噪声,过滤后 WRITE 信号更干净。难题(0/4)保留。

---

## 6. 调参 / 开放问题

- **WRITE window K**:建议从 1 提到 2–3(SWE 单题 reward 噪声比 alchemy 大)。用 metrics 里
  `write_group_zero_std_frac` 和 WRITE mean_r 监控。
- **train episode 数 N / pair 覆盖**:初期 ~50–100,覆盖 9 对中强对(双向);是否掺负对照对训练?
  (co-evolving 主要学 procedural 迁移,掺负对照可能提升泛化——待实验。)
- **段内是否洗序**:默认保留难度 curriculum 序;可 ablation 洗序看 WRITE 是否更稳。
- **是否也留 swe_bench_cl 的 held-out sequence 做第二个 eval**(除 19 codebase 外),看 in-domain
  持续学习增益。后续可加。

---

## 7. 冒烟验证(开发后、上大规模前)

1. **数据自检**:`gen_train_episodes.py` 跑完,断言每 episode 9+10、instance_ids 全在 232 池、
   baseline map 全覆盖 251。
2. **单 episode dry-run**:`num_acts_cap` 设小(如 3),train 跑 1 个 episode,确认:
   - train 用了 SweBenchCLTask(打印 task 类名);django/sympy 题走官方判分(看日志)。
   - ACT/WRITE sample 数、phase、`downstream_trial_pos` 正常(advantage 的 `[codebase_advantage]`
     打印 ACT/WRITE n 与 mean_r)。
   - gain 非零且 baseline 命中(无"baseline missing"警告)。
3. **eval 不回归**:heldout 仍是 19 codebase 题、走 CodebaseAdaptationTask、order_rank 排列。
4. **小规模 1–2 step 训练**:确认 loss/grad 正常、WRITE 流有非零优势(非全 0)。

---

## 8. 开发顺序建议

1. (continual-learning-bench) 写 `scripts/swecl/gen_train_episodes.py` → 产
   `swecl_train_episodes.jsonl` + `baseline_merged.json`,先跑数据自检(§7.1)。
2. (miles) 改 `_make_task`(split→task)+ `schedule.py`(train 走 metadata)。
3. (miles) 改 config + run 脚本指向新数据。
4. 盘点/准备 swe_bench_cl 容器 SIF(§2.4)。
5. 冒烟(§7.2–7.4)→ 调 K/N → 上规模。
