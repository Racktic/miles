# SWE-Bench-CL 训练数据接入 miles —— 运行交接

> 读者假设:你已经在 miles 里跑过很多次 alchemy 和 codebase_adaption 的 train/test。
> 这份文档只讲**增量**:我往 codebase_adaption 里接了一套新的 **SWE-Bench-CL** 训练数据,
> 怎么用它跑实验,以及这套数据的正确性是怎么验证的。miles/GRPO/apptainer 启动方式**一律不变**。

---

## 0. TL;DR

- **train split 现在跑 SWE-Bench-CL**(231 道真实 SWE issue 的难题池),**test/eval split 完全不动**(还是原来的 codebase 19 题 heldout)。
- 启动方式**和你跑 codebase_adaption train 一模一样**(同一个 `run_codebase_adaption_qwen3.5_4B.sh`、同一套 apptainer 启动、同样的嵌套 SWE 容器)。**唯一变化是数据文件和 baseline artifact 的默认值已经指到新数据**,所以你直接按老方式启动即可。
- 想验证链路先用 §4 的 smoke 旋钮跑 1 步;正式训练把旋钮拉满即可。

---

## 1. 我接进了什么(增量清单)

### clbench 侧(`/home/qixinx/continual-learning-bench`)
| 文件 | 改动 | 作用 |
|---|---|---|
| `src/tasks/swe_bench_cl/` (新) | 新 task `SweBenchCLTask(CodebaseAdaptationTask)` | train split 用它取题。只 override 评分路由(django/sympy 走官方 grading,其余 returncode)+ 三个环境修复钩子 |
| `src/tasks/codebase_adaptation/task.py` | 加 `_repo_test_hint` / `_extra_singularity_exec_args` 两个钩子(默认 no-op) | 让子类做**每-repo 范围**的环境微调,不影响 codebase 原路径 |
| `src/tasks/container_backend.py` | `create_interactive_environment` 加 `extra_exec_args` 形参 | 支持给特定 repo 注入 `--env`(如 matplotlib 关 rerunfailures) |
| `data/swe_bench_cl/` (新) | `full.jsonl`(240)、`train_pool_avg0.6.jsonl`(231) | 清洗后的题池 + 难题过滤池 |
| `scripts/swecl/` (新) | 数据构造/富化/过滤/生成 episode 脚本 | 见 §2 |

### miles 侧(`examples/codebase_adaption/`)
| 文件 | 改动 | 作用 |
|---|---|---|
| `codebase_rollout.py` | `_make_task(split=...)`:train→`SweBenchCLTask`(用 episode 的 `instance_ids`+`stage_labels` 精确取题);其它→`CodebaseAdaptationTask`(不变) | train/test 双 task 切换 |
| `schedule.py` | `REPO_BY_STAGE` 加 8 个 swe_bench_cl repo→org 映射(astropy/pytest/sphinx/scikit/matplotlib/django/sympy/xarray) | reset 时按 stage 正确切库 |
| `codebase_config.yaml` | `codebase_split: train`、`codebase_train_dataset: data/swe_bench_cl/full.jsonl`、`codebase_baseline_artifact: .../baseline_merged.json` | 默认走 train split + 新 baseline |
| `run_codebase_adaption_qwen3.5_4B.sh` | `--prompt-data` 默认 → `data/swecl_train_episodes.jsonl`;`CODEBASE_BASELINE_ARTIFACT` 默认 → `data/baseline_merged.json` | 见 §5 已修坑① |
| `data/swecl_train_episodes.jsonl` (新,54) | 训练 episode 池 | 见 §3 |
| `data/baseline_merged.json` (新,250) | gain 的 baseline(231 swecl@avg@4 + 19 codebase) | gain=reward−baseline |

---

## 2. 数据来源 & 清洗

- **来源**:SWE-Bench-CL(8 个 repo 的真实 GitHub issue + F2P/P2P 测试)。
- **清洗**(`scripts/swecl/`):去掉畸形 pytest node(如参数化空串 `test_x[]`),使 “单个 returncode over pytest node 列表” 的判分与逐-test F2P/P2P 等价;django/sympy 不走 pytest,单独走官方 grading。
- **镜像**:每题用官方 SWE-bench 镜像的绝对 `.sif` 路径(dataset 里 `image_name` 就是绝对路径)。
- **`full.jsonl`(240)**:每行富化了 `baseline_qwen35_9b: {solved4, pass1, pass4, avg4, rewards}`(qwen3.5-9B × pass@4 baseline)。
- **`train_pool_avg0.6.jsonl`(231)**:从 240 里**过滤掉 baseline avg@4 > 0.6 的简单题**(这些题没什么训练价值、易引入噪声),再删掉环境有问题的 scikit-14710。**难题保留**(留给 memory/ICL 去啃)。

---

## 3. Episode 池 & baseline(training 用的两个文件)

`scripts/swecl/gen_train_episodes.py` 生成(纯 CPU,可本地空跑):

- **`swecl_train_episodes.jsonl`(54)** = 9 对 repo × 2 方向 × 3 reps。每 episode 是一条 **9+10=19 题的 CL 序列**(段1 学 repoA 9 题 → 段2 迁移到 repoB 10 题),`metadata.instance_ids` + `metadata.stage_labels` 精确指定题目和分段。
  - `prompt` 字段是 miles 要求的 **chat-message list**(`[{"role":"user","content":[{"type":"text",...}]}]`),内容只是占位——rollout 忽略它,自己按 instance_ids 建 chat。见 §5 已修坑②。
- **`baseline_merged.json`(250)** = 231 swecl(reward=avg@4)+ 19 codebase(原 baseline reward)。**必须全覆盖** episode 里用到的所有 instance_id,否则缺 id 会静默取 0 → gain=reward,污染 WRITE reward。

**自检**(gen 脚本跑完自动做):每 episode 9+10、无重复、id 全在池内、**baseline 覆盖全部 216 个 episode id、0 缺失**。改数据后重跑 gen 脚本即可重新自检。

---

## 4. 怎么跑实验

### 4.1 启动(和 codebase_adaption train 完全一样)
用**你平时跑 codebase_adaption train 的那套启动方式**(apptainer exec miles_dev.sif + 你的 bind/环境),跑 `run_codebase_adaption_qwen3.5_4B.sh`。数据默认值已经指到新数据,**不需要额外指定 `--prompt-data` / baseline**。

> ⚠️ 见 §5 已知未决:我自己用 `apptainer exec miles_dev.sif` 裸包装时,嵌套 SWE 容器报 `apptainer: command not found`(容器内无 apptainer)。**你的 codebase-train 启动法已经能起子容器(旧 run 日志里确认过 208 次成功),所以用你的启动法,别用我的**。

### 4.2 smoke(验证链路,1 步就够)
起训练前先用最小配置验证 `rollout→reward→gain→WRITE→advantage→一步 Megatron`:

```bash
export CODEBASE_NGPU=2 CODEBASE_TP=2
export CODEBASE_NUM_ROLLOUT=1 CODEBASE_ROLLOUT_BATCH_SIZE=1 CODEBASE_N_SAMPLES=2 CODEBASE_GLOBAL_BATCH_SIZE=2
export CODEBASE_NUM_ACTS_CAP=2        # 每 episode 只做 2 道 issue(≥2 才有 WRITE reward)
export CODEBASE_MAX_STEPS_PER_ISSUE=3 # 每题 agent 只 3 步
export CODEBASE_EVAL_INTERVAL=0       # 关 heldout eval
export CODEBASE_SAVE_INTERVAL=0       # 不存 ckpt
# ...然后按你的方式启动 run_codebase_adaption_qwen3.5_4B.sh
```
关键约束:`GLOBAL_BATCH_SIZE ≤ NUM_ROLLOUT × ROLLOUT_BATCH_SIZE × N_SAMPLES`,否则 train_iters 取整为 0、LR scheduler assert 挂。

### 4.3 正式训练
去掉上面的 smoke 旋钮(或按需):`NUM_ACTS_CAP=0`(每 episode 全 19 题)、`MAX_STEPS_PER_ISSUE=40`、`N_SAMPLES=8`、`NUM_ROLLOUT` 设你要的步数、`NGPU=8 TP=2`、`EVAL_INTERVAL` 打开(eval 走原 codebase heldout,不变)。

### 4.4 容器 scratch
每道 issue 会起一个嵌套 SWE apptainer 沙盒,量大。把 `TMPDIR`/`APPTAINER_TMPDIR`/`APPTAINER_CACHEDIR` 指到 `/scratch`(大)或 `/dev/shm`(RAM,1T),别让 `/tmp` 打满。

---

## 5. 我已经修掉的两个坑(smoke 帮我抓到的,已在位)

1. **baseline artifact 被旧默认值遮蔽**:run 脚本原来把 `CODEBASE_BASELINE_ARTIFACT` 默认导出成旧的 codebase-only trace,而 `_load_baselines`(`codebase_rollout.py:74`)**env 优先于 YAML** → 216 个 swecl 训练题 baseline 全取 0、gain=reward、WRITE reward 被污染。**已改**:run 脚本默认改成 `${SCRIPT_DIR}/data/baseline_merged.json`。
2. **episode prompt 格式**:原来写成空字符串 `""`,但 qwen3.5 加载了 processor,miles 的 `Dataset`(`miles/utils/data.py:219`)要求 `prompt` 是 **list**,空串直接 assert 挂。**已改**:gen 脚本输出 chat-message list。
3. **train exec-args 泄漏到 eval**:`SweBenchCLTask` 把 `CLBENCH_SINGULARITY_EXEC_ARGS`(SWE-bench 的 testbed-first PATH、`LANG`、且**无 `--fakeroot`**)设成**进程级全局**,`default_singularity_exec_args()` env 优先读它 → 同进程的 codebase eval 容器也会读到 → **codebase eval 悄悄用了 SWECL 参数而非自己 tested 的默认(带 `--fakeroot`+镜像自带 PATH),行为漂移**。(注:并非"python 消失"——泄漏的 PATH 含 `/usr/local/bin`,当前 tablib/tenacity 镜像 python 仍 resolve;问题是未测试的参数漂移 + 对未来镜像不安全。)**已改**:`codebase_rollout._make_task` 模块加载时快照 `_ORIG_SINGULARITY_EXEC_ARGS`,eval 分支**恢复到快照**(orig=None → 回落后端默认;用户显式 override → 保留,不再无条件删除);下轮 train 由 `setdefault` 重设。详见 [EVAL_EXEC_ARGS_LEAK_FIX.md](EVAL_EXEC_ARGS_LEAK_FIX.md),回归测试见 `tests/test_eval_exec_args.py`。**残留设计味**:仍靠进程级 env 传 task 专属参数(依赖 miles train/eval 串行不并发),彻底做法是把 exec args 改成 per-task 传参(未做)。

---

## 6. 数据正确性是怎么验证的

- **环境健康(关键)**:matplotlib/django/sympy 三个 repo 用 **Opus 4.6** 重测 26 题 baseline,**零环境崩溃 / 零超时 / 零 “No module named pytest”**;三个针对性修复都验证生效:
  - matplotlib:`-p no:rerunfailures`(pytest-rerunfailures 在 `--contain` 下 bind localhost 崩溃)→ 修好,10 题全程不崩;
  - django/sympy:注入原生 runner 提示(`tests/runtests.py` / `bin/test`),避免 agent 反射性 `pytest`;
  - 3 道 0 分题经轨迹核对为 `eval_status=tests_failed`(评分正常跑测试后判失败)= **agent 交错 patch,真难,非环境问题**。
- **baseline 富化**:`full.jsonl` 每题 avg@4/pass@4 来自 qwen3.5-9B pass@4 baseline;难题过滤阈值 avg@4≤0.6。
- **episode/baseline 自检**:见 §3(每 episode 9+10、id 全在池、baseline 0 缺失)。
- **CL 迁移信号**:qwen3.5-9B × 3 条 CL sequence(带记忆 icl,每条 5 run)显示段2 跨-repo 迁移有正 gain(procedural knowledge 迁移),数据具备训练价值。详见 clbench `results/swe_bench_cl/qwen35_9b_sequences/`。

---

## 7. 一条未查清的事(留给你)

我没能定位:**miles_dev 容器内没有 apptainer 二进制**(`command -v apptainer` = NONE、全盘 find 无),但旧的 codebase-train run 日志里子容器**确实起了 208 次**。我在 host 直接跑 run 脚本时 `ray: command not found`,包进 miles_dev 又缺 apptainer——两条路我都没走通,但**你的启动法显然是通的**。所以:请直接用你已知能跑通 codebase-train 子容器的那套启动方式;搞清楚它到底怎么让嵌套 apptainer 生效(host 直跑?容器里 bind 了 apptainer?别的 SIF?),就能把上面的 smoke 也跑通。
