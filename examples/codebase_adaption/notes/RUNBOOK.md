# codebase_adaption 训练 Runbook

> 目标读者:不熟悉本代码库的人。照本文档操作即可无 bug 启动/续训/监控一次训练。
> 最后更新:2026-07-15。姊妹文档:`CKPT_RESUME_BUGS_0715.md`(ckpt bug 档案)、`DATA_FILES.md`(数据溯源)。

## 1. 这是什么

在 SWE-Bench-CL 的 episode(同 repo 的一串 issue)上,用 GRPO 训练 Qwen3.5-4B 的两种行为:
- **ACT**:多轮改代码解 issue(reward = 成功时 `round(1-(turns-1)/40, 4)`,失败 0——奖励"又对又快");
- **WRITE**:每题结束后重写一段跨题记忆(当前实验 `CODEBASE_TRAIN_ACT_ONLY=1`:照常写/用记忆,但不训练不给 reward)。

分层结构(外→内):
```
scripts/train_*.sh(实验 wrapper: 一个实验一份, 只设 env)
  └─ launch_codebase_adaption_apptainer.sh(起训练容器, 挂补丁 bind)
       └─ run_codebase_adaption_qwen3.5_4B.sh(容器内: ray start + ray job submit train.py)
            └─ miles train.py(colocate: 8 卡上 Megatron 训练 + sglang 推理轮流占卡)
                 └─ 每个 issue 在独立的 clbench issue SIF 沙箱里判分(嵌套 apptainer)
```

## 2. 前置条件(一次性检查)

| 项 | 路径/要求 |
|---|---|
| 节点 | 1 台 8×A100-80G(RTX PRO 6000 也验证过可跑) |
| 训练容器 SIF | `/data/user_data/qixinx/images/miles_dev-202606081341.sif` |
| HF 模型 | `/data/user_data/qixinx/Qwen3.5-4B` |
| torch_dist 参考权重(KL ref) | `/data/user_data/qixinx/Qwen3.5-4B_torch_dist` |
| clbench 代码 | `/home/qixinx/continual-learning-bench`(env `CLBENCH_ROOT`) |
| issue 沙箱 SIF 库 | `/data/user_data/qixinx/clbench/sifs`(env `CLBENCH_SIF_DIR`) |
| python 依赖包 | `/data/user_data/qixinx/miles_pydeps/codebase_py312_clean`(env `CODEBASE_PYDEPS`) |
| wandb key(可选) | `~/.wandb.env` 里 `WANDB_API_KEY=...`(run 脚本自动 source) |
| /scratch 空间 | issue 沙箱建在 `/scratch/qixinx/tmp`,注意定期清理(见 §8) |

## 3. 标准启动

每个实验一个 wrapper(模板:`scripts/train_4b_actonly_6p6.sh`)。启动:

```bash
cd /home/qixinx/miles/examples/codebase_adaption
nohup bash scripts/train_4b_actonly_6p6.sh > logs/<RUN_ID>.console.log 2>&1 &
```

**约定:console log 文件名 = `logs/${CODEBASE_RUN_ID}.console.log`**,重启前把旧的 `mv` 成带后缀的归档名。

启动成功的判据(按出现顺序 grep console log):
1. `[launch] sglang recapture patch ENABLED`(wrapper 开了该补丁时);
2. `Running entrypoint for job raysubmit_...`(ray job 提交成功,这一行含完整 train.py 参数,可核对);
3. `successfully loaded checkpoint from ... at iteration N`(权重加载;全新 run 加载 HF/torch_dist);
4. 首个 rollout 结束:`[codebase_advantage] ACT n=192 mean_r=0.XX`;
5. 首个训练步后:`[alog-dbg] after train() | ... A_log ... sum=-8.712109375e+01`(哨兵,见 §7)。

新实验命名:RUN_ID 用有含义的名字(如 `swecl-4b-actonly-6p6`),别用日期缺省值。

## 4. 断点续训

先杀干净再重启(**杀任何在跑的 job 都需要用户明确批准**):

```bash
# 杀 ray job + 残留 sglang 引擎(注意 pkill 的模式别匹配到你自己的 shell:
# 用 ps -u $USER -o pid,comm | awk '$2 ~ /^sglang::/' 拿显式 PID 再 kill)
ray stop --force   # run 脚本自身也会做一次
mv logs/<RUN_ID>.console.log logs/<RUN_ID>.console.<原因>.log
```

然后按 run 的"年代"选恢复方式:

| ckpt 年代 | 恢复方式 |
|---|---|
| **cpu-offload 年代的 run**(≤2026-07-15 的所有 run,含 swecl-4b-actonly-6p6) | 只能 weights-only:启动前 `export CODEBASE_TRAIN_EXTRA_ARGS="--no-load-optim --no-load-rng"`。优化器状态会重置(Adam step 从 0 计,lr constant 下影响轻微)。**不要尝试全量恢复,会崩**(见 §8-W1) |
| **no-offload 年代的新 run**(wrapper 里 `CODEBASE_NO_OFFLOAD=1`) | 直接重跑同一条启动命令,原生全量恢复,什么都不用加 |

run 脚本 `--load` 指向 SAVE_DIR,自动从 `latest_checkpointed_iteration.txt` 续。

## 5. 参数表(env,全部可在 wrapper 里覆盖)

### 5.1 必须逐实验设置

| env | 作用 | 参考值 |
|---|---|---|
| `CODEBASE_RUN_ID` | 运行名;决定 ckpt/轨迹/日志路径与 wandb run | `swecl-4b-actonly-6p6` |
| `CODEBASE_MODEL_SCRIPT` | miles 模型定义脚本(`scripts/models/` 下) | `qwen3.5-4B.sh`(**缺省是 9B,必须显式设**) |
| `CODEBASE_HF_CKPT` / `CODEBASE_TORCH_DIST` | HF 权重 / KL 参考权重(**缺省都是 9B 路径,必须显式设**) | 见 §2 |
| `CODEBASE_PROMPT_DATA` | 训练 episode jsonl | `data/episodes_6p6_hard.jsonl` |
| `CODEBASE_BASELINE_ARTIFACT` | per-instance baseline(gain=reward−baseline 的分母;**必须与数据/scaffold 同代**) | `data/baseline_4b_textfmt.json` |
| `CODEBASE_NUM_ROLLOUT` | 总 rollout 数 | `120` |
| `CODEBASE_NO_OFFLOAD` | **新实验一律 `1`**(弃用 cpu-offload,见 §8-W1) | `1` |

### 5.2 训练规模/并行(动前先算显存)

| env | 缺省 | 作用 |
|---|---|---|
| `CODEBASE_NGPU` | 自动检测 | 卡数(colocate:训练与 sglang 同一批卡) |
| `CODEBASE_TP` | 2 | tensor 并行;DP = NGPU/TP |
| `CODEBASE_SEQ_LEN` | 32768 | 训练样本最大长度(6p6 实验用 24576) |
| `CODEBASE_MAX_TOK_PER_GPU` | 20480 | 动态 batch 每卡 token 上限(6p6 用 24576) |
| `CODEBASE_GLOBAL_BATCH_SIZE` | 16 | 全局 batch(样本数) |
| `CODEBASE_ROLLOUT_BATCH_SIZE` | 2 | 每 rollout 取多少个 episode(组) |
| `CODEBASE_N_SAMPLES` | 8 | 每 episode 采样数(GRPO 组大小=2×8=16→n=192 样本/rollout) |
| `CODEBASE_SGLANG_MEM` | 0.5 | sglang 静态显存占比 |
| `CODEBASE_LR` / `CODEBASE_KL` / `CODEBASE_TEMP` | 1e-6 / 0.01 / 1 | 学习率 / KL loss 系数 / 采样温度 |

### 5.3 节奏与 eval

| env | 缺省 | 作用 |
|---|---|---|
| `CODEBASE_SAVE_INTERVAL` | 0(不存!) | 每 N 个 rollout 存 ckpt;正式训练务必设(如 8) |
| `CODEBASE_EVAL_INTERVAL` | 3 | 每 N 个 rollout 跑 heldout eval;0/off 关闭 |
| `CODEBASE_EVAL_PROMPT_DATA` | `data/heldout_episodes.jsonl` | eval 数据(tablib/tenacity 19 题) |
| `CODEBASE_N_EVAL_SAMPLES` | 1 | 每条 eval 数据采样数 |
| `CODEBASE_SGLANG_CONCURRENCY` | 512 | episode 并发闸=该值×卡数;**离线全量 eval(千级 episode 一次投放)须设 4**,否则打爆沙箱构建 |

### 5.4 行为开关

| env | 缺省 | 作用 |
|---|---|---|
| `CODEBASE_TRAIN_ACT_ONLY` | 空 | `1`=WRITE 照常写/用记忆但不训练不给 reward(当前实验模式) |
| `CODEBASE_NO_MEMORY` | 空 | `1`=无记忆模式;**仅支持 eval-only**,训练开会直接报错 |
| `CODEBASE_MAX_STEPS_PER_ISSUE` | 40 | 每 issue 最大轮数(也是 reward 公式的分母来源) |
| `CODEBASE_NUM_ACTS_CAP` | 0(不截) | 每 episode 最多做前 N 个 issue(冒烟用) |
| `CODEBASE_TRAIN_EXTRA_ARGS` | 空 | 透传给 train.py 的额外参数(如 weights-only 恢复) |
| `CODEBASE_SGLANG_EXTRA_ARGS` | 空 | 透传给 sglang 引擎的旗标(如 `--sglang-disable-cuda-graph`,排查用) |

### 5.5 补丁开关(launch 脚本层,bind-mount 进 SIF)

| env | 缺省 | 作用 |
|---|---|---|
| `CODEBASE_SGLANG_RECAPTURE_PATCH` | 0 | `1`=权重更新后重录 CUDA graph + 权重指纹仪表(sglang PR #27140 移植)。**建议保持 1** |
| `CODEBASE_MEGATRON_CKPT_PATCH` | 0 | ckpt merge 补丁 v3.1。**归档品,保持 0**;仅被迫退回 offload 且需全量恢复时,按 §8-W1 流程实弹验收后启用 |

### 5.6 wandb 与杂项

| env | 缺省 | 作用 |
|---|---|---|
| `CODEBASE_USE_WANDB` + `WANDB_PROJECT`/`WANDB_GROUP` | 空 | 开 wandb 及归属;`CODEBASE_WANDB_RUN_ID` 固定 run id(续训不断线) |
| `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` | 3600(run 脚本注入) | NCCL watchdog 心跳(见 §8-W4) |
| `CODEBASE_OBJECT_STORE_MEM` | 16e9 | Ray plasma 上限;不设 Ray 会按物理机内存抢 /dev/shm 导致 cgroup OOM |
| `CODEBASE_SAVE_DIR` | `/data/group_data/rl/yuxiaoq/qixinx/codebase_adaption_runs/<RUN_ID>/ckpt` | ckpt 位置(大文件放 /data) |

## 6. 输出位置

| 内容 | 路径 |
|---|---|
| console log | `examples/codebase_adaption/logs/<RUN_ID>.console.log` |
| 训练轨迹 | `logs/<RUN_ID>/traj/train/rollout_N/ep_*.json`(每文件一个 episode:outcomes + write_audit 全量) |
| eval 轨迹 | `logs/<RUN_ID>/traj/eval/heldout/rollout_N/` |
| ckpt | `<SAVE_DIR>/iter_XXXXXXX/` + `latest_checkpointed_iteration.txt` |
| wandb | project `miles-codebase-adaption` |

## 7. 监控要点(健康信号)

1. **A_log 哨兵**:每个训练步后 `[alog-dbg] after train()` 必须恒为 `sum=-8.712109375e+01`(GPU 与 cpu_backup 两行都要)。偏离=权重被优化器写坏,立即停查;
2. **ACT mean_r**:`[codebase_advantage] ACT` 每 rollout 一条;6p6 数据健康带约 0.25-0.36(训练初期 0.13 起步爬升)。突然掉到 ~0 = 权重坏;
3. **首个 rollout 铁律**:resume 后第一个 rollout 的 `rollout/log_probs` 与 `rollout/ref_log_probs` 若差距异常大(~0.3)= ship 给引擎的权重坏了;
4. **`EPISODE ABORTED`**:环境级故障(沙箱构建失败等)被隔离成 env_error 0-reward 样本,训练继续;偶发可忽略,频发查 /scratch 空间与容器;
5. **success% 与轮数**:从轨迹统计(参考本次:success ~40%,成功平均轮数 26→9,reward 增长主要来自轮数压缩);
6. **memory 质量**(ACT-only 下):write_audit 里 memory 长度中位数应稳定 ~1000+ 字符,超短(<150c)占比 ≤1%。

## 8. ⚠️ 已知问题与警告

### W1【最重要】cpu-offload 优化器 = bug 高发区,新实验一律不开

`--optimizer-cpu-offload + --use-precision-aware-optimizer`(HybridDeviceOptimizer)已确证两族 bug:
- **A_log 写坏**(2026-07-14):孤儿 fp32 参数被第一个 optimizer.step 写成垃圾 → 模型变笨。已通过 A_log 改 bf16 修复,但根源代码路径在 HDO 内;
- **ckpt 优化器状态无法恢复**(2026-07-15):保存端把带垃圾 `step` 的 per-rank 列表漏进 common.pt,全量 resume 必然报 `Cannot merge two lists with different lengths` 或优化器第一步崩 `a Tensor with N elements cannot be converted to Scalar`。上游未修(NVIDIA #1842 / radixark PR #66)。

**规矩**:新实验 wrapper 一律 `CODEBASE_NO_OFFLOAD=1`(TE FusedAdam,结构性免疫,ckpt 原生全量恢复)。旧 offload run 续训只用 weights-only。完整机理与补丁 v3.1 状态见 `CKPT_RESUME_BUGS_0715.md`。
显存代价:4B@TP2/DP4 约 +6GB/卡,80G 卡无压力;新配置首个训练步即显存验证。

### W2 sglang 引擎:TP>1 出乱码 + 睡醒后 CUDA graph 陈旧

- 引擎必须 `--rollout-num-gpus-per-engine 1`(sglang #21039,Qwen3.5 TP>1 输出垃圾;已固化在 run 脚本);
- colocate 睡醒/权重更新后需重录 CUDA graph:保持 `CODEBASE_SGLANG_RECAPTURE_PATCH=1`。

### W3 apptainer 沙箱偶发 FATAL("failed to resolve session directory")

宿主 session 目录瞬时争用,非镜像损坏。已有 episode 级故障隔离兜底(该 episode 记 env_error 0 分,训练不倒)。频发时检查 `/scratch` 空间。

### W4 NCCL watchdog 首步偶发误杀

首个训练步偶发通信器竞态导致 SIGABRT(同一步重跑 158s 即过)。已把 `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` 默认调到 3600 兜底;真正的集合通信挂死仍由 180min 超时守护。

### W5 /scratch 沙箱泄漏

clbench 判分沙箱(`/scratch/qixinx/tmp/minisweagent-*`)在异常退出时会残留,量大时百 GB 级。定期人工清理(mtime 限定,先 `ls` 后删;删除动作需用户批准)。

### W6 默认值陷阱

- run 脚本模型缺省是 **9B**(`CODEBASE_MODEL_SCRIPT/HF_CKPT/TORCH_DIST` 三件套必须一起显式设成 4B);
- `CODEBASE_SAVE_INTERVAL` 缺省 **0=不存 ckpt**,正式训练必须设;
- baseline artifact 必须与数据/scaffold 同代(textfmt scaffold 配 `baseline_4b_textfmt.json`),配错则 gain 全错;
- eval 全量投放必须 `CODEBASE_SGLANG_CONCURRENCY=4`;
- **在线 heldout eval 只是 6 题探针,不是官方 19 题口径**:`schedule.py` 的
  `DEFAULT_TRAIN_COUNTS={"tablib":6,"tenacity":7}` 是早期 13/6 切分的遗产,在线 eval 取
  剩余 3+3=6 题 × 5 题序 = 30 trials。当前训练数据全来自 swe_bench_cl,tablib/tenacity 19 题
  整体未见过——报官方数字须对存档 ckpt 离线跑完整 19 题 × 5 题序,别拿探针分数直接对外比。

### W6.4 纪律:任何测试不上训练节点(2026-07-17)

在正在跑训练的节点上并行做容器/压测类测试有干扰风险(当日 v5-16 的 ray dashboard 500 崩溃与
并行嵌套测试时间重合,因果未证实但教训成立)。测试一律用:空闲节点的 backfill 小 job、
或有我们自己 job 的其它节点(pam_slurm_adopt 允许 ssh)。

### W6.5 GLIBC/fakeroot 机制定性(2026-07-17 实测闭环,修订版)

`GLIBC_2.38 not found (required by /.singularity.d/libs/libfakeroot.so)` 的机制**与节点无关**
(v5-16/A100 上也能复现):嵌套场景下(miles SIF 内调宿主 apptainer 起题容器),apptainer
注入的是 **miles 容器内部(Ubuntu 24.04, glibc 2.39)的 libfakeroot(需 GLIBC_2.38)**;
命运由"目标镜像 glibc × 是否 --fakeroot"决定:
- **训练容器(swe_bench_cl)参数无 --fakeroot** → 结构性免疫,任何节点(实测:v5-16 嵌套
  TRAIN 参数通过;7/12 l5-16/l5-20 历史 814 容器零错);
- **探针 eval 容器(codebase 默认参数,带 --fakeroot)**:tablib/tenacity 镜像 glibc 2.41≥2.38
  → 兼容(v5-16 嵌套实测 uid=0 通过);**glibc<2.38 的镜像(如 swecl 2.35)走此参数必崩**
  (v5-16 嵌套实测复现)。
**7/13 全案还原(日志实证)**:崩的是 astropy(swecl 老镜像)——259 题混合 eval 中,
swecl 与 codebase 两类任务并发读写进程级 `CLBENCH_SINGULARITY_EXEC_ARGS` 产生**竞态**,
astropy 容器吃到了 codebase 的 fakeroot 参数而崩;当时无故障隔离,RuntimeError 炸死整个 job,
未跑完的 tablib/tenacity/sympy 因缺样本进补测名单(陪葬,非病灶)。
**遗留雷(未修)**:该环境变量竞态仍在——混合 swecl+codebase 的并发 eval(如 259 全量重测)
仍会踩;届时须串行化或把 exec 参数改为随任务传递。现行"训练+串行6题探针"结构上不触雷。
宿主级(非嵌套)fakeroot 用的是宿主 el9 的 libfakeroot(≤2.34),对 2.35 镜像可用
(n9-20 实测)。备选与死路:bind 覆盖 `/.singularity.d/libs/*` 被 apptainer 拒绝
(`already exists in layout`);完全去 --fakeroot 时 git/写文件可用(沙箱用户身份解包);
donor 二进制存 `/data/user_data/qixinx/clbench/fakeroot_compat/`。
**含义:凡"对老 glibc 镜像(swecl 池)用带 fakeroot 的 eval 参数"的场景(如 259 全量
baseline 重测)都会踩雷,须去 fakeroot 或换宿主级路径;现行训练+6题探针组合安全。**

### W7 ckpt→HF 导出与已训 ckpt 评测

**`--save-hf`(run 脚本默认开启,但对本任务当前不可用)**:2026-07-17 首航实测,AutoBridge
导出报 `'NoneType' object has no attribute 'param_weight'`——根因是 **miles 的自定义 qwen3_5
spec 没有 bridge 映射插件**(`miles_plugins/megatron_bridge/` 只有 nemotron_h)。失败被
try/except 吞掉,**训练零影响**(每次存档多 ~几秒 + 一行 error 日志)。要让它可用需给
qwen3_5 写 bridge 插件(参照 nemotron_h.py)。`CODEBASE_SAVE_HF=off` 可关。
现行 HF 产物路径:raw 转换 + vision 拼接(下方"旧工具警告"一节,iter_31 实测可用);
评测则优先走"Megatron 加载式 eval"(不需要 HF)。

**评测已训 ckpt(不导出 HF)**:用"Megatron 加载式 eval"——`scripts/eval_ckpt_19q.sh`:
weights-only 加载定格 ckpt + `--start-rollout-id 0` + `CODEBASE_EVAL_BEFORE_TRAIN=1` +
`num_rollout=1`,eval 先于一切训练步执行(权重状态与训练态引擎逐位同构),eval 落盘后杀掉
即可(sbatch 模板 `scripts/eval_iter71_19q.sbatch` 已内置落盘检测+算分)。

**旧工具警告**:`tools/convert_torch_dist_to_hf.py` 是 raw 时代工具,只输出语言塔
(VL 模型缺 `model.visual.*`/`mtp.*` 共 312 个张量)——**勿用于 VL 完整导出**;它不碰优化器
键(与 W1 无关,iter_31 实测 A_log 哨兵逐位一致),仅当只需要语言权重做分析时可用。

### W8 提交环境毒变量 + supervise 终态解析(2026-07-18 双事故)

1. **提交侧 shell 的证书变量会毒死容器内全部 https**。sbatch 默认 `--export=ALL`;若提交
   shell(如 Claude 会话)带 `SSL_CERT_FILE=/etc/pki/...`(RHEL9 宿主路径),它进入 Ubuntu
   容器后指向不存在的文件 → python `SSLError(FileNotFoundError)`、wandb Go 侧
   `x509: certificate signed by unknown authority`,train.py 在 init_tracking 即崩。
   防线(已双侧内置):run 脚本与 sbatch 模板开头 `unset SSL_CERT_FILE SSL_CERT_DIR
   CURL_CA_BUNDLE REQUESTS_CA_BUNDLE ...`。新增提交入口时记得带上。
2. **不要 grep `ray job status` 的人类输出判终态**——失败时它打小写 `Job '...' failed`,
   没有大写 FAILED 枚举,解析永远为空 → supervise 误走不可达分支,**健康 run 也会在
   ~10 分钟被误杀**。正确做法(已内置 `_poll_status`):dashboard HTTP API
   `curl http://127.0.0.1:8265/api/jobs/` 的 JSON `status` 字段(大写枚举)。
3. 附带现象:`ray job logs -f` 对已死 job 每次重连都全量重放日志,.out 会以 ~5s 一轮膨胀;
   状态解析修好后循环会在终态 ~75s 内收敛,该现象自然有界。

## 9. 故障速查

| 症状 | 原因 | 操作 |
|---|---|---|
| 加载报 `Cannot merge two lists ...` | 对 offload 年代 ckpt 做了全量恢复 | 加 `CODEBASE_TRAIN_EXTRA_ARGS="--no-load-optim --no-load-rng"` 重启 |
| 优化器第一步崩 `Tensor with N elements ... Scalar` | 同上(垃圾 step 进了 Adam) | 同上;勿启用 v3.1 补丁除非走完实弹验收 |
| rollout 分数骤降至 ~0 / A_log 哨兵偏离 | 权重被写坏 | 立即停,从最近健康 ckpt weights-only 恢复,排查优化器配置 |
| 首个训练步 SIGABRT(watchdog) | NCCL 首步竞态 | 直接重启,大概率一次通过(heartbeat 3600 已默认) |
| `FATAL ... session directory` 后整个 job 死 | 旧版本无故障隔离 | 用最新代码(已含 episode 级隔离);重启 |
| Ray 起不来 / 端口占用 | 上次残留 | run 脚本自带 `ray stop --force`;顽固时手动 kill gcs_server/raylet |
| /dev/shm OOM | plasma 超配 | 确认 `CODEBASE_OBJECT_STORE_MEM` 生效(默认 16GB) |
| 启动即崩 `x509 unknown authority` / `SSLError(FileNotFoundError)` | 提交 shell 的 SSL_CERT_* 混进容器 | 见 W8.1;确认 run/sbatch 的 unset 行还在 |
| supervised run ~10 分钟无故退出, 日志只有"日志尾随断开"刷屏 | 终态解析为空误判不可达 | 见 W8.2;确认 `_poll_status` 走 HTTP API |
