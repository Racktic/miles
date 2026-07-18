# 新集群从零部署 codebase_adaption RL 训练(SWE-Bench-CL 6p6)

> 目标读者:一个在**全新集群**(什么都没装过)上工作的 Claude Code agent。
> 按本文从上到下执行即可把 `swecl-4b-*-6p6` 系列训练跑起来。
> 本文只讲"部署";日常操作/监控/坑详见同目录 `RUNBOOK.md`(先通读它的 §1 架构图和 §6 Warnings)。
> 撰写:2026-07-18,基于 babel 集群(CMU)上验证过的现役配置逐项核实,无凭记忆内容。

---

## 0. 架构总览(先建立心智模型)

```
sbatch (scripts/sbatch_write_ablation.sbatch)
 └─ launch_codebase_adaption_apptainer.sh          ← 宿主机层:准备 bind,进外层 SIF
     └─ apptainer exec --nv miles_dev SIF           ← 第1层容器:Ubuntu24.04 + CUDA + sglang + Megatron
         └─ run_codebase_adaption_qwen3.5_4B.sh     ← ray start + ray job submit → miles train.py
             └─ codebase_rollout.py                 ← import clbench 仓的 task 模块
                 └─ (嵌套)宿主 apptainer 起每题 issue SIF 沙箱  ← 第2层容器:做题/判分
```

两个关键事实:
1. **两层容器嵌套**:外层 miles SIF 里没有 apptainer,靠 launch 脚本把宿主机的
   apptainer 二进制/配置/依赖库 bind 进去,rollout 在容器内调用宿主 runtime 起题容器。
2. **miles 对 clbench 是硬运行时依赖**:`codebase_rollout.py` 直接
   `from src.tasks.swe_bench_cl.task import ...`(训练判分)和
   `from src.tasks.codebase_adaptation.task import ...`(heldout 探针 eval),
   路径由 env `CLBENCH_ROOT` 指定。clbench 仓必须存在且是**含 swecl 扩展的我们自己的版本**,
   不是上游 pgasawa/continual-learning-bench 原版。

---

## 1. 资产清单(每一项:是什么/多大/从哪拿)

| # | 资产 | 大小 | 来源 | 必需性 |
|---|---|---|---|---|
| A1 | miles 代码 | — | `git clone -b swe_cl-memory-rl git@github.com:Racktic/miles.git`(**不要** `--recurse-submodules`,submodule 仅 experimental/swe-agent 用) | 必需 |
| A2 | clbench 代码(含 swecl 扩展) | — | `git clone git@github.com:Racktic/continual-learning-bench.git`(main 分支;2026-07-18 已含全部 swecl 扩展) | 必需 |
| A3 | 外层 miles SIF | 18 GB | 二选一:① 从旧集群拷 `/data/user_data/qixinx/images/miles_dev-202606081341.sif`;② 重建 `apptainer build miles_dev.sif docker://radixark/miles:dev-202606081341`(SIF label 已核实此为原始镜像) | 必需 |
| A4 | 240 题 issue SIF 库 | ~306 GB / 249 个 | HF 私有数据集 `Racktic/swebench_sifs`(下载需有 Racktic 读权限的 token);**兜底重建**:官方镜像逐题 `apptainer build <id>.sif docker://swebench/sweb.eval.x86_64.<instance_id>`(instance_id 里 `__` 换 `_1776_`,见 SWE-bench 官方命名) | 必需 |
| A5 | eval 探针 SIF(tablib/tenacity) | 259+187 MB | 重建即可:`apptainer build tablib.sif docker://pgasawa2/continual-learning-bench:tablib`(tenacity 同理) | 必需(训练内每 8 步探针 eval 用) |
| A6 | Qwen3.5-4B HF 权重 | 18 GB | `hf download Qwen/Qwen3.5-4B`(公开) | 必需 |
| A7 | Qwen3.5-4B torch_dist | 7.9 GB | 由 A6 现场转换(§3.4),不必搬运 | 必需(现场生成) |
| A8 | pydeps 补充包目录 | 166 MB | 现场重建(§3.5):pip --target 装 swebench==4.1.0 等 5 个包 + guard 补丁 | 必需(现场生成) |
| A9 | 训练/eval episode 数据 | ~1 MB | **已随 A1 入库**:`examples/codebase_adaption/data/`(episodes_6p6_hard.jsonl、heldout_episodes.jsonl、baseline_4b_textfmt.json 等,账本见 data/DATA_FILES.md) | 已含 |
| A10 | 240 题池元数据 | 3.1 MB | 随 A2 入库:`data/swe_bench_cl/full.jsonl`(**注意:内嵌 SIF 绝对路径,§4.2 必须改**) | 已含 |
| A11 | 19 题 eval 元数据 | — | 随 A2 入库:`data/codebase_adaptation/final-dataset.jsonl`(image_name 是 docker 标签,经 `CLBENCH_SIF_DIR` 解析到 A5,无需改) | 已含 |

磁盘预算:资产合计 ~350 GB;另给 ckpt 预留 ≥1 TB(每个保存点含 optimizer state,几十 GB 级),
日志/轨迹 ~几十 GB/run。

---

## 2. 新集群前置核查(一条不过就先解决,别硬跑)

```bash
# ① 计算节点 GPU:8 卡单节点。已验证代数:A100-80G(v5-16)和 RTX PRO 6000 Blackwell。
nvidia-smi | head -15
# ② 宿主 apptainer(计算节点上!):≥1.4,且三件套路径存在(launch 脚本按此 bind,不同则改 §4.3)
apptainer --version; ls /usr/bin/apptainer /usr/libexec/apptainer /etc/apptainer
# ③ 用户命名空间 + fakeroot(eval 探针容器要用;训练容器不用 fakeroot,天然免疫 GLIBC 坑)
grep "^$USER" /etc/subuid || echo "无 subuid 条目 → --fakeroot 会失败,见 RUNBOOK W6.5"
apptainer exec --fakeroot docker://ubuntu:24.04 id   # 期望 uid=0
# ④ 节点本地 scratch(题容器沙箱解包在这,IO 密集,必须节点本地盘)
df -h /scratch 2>/dev/null || echo "无 /scratch → §4.4 改 SCR"
# ⑤ 内存:申请尽量大(babel 上用 --mem=0 拿整节点;cpu-offload 已弃用,但 sglang+判分并发仍吃内存)
# ⑥ 出网:HF 下载、docker:// 拉镜像、wandb 上报(不通则 WANDB_MODE=offline)
curl -sI https://huggingface.co | head -1
```

---

## 3. 部署步骤

### 3.1 目录约定(建议照抄,减少改路径)

```bash
export DEPLOY_ROOT=/path/to/your/storage        # 大容量共享存储
# 代码
git clone -b swe_cl-memory-rl git@github.com:Racktic/miles.git                $HOME/miles
git clone git@github.com:Racktic/continual-learning-bench.git                 $HOME/continual-learning-bench
# 资产目录
mkdir -p $DEPLOY_ROOT/{images,swebench_sifs,clbench_sifs,models,miles_pydeps}
```

### 3.2 SIF 三类

```bash
# 外层 miles SIF(A3;重建方式,约 1h)
apptainer build $DEPLOY_ROOT/images/miles_dev-202606081341.sif docker://radixark/miles:dev-202606081341
# 题库 SIF(A4;HF 下载,306GB,建议 hf CLI + HF_HUB_ENABLE_HF_TRANSFER=1)
hf download Racktic/swebench_sifs --repo-type dataset --local-dir $DEPLOY_ROOT/swebench_sifs
# eval 探针 SIF(A5)
apptainer build $DEPLOY_ROOT/clbench_sifs/tablib.sif   docker://pgasawa2/continual-learning-bench:tablib
apptainer build $DEPLOY_ROOT/clbench_sifs/tenacity.sif docker://pgasawa2/continual-learning-bench:tenacity
```

下载完 A4 必须做完整性核对(240 题一个都不能缺):
```bash
python3 - <<'PY'
import json, os
d = os.environ["DEPLOY_ROOT"] + "/swebench_sifs"
missing = []
for l in open(os.path.expanduser("~/continual-learning-bench/data/swe_bench_cl/full.jsonl")):
    name = os.path.basename(json.loads(l)["image_name"])
    if not os.path.exists(f"{d}/{name}"): missing.append(name)
print("缺失:", len(missing), missing[:5])
PY
```
缺哪个就按 A4 的兜底命令单独重建哪个。

### 3.3 模型权重

```bash
hf download Qwen/Qwen3.5-4B --local-dir $DEPLOY_ROOT/models/Qwen3.5-4B
```

### 3.4 HF → torch_dist 转换(在外层 SIF 里做,一次性)

```bash
apptainer exec --nv --bind $DEPLOY_ROOT,$HOME $DEPLOY_ROOT/images/miles_dev-202606081341.sif bash -c '
  cd $HOME/miles && source scripts/models/qwen3.5-4B.sh &&
  PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint '$DEPLOY_ROOT'/models/Qwen3.5-4B \
    --save '$DEPLOY_ROOT'/models/Qwen3.5-4B_torch_dist'
```
(官方文档:`docs/models/qwen/qwen3-5.md` §3.2。)

### 3.5 pydeps 补充包目录(在外层 SIF 里 pip --target)

SIF 底座刻意不含 swebench 等包;它们住在独立目录,由 run 脚本 PYTHONPATH 前置注入。
背景与 guard 机制:`notes/TRAINING_ENV_AND_SWEBENCH_DEPS.md`(必读,尤其 §3)。

```bash
PYD=$DEPLOY_ROOT/miles_pydeps/codebase_py312_clean
apptainer exec $DEPLOY_ROOT/images/miles_dev-202606081341.sif \
  pip install --no-deps --target=$PYD \
  swebench==4.1.0 mini-swe-agent==2.2.4 litellm==1.81.6 unidiff==0.7.5 fastuuid==0.14.0
```
装完**必须**按 TRAINING_ENV_AND_SWEBENCH_DEPS.md §3 给 swebench 的
`harness/__init__.py` 打 import-guard(不打的话 import swebench 会因缺 bs4/docker-py 直接炸)。
验证:
```bash
apptainer exec $DEPLOY_ROOT/images/miles_dev-202606081341.sif \
  env PYTHONPATH=$PYD:$HOME/continual-learning-bench:$HOME/miles \
  python -c "import swebench, litellm, minisweagent, unidiff; from src.tasks.swe_bench_cl.task import SweBenchCLTask; print('IMPORT_OK')"
```

### 3.6 wandb

```bash
echo 'WANDB_API_KEY=<你的key>' > ~/.wandb.env   # run 脚本自动 source;不上报就不建此文件
```

---

## 4. 路径适配(新集群必改清单——就这四处,其余全是 env 可覆盖)

### 4.1 环境变量(在 sbatch/wrapper 里 export,不用改源码)

| env | babel 现值 | 新集群设为 |
|---|---|---|
| `CODEBASE_MILES_SIF` | /data/user_data/qixinx/images/miles_dev-202606081341.sif | $DEPLOY_ROOT/images/… |
| `CLBENCH_ROOT` | /home/qixinx/continual-learning-bench | clbench clone 路径 |
| `CLBENCH_SIF_DIR` | /data/user_data/qixinx/clbench/sifs | $DEPLOY_ROOT/clbench_sifs(tablib/tenacity 所在) |
| `CODEBASE_PYDEPS` | /data/user_data/qixinx/miles_pydeps/codebase_py312_clean | $PYD |
| `CODEBASE_HF_CKPT` | /data/user_data/qixinx/Qwen3.5-4B | $DEPLOY_ROOT/models/Qwen3.5-4B |
| `CODEBASE_TORCH_DIST` | /data/user_data/qixinx/Qwen3.5-4B_torch_dist | $DEPLOY_ROOT/models/Qwen3.5-4B_torch_dist |

### 4.2 clbench `data/swe_bench_cl/full.jsonl` 的内嵌 SIF 绝对路径(唯一的数据改动)

```bash
cd ~/continual-learning-bench
sed -i "s|/data/group_data/rl/yuxiaoq/qixinx/swebench_sifs|$DEPLOY_ROOT/swebench_sifs|g" data/swe_bench_cl/full.jsonl
```
(miles 侧的 episodes_6p6_hard.jsonl 已核实**零绝对路径**,不用动。)

### 4.3 `launch_codebase_adaption_apptainer.sh` 两处

- L41 `--bind /data,/home/qixinx` → 改成新集群的存储挂载点 + home;
- L17-32 `HOST_LIBS`:这是宿主 apptainer 二进制的动态库依赖(RHEL9 的 /lib64 路径)。
  新集群先跑 `ldd /usr/bin/apptainer 和 ldd /usr/libexec/apptainer/bin/*` 比对,发行版不同(如 Ubuntu 宿主)则按实际路径改;
  库缺失时脚本会显式报 "Missing host Apptainer library",按报错补即可。

### 4.4 train wrapper 脚本(scripts/train_4b_write_*.sh)

- `SCR=/scratch/qixinx` → 新集群的节点本地盘;
- `CODEBASE_HF_CKPT/TORCH_DIST` 两行按 §4.1 改;
- sbatch 模板 `scripts/sbatch_write_ablation.sbatch` 的 partition/qos/account/`--mem=0` 按新集群 SLURM 改
  (核心诉求:8 GPU + 尽量多 CPU(babel 用 128)+ 尽量大内存 + 3 天墙钟)。

---

## 5. 验证阶梯(每级过了再上下一级,全绿才准跑正式实验)

```bash
# L1 外层 SIF + GPU
apptainer exec --nv $CODEBASE_MILES_SIF python -c "import torch;print(torch.cuda.device_count())"  # =8
# L2 嵌套容器链路(纯 CPU):scripts/test_nested_fakeroot.sh 把头部三个路径变量改成新集群值后执行
#    期望: BUILD_OK / TRAIN_GIT_OK / TRAIN_WRITE_OK;EVAL 段若 GLIBC 报错见 RUNBOOK W6.5(老镜像+fakeroot 的已知坑)
# L3 python 依赖链(§3.5 末尾的 IMPORT_OK)
# L4 数据完整性(§3.2 末尾的 缺失: 0)
# L5 单节点冒烟:直接用 delta wrapper 起 1 个 rollout 验证全链路
CODEBASE_NUM_ROLLOUT=1 bash scripts/train_4b_write_delta_6p6.sh
#    盯三件事(RUNBOOK §2 的启动检查点):
#    a. "successfully loaded checkpoint";b. rollout_0 的 env_error 率≈0(非零说明容器链路有问题);
#    c. WRITE 样本 n>0 且 write_signal 字段与 reward mode 匹配
```

## 6. 正式启动(三组 WRITE-reward 消融为例)

```bash
cd ~/miles/examples/codebase_adaption
sbatch --job-name swecl-4b-write-delta-6p6      scripts/sbatch_write_ablation.sbatch scripts/train_4b_write_delta_6p6.sh
sbatch --job-name swecl-4b-write-downstream-6p6 scripts/sbatch_write_ablation.sbatch scripts/train_4b_write_downstream_6p6.sh
sbatch --job-name swecl-4b-write-gated-6p6      scripts/sbatch_write_ablation.sbatch scripts/train_4b_write_gated_6p6.sh
```
三组均:从 base 4B 起训、6p6 数据、`CODEBASE_NO_OFFLOAD=1`(**铁律:新实验一律不开 cpu-offload**,
见 notes/CKPT_RESUME_BUGS_0715.md)、`CODEBASE_RAY_SUPERVISED=1`(sbatch 模板已设,防 ray dashboard
抖动误杀训练)。区别仅 `CODEBASE_WRITE_REWARD_MODE`:delta / downstream / gated_downstream。

## 7. 未决事项(部署前必须先解决)

1. ~~clbench 远程仓~~ 已解决(2026-07-18):swecl 扩展版已全量推到
   `github.com/Racktic/continual-learning-bench` main 分支(commit d9ca86d)。
2. **HF 数据集核验**:`Racktic/swebench_sifs` 是私有仓,部署 agent 需持有 Racktic 读权限 token;
   下载后务必跑 §3.2 的完整性核对(babel 本地目录 251 个文件中含 pull.log/validate.sh/一个旧命名
   重复件,有效题 SIF 覆盖 240 题池即可,不必追求文件数逐一相等)。
3. 训练细节、监控命令、全部已知坑:`RUNBOOK.md`(W1-W7 一个都别跳过)。
