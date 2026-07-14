# 给跑 smoke 的 agent:一个已合入的 eval 修复(exec-args 泄漏)

TL;DR:我在 `codebase_rollout.py` 的 `_make_task` eval 分支恢复了一个进程级 env,修掉"train 的容器 exec 参数会泄漏到 eval 容器"的问题。**和你改的 `--fakeroot`(在 `swe_bench_cl/task.py`)不同文件、不冲突**。

> 本文经跑 smoke 的 agent 复核,已按其反馈更正两处描述不准确,并把无条件 pop 改成"保留用户显式 override"。感谢复核。

---

## Bug 是什么(泄漏是事实)
`SweBenchCLTask.__init__`(`swe_bench_cl/task.py:111`)用 `setdefault` 把 SWECL 的 exec 参数写进**进程级** `CLBENCH_SINGULARITY_EXEC_ARGS`。这串包含 testbed-first 的 `--env PATH=...`、`LANG`,以及(你改后)**没有 `--fakeroot`**。

`container_backend.default_singularity_exec_args()`(`:65`)**env 优先**读它;真正消费它的是 **interactive env 创建** + **grading 原语 `singularity_exec`**(`singularity_start_container` 只 build sandbox,不读 exec args)。

后果:同进程里 train(SWECL)跑过后,`CLBENCH_SINGULARITY_EXEC_ARGS` 一直是 SWECL 那串;等 **eval split 跑 codebase 容器**时也会读到 → codebase eval 容器**不是用 codebase 自己 tested 的默认参数(带 `--fakeroot` + 镜像自带 PATH),而是悄悄用了 SWECL 的参数**。

### 更正:不会"python 消失"
之前我写"eval 里 python/pytest 消失",**不准确**。泄漏的 PATH 本身含 `/usr/local/bin`,而当前两个 codebase eval 镜像(tablib / tenacity)的 python/pytest 就在 `/usr/local/bin`,仍能 resolve。所以对**当前 19 道 eval 题不是直接 breakage**。真正问题是:**codebase eval 容器的运行参数发生了未经测试的漂移**(尤其丢了 `--fakeroot`、PATH 顺序变了),而且对"python 不在泄漏 PATH 上"的未来镜像不安全。

## 修复(已合入,override-safe)
`codebase_rollout.py`:
1. 模块加载时(任何 task 动 env 之前)快照原始值:
   ```python
   _ORIG_SINGULARITY_EXEC_ARGS = os.environ.get("CLBENCH_SINGULARITY_EXEC_ARGS")
   ```
2. `_make_task` 的 eval 分支恢复到该快照(而非无条件 pop):
   ```python
   if _ORIG_SINGULARITY_EXEC_ARGS is None:
       os.environ.pop("CLBENCH_SINGULARITY_EXEC_ARGS", None)   # 回落后端默认
   else:
       os.environ["CLBENCH_SINGULARITY_EXEC_ARGS"] = _ORIG_SINGULARITY_EXEC_ARGS  # 保留用户显式 override
   ```
- eval → 恢复 train 之前的状态(None 则 codebase 用后端默认;用户显式设过则保留其值)。
- 下一轮 train → `SweBenchCLTask.__init__` 的 `setdefault` 再次生效。
- **更正**:原来无条件 `pop` 会删掉用户显式传入的 `CLBENCH_SINGULARITY_EXEC_ARGS`,与 `setdefault` "保留显式 override"的承诺矛盾(当前 launcher 没设它,故对现配置无影响,但这是真边界问题)。现已改为恢复快照。

## 为什么安全 + 怎么验证的
- **依赖前提**:miles 主循环 `train.py:70` 里 train-generate 和 eval-generate 逐个 `await`、严格串行、不并发,所以 eval 时改 env 不会打断还在 exec 的 train 容器。已核对成立。
- 同阶段 siblings 并发调用 `setdefault`/恢复都是幂等的,无竞争。
- **状态机实测**(miles_dev 容器内):train=SWECL(testbed PATH,无 fakeroot)→ eval=codebase(默认,带 fakeroot,无 testbed 泄漏)→ 下轮 train 恢复 SWECL。断言全过。

## 和你的 `--fakeroot` 改动的关系
不同文件、不同关注点,无冲突。恢复逻辑对"SWECL 串带不带 fakeroot"完全无所谓。

## 残留(未做,记一笔)
根子仍是"用进程级 env 传 task 专属参数",只是把泄漏堵住 + override 保护好了,依赖 train/eval 串行。彻底做法是把 exec args 作为 **task 实例参数**传递(要串 `container_backend` + `generic_runtime` + 两个 task),未做。若改成并发 async 训练需要上这个。
