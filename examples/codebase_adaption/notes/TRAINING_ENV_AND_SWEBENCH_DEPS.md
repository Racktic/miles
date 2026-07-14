# 训练环境的两层结构 + swebench 缺依赖的坑

> 读者:在 miles 里跑 codebase_adaption / swe_bench_cl 训练的人。
> 记录训练环境到底怎么组装的,以及一个只在"训练环境 + django/sympy 被解出"时才引爆的
> swebench 缺依赖坑,和我们的修法(为什么安全)。

---

## 1. 训练环境是"两层叠加",不是单一 SIF

```
apptainer exec --nv  miles_dev-202606081341.sif        ← 第 1 层:SIF 镜像(底座,不可变)
   提供 python3.12 + torch / megatron / sglang / ray + requests 等基础栈
   └─ PYTHONPATH 前置 /data/user_data/qixinx/miles_pydeps/codebase_py312_clean   ← 第 2 层:补充包目录
        提供 swebench、mini-swe-agent、jinja2 等 clbench 专用依赖(散文件,可编辑)
```

- 组装点:`launch_codebase_adaption_apptainer.sh:52`(exec 进 SIF) +
  `run_codebase_adaption_qwen3.5_4B.sh:26-28`(把 `CODEBASE_PYDEPS` 前置进 PYTHONPATH)。
- **swebench 只住在第 2 层那个目录里**,SIF 底座里没有(`apptainer exec SIF python -c 'import swebench'`
  会 ModuleNotFoundError)。所以改 swebench = 改那个目录,**不动 SIF 镜像**。
- 这个目录名字里的 `clean` = 当初刻意精简过,只装训练/评分必需的,**故意没装 swebench 的可选重依赖**
  (GitHub 抓取的 bs4/ghapi/fastcore、docker-py 等)。

判断某个包在哪一层:`apptainer exec SIF python -c 'import X; print(X.__file__)'`
→ 路径含 `dist-packages` = SIF 底座;含 `miles_pydeps/codebase_py312_clean` = 第 2 层。

---

## 2. 坑:swebench 的 __init__ 急切导入了用不到的重依赖

### 症状
训练中某个 **django/sympy** 题被 agent 解出并 submit → 触发官方评分 →
`from swebench.harness... import` → 整个 job 崩:
```
ModuleNotFoundError: No module named 'bs4'      (或 docker.errors)
  swebench/__init__.py → from swebench.collect.build_dataset import ...
  swebench/harness/__init__.py → import docker_build → import docker.errors
```

### 根因(不是 bug,是环境缺依赖)
- 评分只需要 swebench 的**纯解析/spec 模块**(`harness.constants/grading/log_parsers/test_spec`)。
- 但 swebench 打包时 `swebench/__init__.py` 和 `swebench/harness/__init__.py` **急切导入了一切**,
  包括 `collect`(GitHub 抓取,要 bs4/requests/ghapi)和 `docker_build/docker_utils/run_evaluation`
  (要 docker-py)。Python 导入任何 swebench 子模块都会先跑这两个 __init__ → 被缺失的可选依赖拖垮。

### 为什么以前没遇到(三条件缺一不可)
1. **环境**:离线 baseline/eval 用 clbench 的**完整 `.venv`**(依赖齐全);这是第一次在**精简的第 2 层**里跑官方评分。
2. **要真提交**:评分只在 agent 解出并 submit 时才跑。3 步 smoke 从没解出 → 评分路径没走到 → 潜伏。
3. **只有 django/sympy**:只有这俩走 import swebench 的官方评分(`swe_bench_cl/task.py:_OFFICIAL_GRADING_REPOS`);
   其余 6 repo 走 returncode 评分,不碰 swebench。

所以是 **warmup 简单题(有胜率)+ 40 步(够时间解出)+ 抽到 django 被解出** 三者叠加才第一次引爆。

---

## 3. 修法:guard 两个 __init__(不删、只容错)

把两个 __init__ 里"评分用不到的可选重依赖"导入包成 `try/except ImportError: pass`,
缺了就跳过,评分要的解析模块照常加载。改的文件(第 2 层目录内):
- `swebench/__init__.py`         —— 原版备份 `swebench/__init__.py.orig`
- `swebench/harness/__init__.py` —— 原版备份 `swebench/harness/__init__.py.orig`
- 另装了 `unidiff`(评分真需要、纯 python 无重依赖):`pip install --no-deps --target=<pydeps> unidiff`

回滚:`cp *.orig` 覆盖回去 + 删 `unidiff*` 目录即可。

### 为什么安全:被 guard 的包评测运行时用不到(三重验证)
1. **代码**:`swe_bench_cl/official_grading.py::evaluate_official_submission` 运行时只调
   `MAP_REPO_VERSION_TO_SPECS / get_test_directives / make_test_spec / MAP_REPO_TO_PARSER /
   get_eval_tests_report / get_resolution_status`(全在**非 guard** 的解析模块里)+ 容器操作走
   **我们自己的 `generic_runtime`(apptainer)**,从不引用 collect / docker_build / docker_utils /
   run_evaluation / versioning.get_versions。
2. **导入**:这些解析函数不需要 bs4/ghapi/docker-py 就能 import;唯一的 `import requests`
   由 SIF 底座满足。
3. **运行时**:debug-4 处理了 160 处 django/sympy、跑完整个 rollout+训练零报错,评分实跑通过。
4. **eval 更无关**:codebase heldout eval 走父类 `CodebaseAdaptationTask` 的 returncode 评分,
   **根本不 import swebench**。

### 备选(没采用)及原因
往第 2 层 `pip install bs4 docker requests ghapi fastcore` 让 `import swebench` 原样可用——
但 docker-py/requests 会拉 urllib3 等**传递依赖**,而 ray/sglang/wandb 也用 urllib3 →
有版本冲突、污染训练栈的风险。故选 guard(零新增重依赖)。若日后重建 pydeps,可考虑装全 +
锁版本,或给训练单独一个装全的 pydeps。

---

## 4. 如果重建/更换 pydeps 环境

guard 的改动在 pydeps 目录里,**重建会丢**。重建后若又在训练里跑到 django/sympy 官方评分,
会再撞这个坑。届时:重新应用本文 §3 的 guard(或装全依赖)。这也是把它记下来的原因。
