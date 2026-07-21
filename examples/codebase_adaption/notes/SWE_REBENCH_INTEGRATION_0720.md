# SWE-rebench V2 接入笔记(2026-07-20)

> 训练主粮候选(DATA_PIPELINE_NEXTGEN_0720.md 定位)。本文记录 P0 进度、判分链
> 定型逻辑、以及逐坑调试出的 5 个 apptainer/环境陷阱(接 task 类时照抄,勿重踩)。

## 数据与选题

- 源:HF `nebius/SWE-rebench-V2`(32K/3617 repo/20 语言,CC-BY-4.0);parquet 落
  `/data/user_data/qixinx/swe_rebench/raw`。Python 子集剔 tablib/tenacity 后 7,229 题/690 repo。
- 自带难度(LLM 评审,easy<15min/medium 15-60min/hard>1h)+ B1-B11 缺陷标记 + confidence,
  在 `meta.llm_metadata`。分布 easy 2214/medium 4638/hard 354。**是先验非实测,120 题先导用于校准**。
- 先导选题:`select_pilot.py` → 120 题/111 repo(48/48/24 分层,pytest 族 parser、每 repo≤3、时序均匀)。
  产物 `/data/user_data/qixinx/swe_rebench/pilot/{pilot_manifest.jsonl, image_list.txt}`。

## 判分链(gold 冒烟 3/3 resolved 验证:isort 46/46, babel 29/29, cookiecutter 15/15)

代码:`clbench/src/tasks/swe_rebench/grading.py`
- `parse_log_pytest` + `normalize_test_name`(剥 ANSI/计时后缀)+ `grade`(**严格相等**:
  实际 PASSED 集 == 归一(F2P+P2P);多一个 PASSED 也判负)——逐字移植官方 eval.py。
- `build_eval_script(base_commit, install_steps, test_cmds, patch_files)` 生成容器侧判分脚本,
  封装下方验证过的**6 步前置**(顺序 load-bearing)。

判分流程(每步都是一个被调试出来的失败模式):
1. `export HOME=/tmp`——镜像把 HOME 设成宿主不存在的路径,裸 `cd` 会失败;
2. 发现 repo 目录——repo 在 `/<basename>`(不是 /testbed),`.git` 在 depth≤3;
3. `git reset --hard <base_commit>`——**镜像 build 后工作树是脏的**(editable 安装等),
   不 reset 则 `git apply` 报 "does not match index";
4. **跑 `install_config.install`**——每仓的环境修正藏在这里(如 babel 的
   `sed 's/[pytest]/[tool:pytest]/' setup.cfg`);漏跑则 pytest 配置解析失败,是我第一版的根 bug;
5. `git apply` model+test 双补丁(官方 `--3way --recount --ignore-space-change --whitespace=nowarn`);
6. 跑 `test_cmd` → 输出喂 `grade()`。

## apptainer/环境 5 坑(接 task 类照抄)

1. **`--writable` 沙箱 + `--bind` 挂载点不存在 → FATAL**:writable 无 overlay,不自动创建挂载点。
   解:补丁文件直接 `cp` 进沙箱目录(如 `<sb>/patches/`),不用 bind。
2. **内联 bash 拼字符串被二次展开**:`$(ls /)` 塞进 for 列表撑爆语法。解:判分逻辑写成独立
   脚本文件拷进沙箱执行,不内联。
3. **repo 目录**:`/<repo basename>`,兜底 `find / -maxdepth 3 -name .git`(排除 /tmp /patches /proc)。
4. **脏工作树**:必须 `git reset --hard base_commit`(见流程步 3)。
5. **install 步骤不可省**(见流程步 4)——这是判分正确性的一部分,不是可选优化。

## 进度(P0)

- [x] 选题 120 题、判分核移植+单测、gold 冒烟 3/3、判分脚本生成器入库+单测
- [x] `SweRebenchTask`(继承 generic runtime,`_evaluate_submission` 走 build_eval_script+grade,
      install_config 随 instance 带入,无 fakeroot)+ codebase_rollout.py split=rebench 路由
- [x] miles eval-only(eval_rebench.sh + rebench_eval.jsonl 单题 no_memory,同 bl-4b-textfmt 路线);
      **端到端冒烟通过**(3 题真机做题+判分)
- [x] **4B 可解性 baseline → 决策门通过**(52 题×avg@4,v5-16 抢占 4×A100):
      **easy pass@4 37% / 样本成功率 21.5%,RL 黄金区间 → rebench 可训练**
- [~] 镜像拉取:120 pilot 仅 55 成功(失败=mksquashfs 偶发+并发过高假失败,非镜像缺失;
      全量 Python 子集 7,229 镜像/~7.2TB,训练只需 ~2,880 题子集)
- [ ] 过门后:全量序列切分(每 repo 时序、序列间零共享,B8-B11 题剔除)+ 训练管线

**逐坑调试史(端到端接入必踩,已全清)**:①`--writable`+`--bind` 挂载点不存在→FATAL(补丁直拷沙箱);
②内联 bash 被二次展开→独立脚本文件;③repo 发现=`/<basename>`;④脏工作树→`git reset --hard base`;
⑤缺 `install_config.install`(babel setup.cfg sed);⑥**GLIBC_2.38/fakeroot→禁 --fakeroot**(W6.5);
⑦`num_instances` 契约→显式声明;⑧eval 落盘目录固定名 `traj/eval/heldout/`(非 rebench),wrapper 曾
等错目录提前拆(数据未丢,已知)。

工件目录:`/home/qixinx/swe_rebench_workdir/`(select/pull/smoke 脚本)、
`/data/user_data/qixinx/swe_rebench/`(raw parquet、pilot manifest、sifs)。
相关:[[DATA_PIPELINE_NEXTGEN_0720]] · RUNBOOK W8(提交防线)。
