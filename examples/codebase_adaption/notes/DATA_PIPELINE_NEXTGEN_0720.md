# 下一代训练/评测数据管线(SWE-Chain · SWE-rebench V2 · R2E-Gym-Subset)

> 2026-07-20。动机:WEIGHT_MEMORIZATION_VS_SEQ_CL_0720.md 证据链判定**数据层是唯一瓶颈**。
> 新数据铁律(用户拍板):①序列间零共享题目;②repo 多样性(权重学不透 → 记忆保值);
> ③同 repo 序列按演化时序排列;④与现训练池 8 仓(django/sympy/astropy/matplotlib/
> scikit-learn/xarray/pytest/sphinx)及 heldout(tablib/tenacity)零重叠;⑤4B 可解性过门才用。

## 0. 已淘汰候选(存档)

| 数据集 | 死因 | 可携带资产 |
|---|---|---|
| SWE-EVO LongChain-50 | 链=重叠窗口,跨链共享 3×(89 题位仅 ~30 去重点) | Python 30 SIF 已转(/data/user_data/qixinx/swe_evo/sifs,待处置) |
| SWE-Milestone/DeepCommit | 序列判据过,但剔 scikit-learn 污染后 Python 存量=0 | DAG+拓扑解锁的序列化设计 |
| SWE Context Bench | 星形深度 1,链被拍平;33% 同 PR 平凡对;与池重叠 66% | **依赖边挖掘配方**(6 型引用+递归+人工核验;需剔同PR对、保留链中间节点) |

## 1. 三活跃数据集:特征 × 定位 × 流程 × 进度

### 1.1 SWE-Chain(定位:heldout 序列内 CL 点火评测)

**特征**:9 包全 Python,12 链/155 过渡;剔 pytest/xarray(池重叠)后 **8 链/106 过渡,7 个干净包**
(attrs/conan×2/flask/jinja2/poetry/pyjwt/urllib3)。链=版本升级序列(每步 next_ver=下步 prev_ver,
物理连续);任务=按 spec 实现该次 release 的变更(~9 条要求/步),**非修 bug**;判分=上游 next_ver
原生测试集 F2P/E2P 六分类。跨链零共享(155 三元组全唯一,实测)。Apache-2.0。
量级小 → 只做评测不做训练。

**转换目标**:接入 miles eval 路径,对 ckpt 跑"有记忆 vs 无记忆"双协议 → 序列内点火测量
(检验 D 的评测版)。每步从 gold prev_ver 起跑(判分独立,记忆只携带知识;agent 工作区连续演化=二期)。

**流程与进度**:
- [x] 环境(babel 无 docker 路线):3 个底座 SIF(py3.9-slim/py3.10-slim/py3.10-bullseye,
  按官方各链 Dockerfile 对齐)+ 每链 venv(忠实执行 Dockerfile 安装行,--system 改道 venv,
  SETUPTOOLS_SCM_PRETEND_VERSION 注入)。**8/8 链 import+测试收集通过**
  (attrs 1180/flask 467/jinja2 322/pyjwt 175/poetry 1373/urllib3 2307/conan 4994+5720);
  版本切换机制验证通过(attrs 21.3.0→23.1.0 重装)。产物:/data/user_data/qixinx/swe_chain/
  (base SIF + chains/<id>/{src,venv,BASE_SIF,install.log,smoke.log})
- [x] 环境坑存档:jinja2 需 setup.cfg `[pytest]→[tool:pytest]` 覆盖(官方 harness 同款反作弊
  配置覆盖,属评测侧环境准备);冒烟勿用 `2>/dev/null|tail`(吞退出码假阳性,已中过一次)
- [ ] 判分接入:移植 oracle 测试集(官方 GH metadata/)+ 逐步版本切换 + spec→题面 + 六分类计分
- [ ] **可解性门**:2 条链 × 4B——不过门则锁定"仅评测",并评估 spec 拆子任务的降粒度改造
- [ ] 对 downstream/v3nothink 终版 ckpt 跑首次点火测量(时间窗:两 run 收官前就位)

### 1.2 SWE-rebench V2(定位:训练主粮)

**特征**(全部实测核验):32,079 题/3,617 repo/20 语言;**Python-only 剔 tablib 后 7,229 题/690 repo**;
created_at 无缺失无并列(时序排序无歧义);真实 issue 文本;双 pass 验证 F2P/P2P + 难度/置信度元数据;
CC-BY-4.0(23.6% 上游 repo license 标 custom,内部训练无碍);与池零重叠(SWE-bench 家族被主动排除)。
**序列产能:L=6 → 943 条 / L=12 → 362 条互不相交时序序列**(需求 240+,达标)。
镜像 docker.io/swerebenchv2/*,每题一个(0.8-1.4GB),匿名可拉。判分 test_cmd+命名 log_parser,
与 swe_bench_cl 同语义(接入最顺)。注意:宣称月更未兑现(2026-03 后无增量);created_at 实为字符串。

**镜像规模(全量 parquet 实数)**:每题一个唯一镜像,无共享。全 rebench V2 = 32,079 镜像(~32TB);
**Python 子集 = 7,229 镜像(690 repo,~7.2TB)**。训练不需全拉:240 条 L=6 序列 ≈ 2,880 题 ≈ ~2.9TB。

**转换目标**:P0 先导(单题分层抽样 → 镜像转 SIF → 4B avg@4 可解性 baseline)→ 过门后全量序列切分
(train 序列 + 独立 heldout 序列,序列内时序、序列间零共享)→ 接入训练管线(episode=序列,复用
swe_bench_cl 双 task 架构)。

**流程与进度**:
- [x] 调研+产能核算完成(parquet 全量 428MB 已扫)
- [x] P0 选题:select_pilot.py → 120 题 pilot(easy/medium/hard 分层,pytest 族 parser,每 repo≤3);
      manifest+镜像清单在 /data/user_data/qixinx/swe_rebench/pilot/
- [x] **判分链**:src/tasks/swe_rebench/grading.py(parse_log_pytest + 严格集合相等判据 +
      build_eval_script);**gold 冒烟 3/3 resolved**(isort 46/46, babel 29/29, cookiecutter 15/15)
- [x] **miles 集成**:SweRebenchTask(继承 codebase generic-PR runtime,加 install_config + 严格判分 +
      无 fakeroot)+ codebase_rollout.py 的 split=rebench 路由 + eval episode 生成;**端到端冒烟通过**
- [x] **P0 可解性 baseline → 决策门通过**(base 4B,52 题×avg@4,v5-16 抢占 4×A100 eval-only):
      **easy pass@4 18/48=37% / 样本成功率 21.5%**;medium 4题样本太小(0/4)。**~20% 样本成功率
      落在 RL 黄金区间**(组内有方差)→ **rebench 确认可训练**
- [~] 镜像拉取:120 pilot 仅 55 成功。**失败非镜像缺失**:mksquashfs 偶发故障 + 我并发过高(4路)
      造成大量假失败(手动重拉 tornado 等 FAIL 项均成功)。教训:建全量池时用带认证 docker + 串行 +
      大 TMPDIR + 对真 mksquashfs 失败者走官方镜像重建
- [ ] 全量序列切分(每 repo 时序、序列间零共享、按镜像可拉性+可解层+剔 B8-B11 筛)+ 训练管线

**接入笔记详**:notes/SWE_REBENCH_INTEGRATION_0720.md(判分 6 步前置 + apptainer 坑清单 + 逐坑调试史)。

### 1.3 R2E-Gym-Subset(定位:辅助/参考系)

**特征**:4,578 题全 Python/仅 10 repo(pandas 1444...coveragepy 108),与池及 SWE-bench 家族零重叠;
commit_date 全有(抽验与 GitHub 一致);产能 L=6→758/L=12→376 条;**DeepSWE/SkyRL 已在其上跑通 RL**
(verl PPO、rollout.n=8、clip_high 0.28、无 KL);SkyRL 有难度分桶衍生集
(NovaSky-AI/r2egym-{trivial,easy,medium,hard})。判分=expected-output 映射(接入中等)。
数据卡未标 license(de-facto 广泛使用;orange3 上游 GPL,再分发注意)。
**弱点:仅 10 repo → 权重会学透仓库知识,记忆保值性差**——故不作主粮。

**用法**:①难度分桶方法与 DeepSWE 超参做参考系/校准;②必要时抽深仓序列混入训练做"深 vs 广"消融;
③prepull_images.py 镜像预拉脚本可借。当前无转换计划,按需启用。

## 2. 执行顺序(2026-07-20 拍板)

**双线并行**:rebench P0(训练关键路径,长周期后台活)优先;SWE-Chain 判分接入(代码活)穿插,
目标在 downstream/v3nothink 收官前就位,给终版 ckpt 做首次序列内点火测量。

相关:[[WEIGHT_MEMORIZATION_VS_SEQ_CL_0720]](动机与铁律来源)· [[WRITE_PROMPT_V3_AB_0719]] ·
RUNBOOK W8(所有新 sbatch 的提交防线)。
