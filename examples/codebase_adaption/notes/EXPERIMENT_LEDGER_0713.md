# 实验台账(截至 2026-07-13,只记录有证据的事实)

## 一、实验时间线(每个 run 的客观结果)

日志根目录:`/home/qixinx/miles/examples/codebase_adaption/logs/`(下表路径均相对于此)。
轨迹 dump 在同名子目录 `<RUN_ID>/traj/train/rollout_N/ep_*.json`;wandb 本地在 `/home/qixinx/miles/wandb/run-*-<RUN_ID>`。

| # | run | 配置要点 | 结果(数字) | 终结原因 | console log | 轨迹 traj |
|---|---|---|---|---|---|---|
| 1 | fix2(7/11) | 4B,512G,5 步验证 | raw_reward: **0.151 → 0.037/0.011/0.041/0.017** | 正常跑完 5 步 | `swecl-4b-route2-seq20480-fix2.console.log` | `swecl-4b-route2-seq20480-fix2/traj/` |
| 2 | formal run1(7/12) | 4B,1600G | rollout_0: ACT 0.2052,completed **73/160**;rollout_1-3: ACT **0.0000**,completed **0/160**,advantage **0.0** | step3 surrogate 序列化崩溃(\udb00) | `swecl-4b-formal-r120.collapsed.log`(备份:`collapsed-run-archive-0713/console.log`) | `collapsed-run-archive-0713/traj/`(rollout_0-3 全) |
| 3a | 9B 试跑 ①(7/12) | l5-16,754G,Blackwell | 架构跑通;内存峰值 751/754 撑过 3 步 | step3 surrogate 崩溃(\ud834) | `swecl-9b-formal-r120.surrogate-crash.log` | `swecl-9b-formal-r120/traj/` |
| 3b | 9B 试跑 ②(7/12) | l5-20 重投(带 surrogate 修复) | step1 完成后 rollout 中 host RAM 顶满 | **OOM exit 137**(754G 这次没扛住;上次 l5-16 是运气) | `swecl-9b-formal-r120.console.log`(内存曲线:`swecl-9b-formal-r120.memtrace.csv`) | 同上目录(被 ② 复用) |
| 4 | formal run2(7/13) | +clean_act 修复 | rollout_0: ACT **0.2258**,completed **79/160**(略好于修复前) | rollout_1 期间 RolloutManager 膨胀 **1.29TB** 被 OOM 杀(dmesg) | `swecl-4b-formal-r120.obsbomb.log` | `collapsed-run-archive-0713/traj-run2-obsbomb/` |
| 5 | formal run3(7/13) | +obs cap 8000 | rollout_0: 0.2264/76;**rollout_1: ACT 0.0000**(崩塌依旧);内存平稳 475G | 主动停(信号已死) | `swecl-4b-formal-r120.console.log`(现存最新即 run3) | `swecl-4b-formal-r120/traj/`(rollout_0/1) |
| 6 | 金丝雀#1(lr=0) | 权重零更新,2 步 | rollout_0: 0.2234;**rollout_1: ACT 0.0000**,raw=0.018421052631578946(与 run3 rollout_1 完全相同——ACT 全 0 后由 baseline 决定的确定性常数) | 正常跑完 | `swecl-4b-lr0-canary.console.log` | `swecl-4b-lr0-canary/traj/` |
| 7 | 金丝雀#3(权重体检) | lr=0 + --check-weight-update-equal | 首次传输后 compare:**文本模型张量全部相等;visual.\* 297 个未被覆盖**(失配名单可 `grep 'name=' log` 提取) | compare 失败即抛异常终止(设计如此) | `swecl-4b-lr0-checkw.console.log` | `swecl-4b-lr0-checkw/traj/`(未到 rollout) |

其他相关日志(非本表实验,备查):
- 更早的 9B/4B 调参与烟测:`swecl-{4b,9b}-route2-seq{20480,24576}*.console.log`、`swecl-9b-warmup-debug-{1..4}.console.log`、`swecl-smoke-*`
- run1 的另一份中途备份:`swecl-4b-formal-r120.surrogate-crash.log`、`swecl-4b-formal-r120.surrogate2.log`(run2 崩后备份)
- 离线 ICL 两组(clbench 侧):augment `/data/user_data/qixinx/clbench/logs/eval-4b-iclsum-9244712.out`、replace `...-9244713.out`;轨迹 `~/continual-learning-bench/results/codebase_adaptation/traces/2026-07-12T16-25-00.{763712,746359}Z/`

## 二、确凿事实(证据在括号里)

1. **每个 run 都在第一次权重更新周期后 ACT 崩塌**:rollout_0 健康(ACT≈0.21-0.23,completed 73-79/160),rollout_1 起 ACT=0、completed=0、advantage=0(runs 2/5/6 一致;fix2 的 raw 曲线同型)。
2. **崩塌与权重数值无关**:lr=0(权重严格不变)照崩(金丝雀#1)。
3. **文本模型的权重传输是正确的**:抹掉后重传,逐张量比对全部相等(金丝雀#3)。
4. **visual.\*(视觉塔,297 张量)不被权重传输覆盖**——ckpt 是 VL 模型,Megatron 纯文本训练。这是事实;**但它是否有害未证明**,反例见"已排除"第 4 条。
5. **崩塌后的输出病理**:thought 文字连贯(还在正确分析报错),command 异常(`pwsh`/`nc -u` 等);57% turn 空命令;94% issue 烧满 40 轮。
6. **约束解码行为指标在睡醒前后剧变**:`finish=stop 且 JSON 未闭合` 从 rollout_0 的 **0.1%**(4/3206)跳到 rollout_1 的 **28.8%**(844/2935)。⚠️ 此数字与两种解释都兼容:①约束机制坏了;②模型输出分布本身崩了。**不能只凭它定罪。**
7. **alchemy 对照**(用户经验 + 脚本核实):同一个 VL ckpt、同 `--colocate`(=每步 sleep/wake)、同 SIF,多步训练无此崩塌。alchemy 与我们的 ACT 生成的结构性差异:**alchemy 不用 json_schema 约束解码**(裸 sampling_params),我们每个 ACT 请求带 xgrammar json_schema。
8. **rollout 变慢(27→63min)是崩塌的果不是因**(profiling):崩塌模型烧满 40 轮 + 每轮 token ×1.77 → decode 量 ×2.6。独立存在的第二问题:16 路并发争抢占 episode 时间 40-50%(尾部无争抢 100s vs 头部 300s+)。

## 三、已排除的假设(排除证据)

| 假设 | 排除证据 |
|---|---|
| 优化器更新过猛 / RL 数学问题 | lr=0 照崩(金丝雀#1) |
| fallback 字面量投毒是主因 | clean_act 修复后照崩(run3);它是真实缺陷但非主犯 |
| loss mask 切错位置 | 实证:loss=1 的 token 解码后精确等于 assistant 文本 |
| 视觉塔垃圾导致文本生成崩 | alchemy 同 ckpt 同 sleep/wake 无恙(用户反例) |
| 轨迹变长/straggler/评测拖慢导致变慢 | profiling 拆帐:变慢主体=崩塌行为本身 |

## 四、已修复并验证的 bug(6 项)

1. surrogate 序列化崩溃两层清洗(修后两 run 越过原崩点 0 复发)
2. observation 无上限 RAM 炸弹(cap 8000 保头尾;修后内存 475G 平稳,曾 1.29TB)
3. ACT 训练文本被规范化 JSON/fallback 替换(clean_act;rollout_0 无退化)
4. 判分 FEEDBACK 不进 WRITE 输入(已按用户口径修:只进 WRITE)
5. wandb success_frac 每题丢 7/8 样本(去重键加副本编号;用户批准)
6. clbench 步数预算 off-by-one(第 40 步不可用;用户裁定原版 bug,修复正确)

## 五、未决问题(不猜,只列)

1. **崩塌机制未定**。已知边界:发生在第一次 sleep/wake 周期后;与权重数值无关;文本权重传输无误;alchemy(无约束解码)免疫。候选区分实验(未跑,待批):lr=0 + 去掉 json_schema 的对照。
2. 训练/推理模板失配(训练序列每轮 assistant 带 `<think>` 脚手架、推理历史没有):存在,与崩塌关系未知。
3. json_schema 约束在 rollout_0 也有 844→(其实 0.1%)……rollout_0 基本正常;约束在新引擎上工作正常。
4. 9B:Blackwell 可跑、754G 勉强够(峰值 751),现无 run。

## 六、待用户裁决

- M1-M8(中风险参数/行为清单,见 SEVEN_POINTS_AUDIT_0713.md 附录 A)
- 我们离线 icl_summary 的"下一题开头 carryover"删不删
- write_transcript_chars 改法(保题面+提额度方案已提)
- 是否跑"lr=0 + 去约束"对照实验定位崩塌
- 七点对账的 3-7 用户尚未审完

## 七(补 2026-07-14):崩塌根因定案
lr=0 双 scaffold 金丝雀 + 三方网络调研交叉 → 根因 = sglang PR #27140(sleep/wake 后 CUDA graph
回放陈旧 mamba 状态指针,未合入)+ SIF 缺 #24954。完整调研与行动方案见
[[SLEEP_WAKE_COLLAPSE_ROOTCAUSE]]。新增实验:textfmt 三套件评测(baseline 0.1245 vs 旧 0.1231;
icl 0.0432;replace 0.0697,gain +0.48)、lr=0 textfmt 金丝雀(rollout_1 ACT=0.0000)、
seq32768 两次在崩塌样本上 step1 GPU OOM(结论:下次用 24576)。

## 七、当前机器状态(2026-07-13 午后)

- babel-v5-16(1600G 分配,job 9244653):**无任何训练在跑**,8 卡空闲
- 9B:无 run;ICL augment/replace:已完成(0.0347 / 0.0706)
- 所有崩溃现场归档:logs/collapsed-run-archive-0713/ + 各 .crashlog/.oldlog
