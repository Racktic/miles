# 七点对账清单(2026-07-13,与用户逐条死磕核实)

状态标记:✅=已核实成立 ❌=被驳回/撤回 ⚠️=部分成立 🔍=核实中
用户逐条拷打,本文档随核实进度更新。**每条都附证据出处,不许口说。**

---

## 1. "并发" 的含义 / 16 题 = 16 容器?——✅ 已向用户讲清,用户确认理解

- 结构:rollout_batch_size=2 prompt × n_samples=8 = **16 条 episode 并行**;每条 episode 内 10 个 issue **串行**(memory 依赖)。
- 任一时刻每 episode 1 个活跃 issue = 1 个容器 → **同时 16 个容器**;每条 episode 一生串行开 10 个(评测另起临时容器)。
- "并发争抢":16 条 episode 共享 8 个 sglang 引擎 + **唯一的 RolloutManager 进程**(全部 tokenize/task.step 调度)+ 节点 CPU。
- 证据:同一 40 轮 issue,尾部单跑 ~100s,16 路并发时 289-362s(profiling 报告 §4)。

## 2. "判分异步化" ——❌ 提议撤回;并确认这是 miles 的**严重对齐 bug**(用户发现)

**用户的反驳(正确)**:官方 trace `results/codebase_adaptation/traces/2026-07-04T23-16-53.705279Z/run_0000.json` turn 17,
submit 后 observation = `"Submission FAILED (eval status: tests_failed).\n\nMoving to issue 2/19..."` ——
判分结果在写 memory 之前必须可见,issue 必须完全弄完。

**核实(离线源码铁证)**,`clbench src/systems/icl_summary/system.py`:
- `:246-251`:instance_complete 时判分 FEEDBACK **append 进 instance_messages**,代码注释原文:
  *"The final outcome must be visible to the summary rewrite, and survives into the next instance's first prompt."*
- `:255` + `:180-184`(replace 模式):判分还存入 `_pending_feedback`,**下一 issue 首 prompt 前置**
  `FEEDBACK FROM PREVIOUS INSTANCE:\n{判分}`(在 [Summary...] 之前)。

**miles 现状(codebase_rollout.py 主循环 ~:655-665)**:submit 后 `break`,判分 obs **不进 trial_messages**:
1. ❌ WRITE(memory 重写)看不到"这次成没成" —— 与离线注释明文要求直接冲突;
2. ❌ 下一 issue 首 ACT prompt 缺 `FEEDBACK FROM PREVIOUS INSTANCE` 段 —— 离线有、我们没有。

**后果**:memory 写手不知道自己总结的方法是否有效(可能正是 4B ICL 负 gain / WRITE 信号弱的一部分原因)。

**用户两条重要纠正(2026-07-13)**:
1. **`icl_summary` 不是 clbench 官方代码,是我们自己写的**(git 铁证:`?? src/systems/icl_summary/` 未追踪、
   零提交;官方只有 ace/claude/codex/human/icl/icl_notepad/mem0)。我此前一直把我们自己的离线实现
   当"官方对齐标准"引用,认知错误。对齐权威链:用户方法设计 > clbench 官方 > 我们的离线实现。
2. **"下一题开头带上一题判分"在 replace 模式下没有意义**:下一题看不到上一题任何内容,单独一句
   "上一把失败了"没有信息量——这是我们离线 icl_summary 自己的设计瑕疵(用户:离线代码的问题回头另说)。

**已实施(用户口径:只加 WRITE 输入)**:`codebase_rollout.py:780-786`——issue 结束时把
`FEEDBACK: {判分文本}`(过 _cap_observation)追加进 trial_messages,再 build_write_messages。
顺序保证:_pack_act_sample 在前(ACT 训练样本不含判分),WRITE 输入在后(含判分)。py_compile 通过。
判分异步化提议作废(判分必须先于 WRITE,在串行链上)。
⚠️ 遗留:我们的离线 icl_summary 还带着"下一题开头 carryover",训练侧现在没有——两侧此处不一致,
待用户决定是否顺手把离线的 carryover 也去掉。

## 3. TP=2 vs TP=1 ——✅ 两者都对,是两个不同的东西

- **训练**(Megatron):`--tensor-model-parallel-size 2`(TP=2,用户记忆正确)。
- **推理**(sglang rollout 引擎):`--rollout-num-gpus-per-engine 1` → **8 引擎 × 1 卡 × TP=1**
  (证据:console log SGLangEngine ServerArgs dump `tp_size=1`;run 脚本该行)。
- 优化建议指**推理侧**改 4 引擎 × TP=2:实测 70% 时刻每引擎仅 1 个请求在 decode(episode 内串行对话喂不满),
  单流 decode ~120 tok/s;推理 TP=2 单流 1.5-1.8×。一行参数。**未实施,待拷打。**

## 4. 优化清单第三条(三个小项)——已拆开解释,均未实施

- **sandbox 预建**:每 issue 起点 `apptainer build --sandbox` ~10s 串行卡链;issue 顺序预知,可后台预建 k+1。
- **MAX_RESP 1200→800**:健康模型均值 371 tok/轮,撞 1200 上限的多为崩塌垃圾。⚠️ 语义变化(动作空间),需标注。
- **连续空命令提前判负**:空命令不耗 task 步数(task.py 退回)但耗 rollout 生成预算,崩塌模型可空转烧满 40 次生成;
  连续 N 轮空命令即强制结束 issue。⚠️ 语义变化。

## 5. json_schema 没生效 vs 截断假说——⚠️ 用户对了一半,两件事都真

崩塌 run rollout_1 实测(3593 turn,archive traj):
| finish | JSON 闭合 | JSON 未闭合 |
|---|---|---|
| stop | 2091 | **844** ← 截断解释不了 |
| length | 0 | **658** ← 用户假说 ✅(被 1200 上限截断) |

- 658 个未闭合确实是截断(用户对);
- 但 844 个 **finish=stop 且停在 JSON 中间**(样例尾:`...The command is `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.`纯散文)
  ——xgrammar 生效时 JSON 未完成前 EOS 被 mask,模型不可能自愿停 → **约束确实没生效**。
- 待办:闲卡实测 sglang 0.5.13.dev28 + Qwen3.5(hybrid/mamba)的 json_schema 路径。

## 6. 我的三处代码改动(全在 codebase_rollout.py)——逐条状态

| # | 改动 | 位置 | 状态 |
|---|---|---|---|
| 1 | surrogate 清洗两层:`_strip_lone_surrogates`(_infer 出口)+ payload 级(_response_from_text 内) | :146-153, :132, :315 | ✅ 修好;两 run 越过原崩点(step3)0 复发 |
| 2 | `assistant_context = clean_act`(原 model_dump_json;fallback 字面量不再被当模型输出训练) | :631 | ✅ 无副作用(rollout_0 73→76/160);❌ 未治好崩塌(帮凶非主犯) |
| 3 | `_cap_observation` 8000 字符保头尾(clbench 给模型的 obs 无上限,RolloutMan 曾 1.29TB 被 OOM 杀) | :162-183, :616, :677 | ✅ 修好(dmesg 铁证;单测 400k→8085 尾部 ERROR 保留) |

⚠️ 待用户拷打点:#2 与离线的偏离(离线 assistant 记录是 canonical JSON,但离线 provider 强制 schema 不会 parse 失败;
我们在约束失效环境下用原文是否合理)、#3 的 8000 上限取值与离线(无 cap+240k 预算)的偏离是否可接受。

## 7. "是不是把代码搞废了"——核实结论:崩塌先于我的全部改动

1. **时间线**:7/11 的 fix2 run(上述 3 处改动都还不存在)raw_reward 已是 0.151 → 0.011-0.041,同款一步崩塌。
2. 用户看到的"实验 bug"消息 = 金丝雀启动时 `ray stop` 清理旧进程的正常日志(`Killed <ray组件>`),监控关键词误捕。
3. 改动范围:仅 codebase_rollout.py 三处(clbench 仓库/miles 框架/数据/reward 逻辑零改动),各带注释,可 5 分钟回滚。

---

# 附录 A:全量扫雷清单(2026-07-13 审计 agent + 主 agent 亲验)

## 高危(H1/H2 已主 agent 逐行亲验)
| # | 内容 | 事实 | 用户裁决 |
|---|---|---|---|
| H1 | clbench task.py 步数预算语义被改(前置检查 `>=`→`>` + 新增执行后检查) | **原版有 off-by-one bug**:`current_steps += 1` 在检查前,第 40 条命令时 40>=40 直接 timeout **不执行**——预算写 40 实际只能用 39 步,而 reward=1−regret/40 按 40 归一。改后恰好 40 步可用、第 40 步 submit 可判分。我们所有 baseline/训练/eval 均用改后代码,内部自洽 | ✅ **用户规则"原版有bug我们修好了就没问题"→ 通过** |
| H2 | wandb `success_frac` 失真:去重键 (episode_id, trial_pos),同组 8 个 rollout episode_id 相同→每题只数第 1 个 rollout,丢 7 个。只影响曲线读数,不影响训练 | 已亲验 codebase_metrics.py:115-125 | 🔍 已用人话重新解释,待批修复(一行:键加 sample index) |
| H3 | singularity 容器默认可联网(docker 官方 --network=none;断网开关存在未开) | — | ✅ 用户早知道,先不管 |
| H4 | 训练中每 8 步的 heldout eval 用 FIFO 截断,预算 240000 为自造值(官方 icl 是模型上限−1024;行为本身对齐官方:删最旧整条+reserve 500) | 仅影响 in-training eval,不影响训练本身 | ✅ 用户:问题不大 |
| H6 | 上一题判分不进下一题开头(只进 WRITE 输入) | — | ✅ 用户拍板的口径,已实施 |

## 中风险(待用户逐条回答)
| # | 内容 | 默认值 vs 出身 |
|---|---|---|
| M1 | django/sympy 官方判分 timeout 600s(上游 generic 默认 180s) | 自定 |
| M2 | django/sympy 题面注入一行测试命令提示(_REPO_TEST_HINTS);官方无;baseline 同带则内部一致 | 自定 |
| M3 | 容器环境注入(PATH 前置/LANG/无 fakeroot/matplotlib PYTEST_ADDOPTS),注释论证"不给新能力" | docker→apptainer 适配衍生 |
| M4 | run 脚本 shell 默认值≠批准值(SEQ_LEN:-32768 批准20480、EVAL_INTERVAL:-3 批准8、SAVE_INTERVAL:-0 不存ckpt、NUM_ROLLOUT:-1),全靠 env 补,忘设即静默变配置 | 建议把批准值写死 |
| M5 | WRITE 生成上限代码兜底 768(yaml 才是批准的 2048),yaml 丢失静默降 | 自定兜底 |
| M6 | MAX_RESP=1200(ACT 每轮生成上限);alchemy 是 2560,非对齐值 | 自定 |
| M7 | 官方判分基础设施故障时 except→记 0 分不中断,无监控指标 | 自定 |
| M8 | WRITE 输出被 length 截断时,半截 memory 会替换旧 memory 继续用;离线失败保留旧 memory | 自定 |

## 已核无问题(抽样,详见审计原文)
- memory 训练路径无任何 FIFO/截断;ACT 尾裁=纯尾裁+超长整条丢弃(符合"宁 drop 不肢解")✓
- `_force_complete_current_issue` 不判分给 0,与 task 超时语义一致 ✓
- obs cap 覆盖全部模型可见入口且仅一处;空 think mask 与 alchemy 逐字同 ✓
- write 采样=ACT 同参仅改 max_new_tokens;eval 采样自动回落 rollout 值 ✓
- ACT 样本 rollout_log_probs=None(packed 多轮无法对齐 logprob)——设计使然,已报备
- 未覆盖:scripts/swecl/gen_train_episodes.py(episode 池生成),下轮补审

## 附:崩塌根因排查现状(与第 7 点相关的开放项)

- 核心矛盾:update0 grad_norm=1.55 × lr=1e-6(微小)vs step1 KL≈0.27(巨幅漂移),数学不相容。
- 二选一:**A. 优化器 bug**(precision-aware+cpu-offload)vs **B. colocate sleep/wake 损坏 sglang 权重**
  (rollout_1 恰是引擎第一次睡醒)。
- **lr=0 金丝雀跑步中**(权重不可能动):rollout_1 健康→A;依然崩→B。
- 另发现待修:训练序列每轮 assistant 带 `<think>\n\n</think>` 脚手架而推理历史没有(mask 已清 loss 但上下文错位);
  已实证 loss mask 本身对齐(loss=1 token 精确=assistant 文本)。
