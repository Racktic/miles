# 训练崩塌根因:ACT 训练文本被 canonical JSON / fallback 字面量替换(off-policy 投毒)

## 现象(三个 run 一致,证据已归档 logs/collapsed-run-archive-0713/)
- rollout_0:ACT mean_r≈0.21,completed 73/160(4B formal run)
- **rollout_1 起:ACT mean_r=0.0000,completed 0/160,advantages=0.00e+00**——一次梯度更新后策略崩塌,之后 ACT 流零学习信号,94% issue 烧满 40 步、57% turn 空 command、每轮 token 膨胀 1.77×(这也是 rollout 27min→63min 变慢的主因)。
- fix2(5 步 run)同型:raw_reward 0.151 → 0.011-0.041。surrogate 垃圾字符也是崩塌症状。

## 根因(codebase_rollout.py,修复前)
1. `assistant_context = action.model_dump_json()`(原 625 行)——进对话历史、也进训练文本的是**规范化重排的 JSON,不是模型采样的原文**。
2. `_response_from_text` parse 失败时返回 fallback:`{"thought": "No parseable thought was produced.", "command": ""}`——这个**模型从没说过的字面量**被 `_pack_act_sample` 整段重 tokenize(`rollout_log_probs=None`)当作策略输出训练。step0 有 79 个这种 turn 混在非零 advantage 组里 → **直接教模型输出空命令** → 空命令引发更多 parse 失败 → 死循环崩塌。
3. 对照:`_pack_write_sample` 用真实 `response_ids + logprobs`(正确 on-policy)——所以 WRITE 流一直有信号、没崩,与观测完全吻合。
4. 帮凶:sglang json_schema(xgrammar)约束疑似未生效(输出停在未闭合 JSON 中间而 finish=stop),使 parse 失败高频。**离线 clbench 无此问题**:它走 provider 强制 structured output(icl_summary/system.py:201),parse 不会失败,且离线不训练。

## 修复(codebase_rollout.py:631)
`assistant_context = clean_act`(模型真实采样文本,已过 _strip_special + _strip_lone_surrogates)。
- 训练文本 = 采样文本,fallback 字面量永不进 loss;
- 结构化 action 仍在 transcript `"action"` 字段(_jsonable)供审计;
- 上下文格式与离线的差异(原文 vs canonical JSON)是可接受偏离——离线的 canonical 前提是 provider 强制 schema,我们不具备。

## 验证判据(重启后)
- **rollout_1 的 ACT mean_r > 0 且 completed > 0** = 治好;
- 若 rollout_1 仍全 0,下一嫌疑:①xgrammar 约束失效(需在闲置卡上实测 /generate + json_schema)②`_pack_act_sample` 重 tokenize 与采样 token 的残余失配(终极解:用 response_ids 增量拼接 + 传 rollout_log_probs,参照 _pack_write_sample)。

## 追记 2026-07-13:第二颗雷——observation 无上限的 RAM 炸弹
on-policy 修复后的 run,rollout_0 健康(ACT 0.2258,completed 79/160,均优于修复前),但 rollout_1 期间
**RolloutManager 膨胀到 1.29TB anon 被 cgroup OOM-kill**(dmesg pid 1774468;1600G 上限都不够)。
- 根因:clbench task.py 只 cap trace(`[:2000]`),**渲染给模型的 obs_content 无上限**;策略采出高输出命令
  (grep -r/cat 大文件)后,巨量 obs 在 messages/trial_messages/transcript 三处累积 + 每轮全量重 tokenize,
  16 episode 同住一个 RolloutManager 进程。**当年 512G 时代 RolloutMan 被杀时 148G 是同一颗雷的小号版本。**
- 修复:`_cap_observation`(默认 8000 字符,保头保尾,`CODEBASE_OBS_MAX_CHARS` 可调),用于 FEEDBACK 消息
  构造与 transcript observation 字段。正常几 KB 输出不受影响,只斩炸弹;离线 clbench 基准未动(离线靠
  240k token 预算天然扛住,不可比性风险小)。

## 相关
- [[WRITE_REWARD_GAIN_VS_RAW]](WRITE reward 用 gain 与 raw 等价的证明)
- profiling 报告结论:rollout 变慢主因即本崩塌(decode 量 2.6×);并发争抢 ~40-50% 是次要独立问题。
