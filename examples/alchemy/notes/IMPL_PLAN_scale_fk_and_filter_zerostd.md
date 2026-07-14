# 实现计划:scale F_k 覆盖 + 过滤 zero-std group

两个改进的动机见 `ANALYSIS_offline_actonly_vs_cotrain.md` §10/§11。本文件是基于代码调研的实现方案。

## 调研结论(关键代码点)
| 环节 | 位置 | 现状 |
|---|---|---|
| transition 收集 | `alchemy_rollout.py:272` `_collect_transition` | 每步 valid potion 后收进 `all_transitions`(本 rollout 局部) |
| **F_k 构建** | `alchemy_rollout.py:285` `_build_fk(all_transitions,k,cap)` | **只用这一个 rollout 自己的** future transitions(trial>k, dedup, cap=12, min_fk=3) |
| WRITE reward | `alchemy_rollout.py:327` `_score_memory_accuracy` | G'=4 候选各对 F_k 逐条 greedy 生成+exact match,平均 acc∈[0,1] |
| advantage | `alchemy_advantage.py:76` `reward_post_process` | WRITE 按 (episode_id,rewrite_idx) 分组白化;zero-std→adv=0 |
| **sibling 数据** | `generate(input)` 只收单个 Sample | ❌ **跨 rollout 不可达**(要 union 得改框架 ~300 行 + 同步 barrier) |
| **oracle 转移** | `eval/wm_probe.py:81` `episode_tables(ep)` | ✅ 离线全表(~96 条 (coord,potion)→result),只需 episode_index(metadata 已有) |
| remove_sample | `utils/types.py:34` + `train_data_conversion.py:53` | 置 True → loss_mask 全 0(零梯度)**但仍占 batch** |
| **batch packing** | `utils/data.py:261` 基于 `response_length` | ⚠️ **advantage=0 / remove_sample 都不释放 token 名额** |
| **rollout sample filter** | `inference_rollout_train.py:157` `--rollout-sample-filter-path` | 在 reward_post_process **之前**、可**物理删除** sample → 真正释放名额 |

---

## 改进 1:Scale F_k 覆盖 → 走 oracle-sampled(推荐,~80 行,不动框架)

**为什么不走 sibling-union**:generate() 只拿到单个 Sample,sibling 的 `all_transitions` 完全不可达,要做 union 得改 `GenerateFnInput` + rollout 框架核心 + 加组同步 barrier(~300 行,打破 fire-and-forget)。成本太高,先不做。

**oracle-sampled 方案**(每个 rollout 独立,F_k 来自该 episode 的离线全表):
1. `alchemy_rollout.py` 顶部 `from examples.alchemy.eval.wm_probe import episode_tables`(注意:确认它在 ray worker 里 import 不拖重依赖;wm_probe 若带 llm_client/requests,需像 `_parse_stone` 那样把 `episode_tables` 的纯逻辑拷过来,避免污染 worker)。
2. 新增 `_build_fk_oracle(episode_index, k, cap, changed_only=True)`:从 `episode_tables(ep)["transitions"]` 取转移,可选只留 `changed`(非 no-op),截到 cap。注意 oracle 表是**完整可能转移**,与"trial>k"无关——这正是要的(让 memory 学完整世界模型,而非只复述本轮看到的)。
3. `_generate_trial` 里加开关:`ALCHEMY_FK_SOURCE=oracle|observed`(默认 observed 保持现状,不侵入)。oracle 时 `boundary_fk = {k: _build_fk_oracle(ep,k,cap) ...}`。
4. `write_audit` 记 `fk_source`,便于离线审计。

**风险/验证**:oracle F_k 更大更全(~96 vs 3-12)→ 候选 memory 更难全对 → 可能整体 acc 下降,但**候选区分度上升、zero-std 下降**(正是目的)。需 A/B:`observed` vs `oracle(changed_only)` vs `oracle(全)`,看 `write_group_zero_std_frac` 和最终离线分。
**折中选项**:若全表太难致 reward 全低,可对 oracle 转移**采样**到和原 cap 接近的规模、但覆盖比"单 rollout 实际走到的"广。

---

## 改进 2:过滤 zero-std group → 必须走 `--rollout-sample-filter-path`(不是 advantage hook)

**关键发现(否决直觉做法)**:在 `reward_post_process` 里把 zero-std 的 advantage 设 0、或标 `remove_sample=True`,**都不会释放 batch 名额**——batch packing 按 `response_length` 算(`data.py:261`),loss_mask/advantage 不影响打包。所以那只是"零梯度占位",省不下算力。

**正确实现**:用 `--rollout-sample-filter-path`(`inference_rollout_train.py:157`),它在 reward_post_process **之前**跑、能从 data 里**物理删 sample**。
- 时机 OK:WRITE 的 candidate reward 在 rollout 阶段就写进了 `sample.reward`(`alchemy_rollout.py:712`),filter 时已可读。
- 写 `filter_zero_std_groups(args, data)`:按 `(episode_id, rewrite_idx)` 给 **WRITE** sample 分组,组内 reward 全等(std≈0)则把该组样本从 data 删除。
- **必须只删 WRITE 的 zero-std 组**,别碰 ACT(ACT 是 per-token advantage、且 zero-std 比例低);保留 group 完整性(要么整组删要么不删,GRPO 才正确)。
- 删完后 reward_post_process 对剩余样本正常白化。

**收益**:真正释放 token 名额 → 同 `max-tokens-per-gpu` 下能塞进更多有梯度的 WRITE/ACT 样本。**注意**:这是效率优化,不增加学习信号总量;减少 zero-std 产生靠改进 1。

**caveat**:确认 `--rollout-sample-filter-path` 的 `data` 结构(flat list 还是 list[list],决定分组遍历写法);确认删样本后 ACT/WRITE 配比不被破坏到某个 episode 一条 WRITE 都不剩(可加保护:每 episode 至少留 N 个有梯度组)。

---

## 优先级与验证
1. **先做改进 1(oracle F_k)**——治本:同时减少 zero-std 产生 + 修 §10 封顶。指标:`alchemy_grpo/write_group_zero_std_frac` 应明显下降。
2. **再做改进 2(filter)**——治标:把改进 1 之后仍残留的 zero-std 组的 batch 名额省出来。
3. 端到端验证:co-train 重训,看 ① zero_std_frac↓ ② WRITE 有效梯度↑ ③ 双训离线分(co-train 全程)能否从 0.373 往 act-only 0.46 逼近。
