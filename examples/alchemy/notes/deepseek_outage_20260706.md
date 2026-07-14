# DeepSeek 欠费事件记录(2026-07-06)

## 发生了什么
DeepSeek 账户余额耗尽(API 返回 402 Insufficient Balance),explore judge(β0.3 的
memory-delta 打分)按设计优雅降级:judge 返回 None → 该样本 explore 优势加 0 → 训练不崩、
无报错,但受影响步实际 β=0(task reward 训练照常)。当日上午充值后 judge 自动恢复,无需重启。

## 受影响范围(用 traj/train/rollout_<step>/*.json 的 act_explore 出分数逐步核实)

| 实验 | 受影响步 | 占 120 步 | 备注 |
|---|---|---|---|
| E1 k0down (`...k0down-r120-e10-20260705`) | **53, 54, 55**(共 3 步) | 2.5% | step 56 起恢复(充值当步即出分) |
| E2 k0skip (`...k0skip-r120-e10-20260705`) | **59-63**(共 5 步) | ~4% | 见下:实际残留污染只有 step 59 |

## E2 后续:2026-07-06 上午在 step ~63 处主动 scancel(jobid 8970067)
原因:eval 分数一般 + 给他人腾 rl 节点(n9-20)。ckpt/iter_0000059 已完整保存
(latest=59, 53G)。**若日后 resume,会从 iter_59 重跑 60+(judge 已恢复),残留污染只剩
step 59 本身这一步**(iter_59 包含了 β=0 的 step 59 训练)。

## 分析终点结果时注明
- E1:step 53-55 为 β=0 窗口(训练中段,3/120 步,预期淹没在噪声里,参照 S0 vs S03 全程
  β 差异终点也只差 ~0.06)。
- E2:如 resume,只有 step 59 一步 β=0。
