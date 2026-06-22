# Alchemy / Qwen3.5-4B 训练调试日志(重写版)

> 目标:训 Symbolic Alchemy 两条流 GRPO(ACT+WRITE)on **Qwen3.5-4B**(GDN 混合模型)+ miles/Megatron + sglang colocate。
> 环境:Apptainer SIF,8× A100 80GB(sm_80),Triton 3.6.0,torch 2.11.0,megatron-core 0.16.0rc0。
> 维护:Claude。最近重写:2026-06-18(按"已验证/猜测/计划"三层重整,删去早期错误结论)。
>
> **本文档纪律:**
> 1. §1 只放实验/代码证据确凿的;§2 是未验证猜测;§3 是计划;§4 是已证伪、别再走的弯路。
> 2. **控变量铁律:任何"A 比 B 如何"的结论,A、B 必须只差一个变量。差≥2个变量 → 禁止下结论,只能记录现象。**
> 3. 我们的并行约束:`TP×CP×DP=8`(固定 CP 时 TP 与 DP 绑死),**所以无法单独归因 TP 或 DP,只能归因到"并行配置"这个整体**。

---

## §0 一个训练步的阶段(对照日志 Timer)
colocate:训练(megatron)与推理(sglang)共用 8 卡,每步换手。
`update_weights → rollout(train_wait)→ reward/advantage hook → wake_up → data_preprocess → ref_log_probs → log_probs → train(fwd+bwd+optim)`。
- ref_log_probs / log_probs = 纯前向(`@torch.no_grad()`),**无跨 DP 梯度 collective**。
- train = 反向 + 跨 DP 梯度 reduce_scatter/allreduce(+recompute 重跑 GDN)。最重。

---

## §1 已实验/代码验证的结论(确凿)

**V1. stack 本身没问题。** 官方 math example(`run-qwen3.5-4B.sh`,TP2/CP1/DP4,dapo-math,短单轮 ≤8192)在我们环境跑得干净:ref 54s / logp 23s / train 298s,无 OOM/hang。→ Qwen3.5-4B GDN + miles + A100 对**短单轮**完全 OK。(run `official-smoke-20260617-204024`)

**V2. 我们之前的"hang"是慢,不是死锁。** TP8/DP1 单轮(steps1)**跑完了一个完整 step**:ref 614s + logp 604s + train 3233s ≈ 74min,但完成。→ 之前判"hang/死锁"是因为我没等够就 kill。(run `20260617-220350`)

**V3. 慢是配置造成的(~37×)。** 单变量:只把 TP8/DP1 → TP2/DP4(同 steps1 数据),ref 614→16s、train 3233→94s、一步 74min→~2min。连跑 3 步稳。(run `20260617-231242`)

**V4. GDN 在 TP 下复制、不切分(代码事实)。** `qwen3_5.py` 的 GDN 用普通 `nn.Linear`(`grep ColumnParallel=0`);config `num_layers=32, full_attention_interval=4` → **24/32 层是 GDN**。→ 高 TP = 每 rank 冗余算整个 GDN = V3 慢的机制。

**V5.(单变量=并行配置)同样 64ep/steps5,(TP2,DP4) 前向比 (TP4,DP2) 快 ~4×。** ref_log_probs:exp2(TP2/DP4)66s vs exp3(TP4/DP2)260s——这俩只差并行配置,干净。**不能再细分是 DP 还是 TP 的功劳(它俩绑着变);也不要拿不同 ep/数据的 run 凑"DP 序列"(我之前犯过)。**

**V6. reload-timeout 是真 bug(代码+实测)。** `reload_process_groups` 重建 NCCL 组只传 ranks、不传 timeout;torch `new_group(timeout=None)` 回落 `default_pg_nccl_timeout`=10min(实测常量 0:10:00)。实测:看门狗恒在 `Timeout(ms)=600000`,无视 `--distributed-timeout-minutes 120`。→ colocate 每次重建后 NCCL 组超时静默回到 10min。(slime 上游同此 bug,见 V11)

**V7. 减 trial 数 ≈ 线性降 rollout。** exp1:num_trials 10→6,rollout 568/808→385/537s(~0.67×),ACT n 160→96(cap 精确)。(run `20260618-004108`)

**V8. 提并发 = 降 rollout 的主杠杆。** exp2:16ep→64ep(4×),rollout 568→683s(只 1.2×),GPU util <30%→~90%,每 episode 效率 ~3.3×。(run `20260618-011411`)

**V9. TP2/DP4 能跑多轮训练,但显存紧。** steps5 多轮 64ep 跑完 2 步,训练侧快(ref45/train187-237s),但 softOOM 159、train step 摸 80GB。(run `20260617-235534` / exp2)

**V10. TP4/DP2:显存好但慢、且 train step 撞 10min 看门狗。** exp3:softOOM=0(TP=4 把显存治好✅),但 ref 260s(慢)、train step 崩于 NCCL `_REDUCE_SCATTER_BASE` 10min 超时。(run `20260618-092357`)

**V11. slime 不值得换。** 代码确认:slime 的 GDN 同样不切 TP(逐行同 miles)、同一个 reload-timeout bug;且 miles 反而领先(有已合并的 GDN context-parallel 路径 `build_gdn_cp_context`,slime 那个 CP PR #1816 关掉没合)。miles 是 slime 的 fork。

**V12. 运维事实:debug run 即便 `--save-interval 999999` 也会强制存一次 ~66-67GB ckpt。** 必须跑完删 `alchemy_runs/<RUNID>`(今天累计清了 ~200GB)。

**V13.(单变量=reload-fix)reload-timeout 修复有效。** 092357(无fix)vs 102136(有fix),其余全同(TP4/DP2,64ep,steps5):092357 train step 10min 整崩(NCCL reduce_scatter 看门狗);102136 train step 跑 ~35min **无任何看门狗**(在 60min cap 主动杀,非崩)。→ 修复让 `--distributed-timeout-minutes` 真正传进重建的 NCCL 组(确认 V6 的 bug + 修复)。

**V14.(单 run 观察)TP4/DP2 在 64ep/steps5 下 train step 后段显存 thrash。** 102136:越过 10min 后 softOOM 0→97、GPU 死贴 80GB,35min 未完成(被 cap 杀,非硬 OOM,完成与否未知)。→ TP4/DP2 把"看门狗崩"换成"显存 thrash+极慢",对此 workload 不可行。

---

## §2 未验证的猜测(待证,勿当结论)

**H1. → 已验证,升为 V13。**(reload-timeout 修复有效;附带观察 V14:TP4/DP2 后段显存 thrash。)

**H2. exp3 的 train-step 10min 崩,机制是 (a) 还是 (b) 未分清**:(a) 两 DP 组到达 collective 差 >10min;(b) 该 collective 卡在 >10min 的慢 GDN 计算后面、不需 DP 不均。balance-data 已均衡 token → 倾向 (b),但没用 flight recorder 证。**对修复无影响(都靠抬超时解决)。**

**H3.(本测不支持,甚至被否)** "TP2×CP2×DP2 是长样本正解"——222(113259)实测 **train step 硬 OOM**。CP=2 本应切序列省显存,却比 exp2(TP2/CP1/DP4,完成)更差。猜测:miles 的 GDN-CP 路径自身有显存开销,抵消了切序列收益。**(单 run、vs exp2 差 CP+DP 两变量,未隔离;但 P3 的"CP 救显存"假设这一测没成立。)**
- 现状:64ep/steps5 下,**只有 exp2(TP2/CP1/DP4)完成过 train step(紧,softOOM159)**;TP4/DP2 thrash 未完;TP2/CP2/DP2 硬 OOM。

**H4. 换 Qwen3(非 GDN)可能消除大部分性能痛。** 依据:miles 所有长多轮 agentic 例子都是非 GDN 标准注意力、跑得好。**没在我们环境实测。** 是否换取决于 Qwen3.5/GDN 对课题是否必需(待查设计文档)。

---

## §3 待验证 / 计划

**P1. ✅ 完成**(→ V13:fix 有效)。

**P2.** 整理 reload-timeout 修复为 **PR 给 radixark/miles**(注明 slime 上游同存);参考/cross-ref miles #705(同函数加 barrier)。**(代码已改好,可随时提。)**

**P3.(当前主线)** 测 **TP2 × CP2 × DP2**(GDN 正解扩展)——V14(TP4/DP2 不可行)后唯一没试的可行方向。
- ⚠️ 控变量提醒:8 卡固定 TP=2 时,TP×CP×DP=8 → 改 CP(1→2)必然带着 DP(4→2)一起变,**无法单独隔离 CP**。所以这是"**候选配置可行性测试**"(能否不 OOM/不超时、多快),**不能据此给 CP 单独归因**。对照 exp2(TP2/CP1/DP4)是 2 变量差(CP+DP)。
- 若要真隔离 CP:得变总卡数让 TP/DP 不动(如 4 卡 TP2/CP1/DP2 vs 8 卡 TP2/CP2/DP2),另算。

**P4.(潜在雷)** balance_data 要求有效样本数被 dp_size 整除;我们 `remove_sample` 丢无效 WRITE 样本,迟早不整除 → 崩(Karmarkar-Karp 断言)。上游修复 = miles #1356。需要时带上。

**P5.(可选)** flight recorder(`TORCH_NCCL_TRACE_BUFFER_SIZE`)区分 H2 的 (a)/(b)——锦上添花。

---

## §4 已证伪的错误结论(别再走)
- ❌ "GDN fla Triton kernel 死锁 / fla-org #947 就是我们的 bug / CP 是避免死锁所必需" —— **错**。真相:配置导致的**慢**(V2/V3)+ reload 把 NCCL 超时回落到 10min(V6)。fla #947 是 H200 上"训练 8000-12000 步后"的不同症状,与我们 A100 第一步就慢对不上。
- ❌ "减 max_steps / 过滤超长样本 是避免 hang 的必需手段" —— **moot**(没有死锁)。减 trial(V7)只是降 rollout 的可选杠杆,不是救命。
- ❌ 早期把 CP 当成"错诊的弯路" —— 不准确。CP 不是用来避死锁的,但它**是 GDN 长序列/显存的正解扩展轴**(H3 待验)。

---

## §5 运行记录(全变量)
**所有 alchemy run 共同常量**(故不单列):`max-tokens-per-gpu=20480`、`log-probs-chunk-size=512`、`--sequence-parallel` on、`--optimizer-cpu-offload` on、`--balance-data` on、`--distributed-timeout-minutes=120`(但 pre-fix 对重建组无效,见 V6)。数据:steps1=每trial≤1轮 / steps5=≤5轮(均 num_trials 原生10,除非 cap)。

| RunID | TP/CP/DP | ep(batch×n) | max_steps/trial | num_trials | #roll | reload-fix | 关键结果 |
|---|---|---|---|---|---|---|---|
| 220350 | **8**/1/**1** | 16 (4×4) | 1 | 10 | 50 | 否 | ref614/logp604/train3233s,一步74min**完成** |
| 231242 | **2**/1/**4** | 16 (4×4) | 1 | 10 | 50 | 否 | ref16/logp7/train94s,~2min,3步稳 |
| 235534 | 2/1/4 | 16 (4×4) | 5 | 10 | 2 | 否 | rollout568/808;ref45/train187-237;softOOM57 |
| 004108 | 2/1/4 | 16 (4×4) | 5 | **6** | 2 | 否 | rollout385/537;ACT n=96 |
| 011411 | 2/1/4 | **64 (8×8)** | 5 | 10 | 2 | 否 | rollout683/929;ref66/train419/690;util~90%;softOOM159 |
| 092357 | **4**/1/**2** | 64 (8×8) | 5 | 10 | 2 | 否 | softOOM0;ref260;**train step崩(NCCL 10min)** |
| 102136 | 4/1/2 | 64 (8×8) | 5 | 10 | 1 | **是** | ref260/logp250;**train step 跑~35min 无看门狗(fix 有效)**;但后段 softOOM 0→97/GPU满,未完成(cap杀) |
| 113259 | 2/**2**/2 | 64 (8×8) | 5 | 10 | 2 | 是 | ref128/logp114 完成;**train step 硬 OOM**(Triton CUDA OOM,softOOM44) |
| official-smoke-204024 | 2/1/4 | (math 32×8) | 单轮math | — | — | 否 | ref54/logp23/train298s 干净(注:不同脚本/数据,仅作 V1 "stack ok") |

**单变量对照关系(各结论的依据):**
- **V3** = 220350 vs 231242:只差**并行配置**(TP8/DP1↔TP2/DP4),其余全同 → ~37×。
- **V7** = 235534 vs 004108:只差 **num_trials**(10↔6)→ rollout 0.67×。
- **V8** = 235534 vs 011411:只差**并发**(16ep↔64ep)→ 4×ep 仅 1.2× rollout、util 30%→90%。
- **V5/V10** = 011411 vs 092357:只差**并行配置**(TP2/DP4↔TP4/DP2)→ 前向 66s↔260s、softOOM 159↔0、train 完成↔崩。
- **H1** = 092357 vs 102136:只差**reload-fix**(否↔是)→ 崩↔(>18min 未崩,进行中)。
- ⚠️ 220350(16ep/steps1)与 011411/092357(64ep/steps5)**多变量不同,禁止互比**。

---

## §6 关键配置/事实速查
- 数据:`alchemy_train_500.jsonl`(max_steps=20,num_trials原生10);`_steps5.jsonl`(max_steps=5);`_steps1.jsonl`(max_steps=1)。
- `ALCHEMY_NUM_TRIALS_CAP=N`:rollout 里截断到 N 个 trial(代码已加,fixed episode 的 num_trials 烤死在 env)。
- `ALCHEMY_TP` / `ALCHEMY_CP`:DP = 8/(TP×CP)。
- 官方 math 参考配置:TP2/CP1/DP4,max-tokens-per-gpu 9216,--sequence-parallel,--balance-data。
- debug 跑完务必:`ray job stop` + `rm -rf /data/user_data/qixinx/alchemy_runs/<RUNID>`。
