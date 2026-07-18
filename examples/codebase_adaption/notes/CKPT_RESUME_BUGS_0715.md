# cpu-offload 优化器的 ckpt resume 双 bug 与退役决定(2026-07-15)

## TL;DR

`--optimizer-cpu-offload --use-precision-aware-optimizer`(HybridDeviceOptimizer, HDO)
配置下保存的 torch_dist ckpt **无法全量恢复优化器状态**,上游从未修好
(NVIDIA/Megatron-LM #1842, radixark/Megatron-LM PR #66 closed 未合)。
**最终决定(用户拍板,2026-07-15 两次修订)**:当前 swecl-4b-actonly-6p6 这组**全程保持
offload 不变**(组内不换优化器实现;其续训一律 weights-only)。**之后所有新实验统一
`CODEBASE_NO_OFFLOAD=1`**,回归官方 `scripts/run-qwen3.5-4B.sh` 的标准优化器配置
(TE FusedAdam),bug 家族连根消失;新 run 从第一个 ckpt 起原生可全量恢复,不存在切换期。
加载端补丁 v3.1 已离线验证但**归档不启用**,仅在被迫退回 offload 时走实弹验收后启用。

## Bug 一:common.pt 列表长度按 rank 不同 → "Cannot merge two lists with different lengths (117 and 122)"

dp_reshardable 保存路径把 per-param state 列表写进 rank0-only 的 common.pt,
但列表长度 = 该 rank 的 param 条目数 + padding 条目数,4 个 DP rank 分别为
{117, 122, 122, 68}(两个 TP 组布局逐 chunk 相同,已用分片索引实测)。
加载时 `serialization.py:126` 的 `merge(common, 本地padding骨架)` 在长度不等的 rank 上直接抛错。

## Bug 二:padding 条目携带垃圾 `step` → "a Tensor with 32 elements cannot be converted to Scalar"

出生地 `distrib_optimizer.py:1667`(SIF 内):padding 插入无差别复制邻居条目的所有
tensor 键(含 `step`)→ `step = torch.empty(空洞长度)` 未初始化垃圾。落盘时
`if key == 'step': continue` 使 step 漏进 common.pt;padding 标记是
LocalNonpersistentObject 不落盘。我们每个 ckpt 里固定 12 条 (32,) 垃圾
(索引 21,23,36,38,51,53,74,76,89,91,104,106 = Qwen3.5 的 32 元素小参数造成的 bucket 空洞)。

### 为什么只有 HDO 用户中招

- HDO 底层 torch.optim.Adam **per-param 存 step 标量** → step 是唯一漏进 common 的东西,
  也是 param_state 列表存在于 common.pt 的唯一原因;
- 普通用户(TE FusedAdam)step 在 param_group 里,per-param state 无 step →
  条目全部被包成 ShardedTensor,空条目被剪掉,**common.pt 根本没有这个列表** → 全家免疫;
- 官方 4B/9B 脚本不开 offload;只有 ≥27B 官方配方开;且只有带优化器状态的真 resume 才触发
  (miles 在无 ckpt 时自动 no_load_optim, arguments.py:2103)。

### A_log 同源

2026-07-14 的 A_log 写坏(孤儿 fp32 dtype 桶被第一个 optimizer.step() 改成垃圾)同样
发生在 HDO 代码路径内。两大基建 bug 同居一个角落 → 退役 offload 一并根除。

## 补丁演化(patches/megatron/dict_utils.py)

- **v2**(PR #66 移植 + 截断分支):2026-07-15 实弹 gate-3 崩。验尸:①截断分支(DP3,
  68<117)保留 common 前 68 条,step 按 DP0 布局排,垃圾错位落到 real 参数 → fused adam 崩
  (原版本会加载报错,v2 把它变成带毒加载);②`x1[:]=x2` 分支(DP1/2)条目无 step →
  静默保持 dummy_step 的 1。
- **v3.1**(现存归档版):param_state 列表 merge 统一规则——长度不等整体采用本地结构,
  等长逐条合并;两种情况最后都从 common 取唯一合法标量 step(实测每 ckpt 全等)逐条
  **clone** 盖印(共享同一 0-dim tensor 会被 Adam 原地 `step+=1` 按参数数放大,静默爆炸——
  自查发现)。已用 iter_23 真实 common 数据对 4 rank × 两次 merge 全序列离线验证 PASS,
  含 clone 独立性、非 param_state 回归、TE 用户无感三项。**未实弹验证,不启用**。
- 若需启用:`CODEBASE_MEGATRON_CKPT_PATCH=1`(launch 脚本 bind-mount),
  必须重走四关验收 + 新增探针(加载后打印各 rank step 值与动量校验和,两个训练步后 step 恰 +2)。
- 保存端一行根修(pad_tensors 推导加 `and k != 'step'`)未移植——退役 offload 后无对象。

## 时间线与现场

- 首次全量 resume 失败(Cannot merge)→ weights-only 临时恢复:`.resumefail.log`
- v2 实弹 gate-3 崩(fused adam):`.ckptpatch-gate3fail.log`
- 每次 weights-only 恢复会把 Adam step 重置(iter_23 的 step=8 而非 24 即此故);
  lr constant 下仅偏置校正轻微次优,非致命。
- 新实验直接以 no-offload 起跑(不加载 HDO ckpt),从第一个 ckpt 起即可全量恢复,
  无任何切换期。若某个旧 run(HDO ckpt)想在 no-offload 配置下续训权重,首次加载须
  weights-only。新配置首个训练步 = 显存验证(4B@TP2/DP4 约 +6GB/卡),OOM 则回退。

## 上游参考

- https://github.com/NVIDIA/Megatron-LM/issues/1842 (同配置 resume 后 loss 0.31→10, open)
- https://github.com/radixark/Megatron-LM/pull/66 (长度失配修复尝试, closed 未合)
