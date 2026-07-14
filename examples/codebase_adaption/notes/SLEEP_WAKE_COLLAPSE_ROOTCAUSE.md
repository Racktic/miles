# rollout_1 崩塌根因调研:sleep/wake 后 CUDA graph 回放陈旧 mamba 状态指针

日期:2026-07-14。三个网络调研 agent(sglang 仓库 / RL 框架社区 / torch_memory_saver 机制)交叉验证 + 本仓自有实验证据链。

## 一、问题陈述

miles + sglang colocate RL 训练 Qwen3.5-4B(混合架构:GDN 线性注意力 + conv,即 mamba 类递归状态):
**rollout_0 健康(ACT≈0.21-0.23,解题 76-90/160),引擎第一次 sleep/wake + 权重重传之后,rollout_1 起
ACT≈0、解题≈0**。生成病理:语言流畅但标识符拼写打乱(如 `_formated_remaninet`)、格式纪律丢失、
幻觉路径 6→616、每题第 1 轮(短 prompt)即坏 43.8%。

## 二、本仓实验证据链(判决前的排除工作)

| 实验 | 结论 |
|---|---|
| lr=0 金丝雀(旧 JSON scaffold) | 权重零变化照崩 → 排除训练更新 |
| lr=0 金丝雀(新纯文本 scaffold) | 照崩 → 排除输出格式/json_schema 约束 |
| --check-weight-update-equal 体检 | 文本模型权重逐字节正确(仅 visual.* 未覆盖,已证无害)→ 排除权重传输 |
| 训练样本逐 token 审计 | loss 段与采样原文逐字节一致 → 排除打包/掩码 |
| reward 结构审计 | 19/20 组有方差 → 排除 reward 信号 |
| alchemy 对照(同 SIF、同引擎配置、同模型) | 短生成低负载不复现 → 指向负载相关的引擎状态问题 |
| flush 日志 | 睡前+更新后均 "Cache flushed successfully" → 排除显性 flush 失败 |
| mamba usage | rollout_0/1 均 ≤0.07 → 排除槽位泄漏打满形态 |

## 三、根因(头号,判决书级匹配)

### sglang PR #27140 — "fix: recapture CUDA graphs for RL online weight updates"(至今未合入)
https://github.com/sgl-project/sglang/pull/27140

**机制**:混合模型的 conv_state/ssm_state 递归状态 buffer 挂在 torch_memory_saver 的 `kv_cache`
标签下。sleep 时物理页释放(虚拟地址保留),训练期间显存被共卡的 Megatron 重排,resume 时该
buffer 可能映射到**不同 GPU 物理位置/新分配的张量地址**;而 **CUDA graph 捕获时把旧 data_ptr
烤死在图里**,decode 回放照旧地址读 → 读到垃圾递归状态。**权重不受影响**(update_weights 用
原地 copy_,地址从不变——这就是 weights_checker 永远通过的原因)。

**报告者场景与我们逐条相同**:slime colocate 训 Qwen3.5-35B-A3B,step 0 正常,一轮训练+同步后
全乱码、无报错。作者验证打 patch 后 RL 正常收敛。PR 评论区 2026-07-11 有 miles+GB200 同类报告。

**对我们全部观察的解释**:权重体检通过(地址不变)/ lr=0 复现(与权重值无关)/ 双 scaffold 复现
(与格式无关)/ **alchemy 不复现(低负载时 buffer 恰好落回原地址——运气)**/ 第 1 轮即坏(状态
污染从首 token 生效)/ 拼写级乱码(逐 token 记忆被垃圾污染)。

### 次号(可叠加):我们的 SIF 缺 #24954
mamba radix cache 快照竞态修复(overlap scheduler 下未等 copy_done 就快照 ping-pong buffer,
污染态写进缓存),上游 2026-05-19 合入;**我们的 SIF(sglang 0.5.13.dev28+g8d51c43,基线 HEAD
6月4日)经进容器 grep 代码验证不含该修复**(无 `mamba_value_donated` 标记)。
https://github.com/sgl-project/sglang/issues/24221 → https://github.com/sgl-project/sglang/pull/24954
(已验证我们含 #26430 GemmaRMSNorm 修复,该项排除。)

## 四、机制背景(为什么"参数一模一样,推理天差地别")

推理输出 = 权重 × 运行时状态。torch_memory_saver 的 resume **不保证内容**(除 weights 可选
cpu_backup):KV 池、**mamba 状态池、req_to_token 表**恢复后全是垃圾;一致性完全依赖 release
时那一次 flush_cache 的账本重置 + "无任何组件持有跨睡眠的 GPU 指针"这一假设。CUDA graph 恰恰
违反该假设(指针烤死在图里)。相关佐证:
- torch_memory_saver #61/#47:cuda graph 区域的 pause/resume 本身不可靠(仅 preload 模式支持)
- sglang #7939/#15246、vllm #17103/#20627/#29341:sleep/wake 后 gibberish 是跨引擎通病
- sglang #28679:Qwen3.6 GDN 在 A100 纯 serving 也有间歇性静默退化(open)——GDN 状态管理仍有坑
- 社区防御:verl 历史默认 enforce_eager(禁 graph)、禁 "CUDA graph + free_cache_engine" 同开

## 四·五、四链接精读补充(2026-07-14,用户指定链接逐一核验)

1. **#27140(未合入,仍 open)**:修法只动 `weight_updater.py` —— resume_memory_occupation 时
   `mark_cuda_graphs_stale()` + 权重更新后 `recapture_cuda_graphs_after_weight_update()`。
   作者 Gao016 验证 qwen35b RL 打 patch 后正常收敛;评论区有 miles+GB200 同症状报告。
2. **miles #1634**:与本 bug 无关(GB200 HybridEP 的 THD padding 修复),排除。
3. **slime #2091**:EazyReal 给出同一机制解释;Gao016 2026-07-08 称 `slimerl/slime:latest`
   已修;yuanlehome 2026-07-13 唱反调称"真正的修复是 #26430"。**对我们这个争议已可裁决**:
   进容器验证我们 SIF **已含 #26430**(layernorm.py:608 `torch.add(param.data, 1.0,
   out=self.gemma_weight)`,原地更新)却照样崩 → #26430 对我们不是答案,#27140 机制(mamba
   conv/ssm 状态 buffer 搬家 + graph 烤死旧指针)才是。"最新镜像已修"的说法未验内容物:若它
   只是带 #26430 的新 sglang,对我们无效;需拆镜像确认是否含 recapture 逻辑再信。
4. **#26430(已合入 5/28)**:GemmaRMSNorm 每次 forward 新分配 gemma_weight 张量破坏 CUDA
   graph 地址假设,同一 bug 家族;我们已含,排除(第三节已记)。

### 重大新发现:我们的 SIF 里已有"重捕获"完整机器,只是没接到 tensor 路径

进容器逐行核验:
- `model_runner.py:2977 init_device_graphs()` —— 丢弃旧 graph_runner、按**当前** buffer 地址
  重新捕获全部 CUDA graph。**这就是 #27140 疗法的本体,已经在我们构建里。**
- 开关 `recapture_cuda_graph: bool = False`(io_struct.py:1431)**只挂在
  `UpdateWeightFromDiskReqInput`**(从磁盘重载权重的请求)上:tp_worker.py:101 →
  model_runner.py:1873 `if recapture_cuda_graph: self.init_device_graphs()`。
- miles 实际用的 **`UpdateWeightsFromTensorReqInput`(io_struct.py:1475)没有这个字段**,
  tp_worker.py:159 的 tensor 更新路径零 recapture;HTTP `/update_weights_from_disk` 虽带开关
  但会用磁盘旧权重覆盖训练权重,不可借用。
- miles 调用点:`miles/backends/sglang_utils/sglang_engine.py:287` 纯 JSON POST
  `/update_weights_from_tensor`。

**由此得出的最小修复(候选路线 a)**:用本项目现成 pydeps overlay 给 sglang 打 3 处小补丁
(① io_struct 给 tensor 请求类加同名字段;② tp_worker.py:159 透传;③ model_runner
tensor 更新成功后 `if recapture_cuda_graph: self.init_device_graphs()`),miles 侧 payload
加一行 `"recapture_cuda_graph": True`。时机正确性:colocate 流程是 wake → update_weights →
rollout,更新时点在 resume 之后,重捕获拿到的就是新地址。代价:每个训练步多一次 graph
capture(数十秒量级,相对 27-63min 的 rollout 可忽略);风险点:#27140 评审区提过反复重捕获
的 graph 内存池回收问题,需盯 avail mem 日志。

## 五、行动方案

1. **确诊金丝雀(待批)**:lr=0 + `--sglang-disable-cuda-graph`,2 rollout。rollout_1 恢复健康
   → #27140 定案;该 flag 同时即应急 workaround(decode 慢 ~30-50%,训练可先恢复)。
   若仍坏 → 加 `--disable-radix-cache` 区分 #24954 机制。
2. **正规修复(按优先级)**:
   a. **最小 overlay 补丁(推荐)**:把 SIF 里现成的 recapture 机器接到 tensor 路径
      (3 处 sglang 小改 + miles payload 1 行,见"四·五"节),无需取 PR 分支、无需重建 SIF;
   b. cherry-pick #27140 分支 + #24954(PR 基于远新于我们 6/4 基线的代码,冲突风险高);
   c. 换 slimerl/slime:latest 镜像(内容物未验,#2091 有争议,拆验后再信)。
3. **应急 workaround**:`--sglang-disable-cuda-graph`(即确诊金丝雀本身,decode 慢 30-50%,
   训练可先恢复)。
4. **上游反馈**:在 #27140 评论区补充我们的复现(Qwen3.5-4B / A100 / lr=0 复现 / weights_checker
   通过 / 病理画像),与 miles GB200 报告互证,推动合入。

## 六、来源索引

- 头号根因:[sglang PR #27140](https://github.com/sgl-project/sglang/pull/27140) ·
  [slime issue #2091](https://github.com/THUDM/slime/issues/2091)(同症状用户串,slime 镜像已修)
- 次号:[issue #24221](https://github.com/sgl-project/sglang/issues/24221) /
  [PR #24954](https://github.com/sgl-project/sglang/pull/24954)(SIF 验证缺失)
- 已排除:[PR #26430](https://github.com/sgl-project/sglang/pull/26430)(SIF 已含)·
  tms #71 cpu_backup 损坏(我们未开)· #12099 flush 失败(我们 flush 均成功)
- 机制:[torch_memory_saver](https://github.com/fzyzcjy/torch_memory_saver)(#61/#47/#36)·
  [sglang RL 文档](https://docs.sglang.io/docs/advanced_features/sglang_for_rl) ·
  [RFC #7009 多阶段唤醒](https://github.com/sgl-project/sglang/issues/7009) ·
  [#22243 kv-cache-cpu-backup 提案](https://github.com/sgl-project/sglang/issues/22243) ·
  [PyTorch 博客: Hybrid Models Meet SGLang](https://pytorch.org/blog/hybrid-models-meet-sglang-more-than-full-attention/)
- 同类通病:sglang #7939/#15246/#6367;vllm #17103/#20627/#29341/#32714/#9744;verl #995
- hybrid 其余已知坑:#20774(offload 未适配 mamba dtype)/ #20763(9B tp>1 乱码)/ #24121 /
  #26941(ping-pong 槽泄漏,6月修)/ #28185(int8 池回收,6月修)/ #29449 / #29349 / #29633

## 相关
- [[EXPERIMENT_LEDGER_0713]](实验台账) · [[TRAINING_COLLAPSE_ONPOLICY_FIX]](数据侧修复史)
