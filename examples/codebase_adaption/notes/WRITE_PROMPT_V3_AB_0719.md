# WRITE prompt v3 + thinking 离线 A/B(2026-07-19)

> 起因:精读发现 WRITE 输出的头号质量杀手是**串台**(该写记忆时输出 ACT 式 bash/做题话术),
> base 模型串台率 ~40-50%;结构根源是 ~80 余条做题消息之后才出现一条改写指令。
> 用户设计三明治框架语 + 成败反思引导(v3, 逐字审定),在 30 个真实场景上配对验证。

## 实验设置

- 场景:delta run rollout_0 的 30 个真实 WRITE 调用(input_messages 原样,早/中/晚题位各 10);
- 配对:同场景 × {prompt v1, v3} × {thinking on, off},sglang 离线引擎,temp 1.0 对齐训练;
- 指标:串台率(记忆混入 ```bash/提交标记/做题话术)、严格合规率(`memory_format_ok`)、
  内容 case 精读(用户标准:成功轨迹→提炼成功关键;失败轨迹→定位失败原因)。

## 结果

| 指标 | v1 think | v1 nothink | **v3 think** | **v3 nothink** |
|---|---|---|---|---|
| 串台率 | 12/30 | 16/30 | **2/30** | **4/30** |
| 严格合规 | 17/30 | 16/30 | **23/30** | **24/30** |
| FAILED 案例保持双节结构 | 1/4 | (并入左) | **4/4** | (并入左) |

case 精读:v3 的 FAILED 案例给出精确根因链(如 `_split_gcd` 空参取 `a[0]` 的 IndexError
+ guard-clause 修法);成功案例显式提炼可复用关键("dict() 拷贝防 attrs 引用突变"+验证清单)。

样本长度经济学:输入均值 ~17.1k tokens(最长 ~32k,**超过训练 seq_length 24576,
超长 WRITE 样本会被长度过滤截断/丢弃——WRITE 覆盖率的隐性天花板,与本改动无关但记录在案**);
thinking 输出 996 vs 574 tokens——整样本 +2.4%(算力视角),受训 token +73%(信号视角)。

## 结论与落地

1. **v3 必上**:一次 prompt 修改,串台 ~50%→~10%,合规 ~55%→~78%,失败归因从结构崩坏变为可用;
2. **thinking 降级为可选**:v1 下它是抗串台锚(12 vs 16),v3 把锚定做进 prompt 后差距缩到噪声级
   (2 vs 4;合规打平),每次多 ~420 token 输出开销;
3. 生产开关(默认全关,现役 run 重启口径不漂移):`CODEBASE_WRITE_PROMPT_V3=1`(prompts.py 的
   sandwich+引导)、`CODEBASE_WRITE_THINKING=1`(WRITE 专属 enable_thinking;`</think>` 后可见
   正文进记忆,think 块只训不外泄,未闭合视为无输出回退旧记忆);两者已进 runtime env 白名单。

## 教训存档(过程中的两次自伤)

- GRPO 8 兄弟共享 episode_id:任何按 `(episode_id, trial_pos)` 的映射都会被兄弟覆盖——
  **对齐 trial 一律用来源文件或 input_messages 精确匹配**(本次曾因此误报"判分反馈造假");
- 判分反馈("Submission PASSED/FAILED")自 7/13 起就正确地只进 WRITE 输入(rollout ~:815),
  形式上就是对话末条 FEEDBACK;不要把"哪行代码追加的"误当"输入结构差异"。

实验台与原始数据:`scripts/write_prompt_lab/` + babel `/home/qixinx/think_write_test/`。
相关:[[SEVEN_POINTS_AUDIT_0713]] §2(判分可见性决策)· RUNBOOK W8(提交环境防线)。
