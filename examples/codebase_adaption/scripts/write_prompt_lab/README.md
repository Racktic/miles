# WRITE prompt 离线实验台(2026-07-18/19)

30 个真实 WRITE 场景(delta rollout_0 抽样, 早/中/晚题位分层)上的配对 A/B:
prompt v1(原版)/ v3(三明治框架语+成败反思引导)× thinking on/off。

- `run_compare.py <scenarios.json> <results.json>` — sglang 离线引擎批量生成 + 严格合规/串台指标
- `think_write.sbatch` — general 分区 1 GPU 提交模板(注意 SSL unset 防线, RUNBOOK W8)
- 场景/结果原始档案在 babel `/home/qixinx/think_write_test/`(scenarios{,_v3}.json,
  results.json=v1, results_v3_used.json=v3, v3_cases_side_by_side.md=30 case 并排精读)

结论(数字见 notes/WRITE_PROMPT_V3_AB_0719.md): v3 使串台 12-16/30→2-4/30,
严格合规 16-17/30→23-24/30; v3 之下 think/nothink 差距缩到噪声级, thinking 改为可选。
生产开关: CODEBASE_WRITE_PROMPT_V3=1 / CODEBASE_WRITE_THINKING=1(默认全关)。
