# data/ 文件账本(每个文件怎么来的)

> 本文件就住在 data/ 目录里(2026-07-16 从 notes/ 迁入,用户要求)。新增/弃用数据文件时同步更新本账本。
> **零引用的弃用文件已移入 `data/archive/`**(warmup_episodes.jsonl、baseline_warmup.json、smoke_textfmt_eval*.jsonl);其余"弃用"文件因被脚本默认值或金丝雀/聚合脚本引用而保留原位,勿删勿移。

更新:2026-07-16(初版 2026-07-14)。目录:`examples/codebase_adaption/data/`。
生成器所在仓库:clbench = `/home/qixinx/continual-learning-bench`。
凡"9B 口径"均为旧 JSON scaffold 离线测量;凡"textfmt 口径"均为纯文本 scaffold 在 miles eval-only 下测量。

## 训练 episode 数据

| 文件 | 行数/结构 | 来源 | 状态 |
|---|---|---|---|
| **episodes_6p6_hard.jsonl** | 262 行,每行 6+6(块=4 anchor 加噪易→难 + 2 难题) | `clbench/scripts/swecl/gen_6p6_episodes.py 260`(2026-07-14)。题池=全 240;分类用 baseline_4b_textfmt(mid: 0<avg4≤0.6;gap: 4B=0 且 9B>0;hardcore: 双 0)。anchor=mid 均匀抽样(mid<8 的仓库补入 easy;排序难度加 ±0.15 噪声,依赖拓扑保留);难题=gap 优先+硬核补足。9 迁移配对×2 方向,reps 按两仓库存货和加权;**近重复约束:组内任两 episode 逐位相同 ≤8/12**(实测 max=8,均值 2.7)。覆盖 218 题(mid 78 全、gap 23 全、hardcore 114) | **现役**(swecl-4b-actonly-6p6 起) |
| formal_episodes.jsonl | 258 行,5+5 | 7/12 生成,**生成脚本未存档**(一次性脚本,与 gen_warmup 同机制:题池=102 个 9B avg@4>0 可解题,wrand 分层+依赖拓扑)。内容可由数据自证:唯一题恰 102 | 已让位 6p6;被全部金丝雀脚本引用,保留(复现旧 run 必需) |
| warmup_episodes.jsonl | 54 行,5+5 | `clbench/scripts/swecl/gen_warmup_episodes.py 3`(7/11)。同 102 可解池,9 配对×2×3 reps | **已移 archive/**(零引用) |
| swecl_train_episodes.jsonl | 54 行,9+10 | `clbench/scripts/swecl/gen_train_episodes.py`(7/9)。题池=train_pool_avg0.6(240 剔 avg4>0.6),含大量 9B 不可解题 | 弃用,但为 run 脚本 PROMPT_DATA 缺省值,保留原位 |
| train_episodes.jsonl | 1 行占位 | 手写 seed(7/7),给 miles loader 提供非空 prompt-data;原 codebase_adaptation 19 题模式用 | 占位,别删(部分 wrapper 兜底引用) |

## eval episode 数据

| 文件 | 行数/结构 | 来源 | 状态 |
|---|---|---|---|
| heldout_episodes.jsonl | 5 行(order_rank 0-4) | 手写 seed(7/7):codebase_adaptation heldout,题序由 task 内部按 order_rank 排列。**注意:训练中 eval 实际展开为 6 题 tablib 子集/行**(轻量探针,非 19 题全量) | **现役**(训练中每 8 步 eval) |
| baseline_textfmt_eval.jsonl | 259 行单题(240 swecl split=train + 19 codebase split=heldout,均 no_memory) | `miles/examples/codebase_adaption/scripts/gen_textfmt_eval_data.py`(7/13) | 已消费(bl-4b-textfmt run) |
| baseline_textfmt_eval_patch.jsonl | 30 行 | 同脚本 `--patch`(7/13):l5-16 GLIBC 崩掉的缺样本题重测名单 | 已消费 |
| baseline_textfmt_eval_small.jsonl | 10 行 | 同脚本 `--small`(冒烟) | 冒烟用 |
| icl_textfmt_eval.jsonl | 5 行 9+10(no_memory) | 同脚本:题序从官方 clbench 5-run 轨迹提取并双组 assert(traces/2026-07-12T16-25-00.\*Z) | 已消费(icl-4b-textfmt) |
| replace_textfmt_eval.jsonl | 5 行 9+10(memory 模式) | 同上 | 已消费(iclmem-4b-textfmt) |
| smoke_textfmt_eval.jsonl / eval9.jsonl | 各 1 行(3 题 / 9 题 tablib) | 7/13 纯文本 scaffold 冒烟用手写 | **已移 archive/**(零引用) |

## baseline artifact(gain 的分母)

| 文件 | 覆盖 | 口径 | 来源 | 状态 |
|---|---|---|---|---|
| **baseline_4b_textfmt.json** | 259(240+19) | **textfmt avg@4** | `scripts/aggregate_textfmt_eval.py baseline`(7/14 00:17):bl-4b-textfmt + bl-4b-textfmt-patch 轨迹聚合,补测题整题替换 | **现役**(训练 gain 用它) |
| baseline_4b_merged.json | 259 | swecl 侧 JSON avg@4;**19 题 codebase 侧=单样本**(7/11 官方 clbench trace `2026-07-11T20-52-31.105204Z` 的 baseline 阶段,reward=1-(steps-1)/40 恒等式可验) | 7/12 合并 | 弃用口径,但被 aggregate/eval_suite 脚本引用,保留原位 |
| baseline_9b_merged.json | 259(240 缺 1) | swecl 侧 JSON avg@4(9B 过夜 pass@4 run 的 passk_detail);codebase 侧沿用旧 baseline | 7/12 合并 | 参考用(9B 无 textfmt 口径);6p6 生成器用它判 gap |
| baseline_merged.json | 251(232+19) | 9B JSON avg@4 | 7/9 首版合并(见 SWECL_INTEGRATION_DESIGN.md §2.3) | 弃用,但为 run 脚本 BASELINE_ARTIFACT 缺省值,保留原位 |
| baseline_warmup.json | 102+ | 9B JSON avg@4 | gen_warmup_episodes.py 附带产物(7/11) | **已移 archive/**(零引用) |

## 上游源数据(不在本目录,但一切由它来)

- `clbench/data/swe_bench_cl/full.jsonl`:240 题全池(含 repo、难度标注、依赖关系、
  `baseline_qwen35_9b.{avg4,solved4}` 字段);
- codebase_adaptation 19 题(tablib 9 + tenacity 10):clbench 官方 `data/codebase_adaptation/final-dataset.jsonl`。

## 相关
[[SWECL_INTEGRATION_DESIGN]](题池/合并设计)· [[WRITE_COLLAPSE_ANALYSIS_0714]](6p6 数据的动机)·
[[SLEEP_WAKE_COLLAPSE_ROOTCAUSE]](textfmt baseline 重测的动机)
