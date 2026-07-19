#!/usr/bin/env python3
"""thinking on/off 写 memory 配对对照: 同 30 个真实 WRITE 场景, 各生成一次。
sglang Engine 用 spawn 起子进程会重新 import 本文件 —— 主逻辑必须在 __main__ 守卫内。"""
import json, re, sys
sys.path.insert(0, '/home/qixinx/miles/examples/codebase_adaption')
from codebase_advantage import memory_format_ok

MODEL = '/data/user_data/qixinx/Qwen3.5-4B'


def split_think(text):
    if '</think>' in text:
        th, _, rest = text.partition('</think>')
        return th.replace('<think>', '').strip(), rest.strip()
    return '', text.strip()


def main():
    scen_path = sys.argv[1] if len(sys.argv) > 1 else '/home/qixinx/think_write_test/scenarios.json'
    out_path = sys.argv[2] if len(sys.argv) > 2 else '/home/qixinx/think_write_test/results.json'
    scen = json.load(open(scen_path))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    prompts, meta = [], []
    for mode in ('think', 'nothink'):
        for s in scen:
            txt = tok.apply_chat_template(
                s['input_messages'], tokenize=False, add_generation_prompt=True,
                enable_thinking=(mode == 'think'))
            prompts.append(txt)
            meta.append((mode, s))

    import sglang as sgl
    engine = sgl.Engine(model_path=MODEL, tp_size=1, mem_fraction_static=0.7,
                        trust_remote_code=True)
    outs = engine.generate(prompts, {"temperature": 1.0, "top_p": 1.0,
                                     "max_new_tokens": 4096})

    results = []
    for (mode, s), o in zip(meta, outs):
        raw = o['text'] if isinstance(o, dict) else o
        think, mem = split_think(raw)
        prev = next((m['content'] for m in reversed(s['input_messages'])
                     if isinstance(m.get('content'), str)
                     and '### Repository Knowledge' in m.get('content', '')), '')
        results.append({
            'mode': mode, 'episode': s['episode'], 'trial_pos': s['trial_pos'],
            'instance_id': s['instance_id'],
            'think_chars': len(think), 'mem_chars': len(mem),
            'compliant': bool(memory_format_ok(mem, prev or None)),
            'bullets': len(re.findall(r'^\s*[-*]\s+\S', mem, re.M)),
            'paths': len(re.findall(r'`[^`]*/[^`]*`', mem)),
            'think': think[:2000], 'memory': mem,
        })
    json.dump(results, open(out_path, 'w'),
              ensure_ascii=False, indent=1)

    for mode in ('think', 'nothink'):
        R = [r for r in results if r['mode'] == mode]
        print(f"[{mode}] n={len(R)} 合规率={sum(r['compliant'] for r in R)}/{len(R)}"
              f" 平均bullets={sum(r['bullets'] for r in R)/len(R):.1f}"
              f" 平均路径引用={sum(r['paths'] for r in R)/len(R):.1f}"
              f" 平均正文长度={sum(r['mem_chars'] for r in R)/len(R):.0f}字"
              f" 平均think长度={sum(r['think_chars'] for r in R)/len(R):.0f}字")
    byk = {}
    for r in results:
        byk.setdefault((r['episode'], r['trial_pos']), {})[r['mode']] = r
    both = [v for v in byk.values() if len(v) == 2]
    win = sum(1 for v in both if v['think']['compliant'] and not v['nothink']['compliant'])
    lose = sum(1 for v in both if not v['think']['compliant'] and v['nothink']['compliant'])
    print(f"配对合规: think胜 {win}, nothink胜 {lose}, 平 {len(both)-win-lose}")
    print("COMPARE_DONE")


if __name__ == "__main__":
    main()
