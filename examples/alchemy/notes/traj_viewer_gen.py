#!/usr/bin/env python3
"""Config-driven Alchemy trajectory comparison viewer.

Edit only the CONFIG block (RUNS + GROUPS). Everything else is auto-derived from the
eval output dirs: median seed, eval norm_score / norm_improve, memory window, and the
full per-trial trajectories (with the end_trial-turn fix). The explore-judge prompt is
pulled live from alchemy_judge.py.

Usage:  python3 traj_viewer_gen.py          # -> traj_viewer_v2.html
"""
import json, os, glob, statistics, re, html as _html

# ---------------------------------------------------------------- paths
TRAJ_ROOT  = "/home/qixinx/miles/examples/alchemy/logs/offline_evals/trajectories"
JUDGE_FILE = "/home/qixinx/miles/examples/alchemy/alchemy_judge.py"
OUT        = "/home/qixinx/miles/examples/alchemy/notes/traj_viewer_v2.html"

# ================================================================ CONFIG
# Each run: prefix = eval dir prefix (the tool globs "<prefix>-rep*" + "<prefix>", reads
# each rep's normalized.json, and auto-picks the MEDIAN-score rep). The other fields are
# training config that the eval output does NOT contain, so you must type them (rarely).
# Auto-filled per run: eval_score, eval_improve, window, picked seed.
RUNS = {
  "win1":    dict(prefix="eval-ckpt-sig4norm119",          run="sig4normimprove-userdata-r120-e10-20260625-231959", iter=119,
                  train_act=1, train_write=1, write_signal="downstream_norm_improve*", beta=0.0),
  "norm119": dict(prefix="eval-sig4norm-w3-b0-iter119",    run="sig4norm-w3-r120-e10-20260627", iter=119,
                  train_act=1, train_write=1, write_signal="downstream_norm_improve", beta=0.0),
  "act119":  dict(prefix="eval-actonly-w3-b0-iter119",     run="actonly-w3-r120-e10-20260627", iter=119,
                  train_act=1, train_write=0, write_signal="- (WRITE not trained)", beta=0.0),
  "norm99":  dict(prefix="eval-sig4norm-w3-b0-iter99",     run="sig4norm-w3-r120-e10-20260627", iter=99,
                  train_act=1, train_write=1, write_signal="downstream_norm_improve", beta=0.0),
  "expl99":  dict(prefix="eval-sig4norm-w3-expl03-iter99", run="sig4norm-w3-expl03-r120-e10-20260628b", iter=99,
                  train_act=1, train_write=1, write_signal="downstream_norm_improve", beta=0.3),
}

GROUPS = [
  dict(id="g1", title="Group 1 · memory window 1 → 3",
       question="Both ACT+WRITE trained (norm_improve); the ONLY change is memory window 1 → 3.",
       A="win1", B="norm119", diff_keys=["window"]),
  dict(id="g2", title="Group 2 · train WRITE or not (window=3)",
       question="At window=3: ACT-only (WRITE not trained) vs ACT+WRITE (WRITE = norm_improve).",
       A="act119", B="norm119", diff_keys=["train_write", "write_signal"]),
  dict(id="g3", title="Group 3 · with / without explore reward (window=3)",
       question="At window=3: baseline (β=0) vs +ACT explore reward (β=0.3); both iter99.",
       A="norm99", B="expl99", diff_keys=["beta"]),
]

COMMON = "Qwen3-4B-Instruct-2507 · curr950 · GRPO · r120 rb8 n8 global64 · KL0.01 lr1e-6 · eval=hard20 summary-replace +explore_v2 no-thinking"
# ============================================================ end CONFIG

def _reps_for(prefix):
    """All rep dirs for a prefix that have a valid normalized.json -> [(dir, score, improve)]."""
    cands = sorted(set(glob.glob(os.path.join(TRAJ_ROOT, prefix + "-rep*")) +
                       ([os.path.join(TRAJ_ROOT, prefix)] if os.path.isdir(os.path.join(TRAJ_ROOT, prefix)) else [])))
    out = []
    for d in cands:
        nj = os.path.join(d, "normalized.json")
        if not os.path.exists(nj): continue
        n = json.load(open(nj))
        if n.get("performance_mean") is None: continue
        out.append((d, float(n["performance_mean"]), n.get("i_score_mean")))
    return out

def _pick_median(reps):
    reps = sorted(reps, key=lambda r: r[1])
    return reps[len(reps) // 2]            # middle by score

def _load_trajectories(d):
    eps = {}
    for fn in sorted(os.listdir(os.path.join(d, "traj"))):
        if not fn.endswith(".json"): continue
        j = json.load(open(os.path.join(d, "traj", fn)))
        summaries = j.get("summaries", [])
        # group turns by trial; an "end the trial" turn ENDS the current trial (the `trial`
        # field mislabels it as the next trial), so attach it to the current trial.
        trials, last_real = {}, 0
        for t in j.get("transcript", []):
            parsed = t.get("parsed") or {}
            if parsed.get("kind") == "end_trial":
                tr = last_real
            else:
                tr = t.get("trial", 0)
                if tr < 0: tr = last_real
                last_real = tr
            trials.setdefault(tr, []).append(dict(
                turn=t.get("turn"), user=t.get("user"), assistant=t.get("assistant"),
                parsed=t.get("parsed"), reward=t.get("reward")))
        tl = []
        for tr in sorted(trials):
            turns = trials[tr]
            tl.append(dict(trial=tr, reward=sum((x["reward"] or 0) for x in turns),
                           memory=(summaries[tr] if tr < len(summaries) else None), turns=turns))
        eps[j.get("episode_index")] = dict(window=j.get("summary_memory_window_size"), n_trials=len(tl), trials=tl)
    return eps

def build_run(key, cfg):
    reps = _reps_for(cfg["prefix"])
    if not reps:
        raise SystemExit(f"[{key}] no rep dirs with normalized.json for prefix {cfg['prefix']}")
    d, score, improve = _pick_median(reps)
    eps = _load_trajectories(d)
    win = next(iter(eps.values()))["window"] if eps else None
    params = dict(run=cfg["run"], iter=cfg["iter"], window=win,
                  train_act=cfg["train_act"], train_write=cfg["train_write"],
                  write_signal=cfg["write_signal"], beta=cfg["beta"],
                  eval_score=round(score, 3), eval_improve=(round(improve, 3) if improve is not None else None),
                  seed=os.path.basename(d).split("-rep")[-1] if "-rep" in os.path.basename(d) else "1",
                  n_reps=len(reps))
    return dict(params=params, eps=eps)

DATA = {k: build_run(k, c) for k, c in RUNS.items()}

# definitions + live judge prompt
DEFS = """
<h2>Task — Symbolic Alchemy</h2>
<p>Each <b>episode</b> hides a latent rule (which potion transforms which stone, and how the stone's reward changes). An episode has K <b>trials</b> (here 10) that all <b>share the same hidden rule</b>. Each trial spans several <b>turns</b>: the agent applies potions to stones, then cashes stones in the cauldron for their (latent) reward. <code>r<sub>k</sub></code> = total reward cashed in trial k. After trial k the agent <b>writes</b> a natural-language memory <code>M<sub>k</sub></code>; trial k+1 <b>acts</b> conditioned on the last <b>W</b> memories (W = memory window).</p>
<h2>GRPO advantage (both streams)</h2>
<p>Within a group of sibling samples: <code>adv<sub>i</sub> = (reward<sub>i</sub> − mean<sub>group</sub>) / (std<sub>group</sub> + ε)</code>. ACT and WRITE are whitened in <i>separate</i> groupings.</p>
<h2>ACT reward</h2>
<p>= the trial's raw task score <code>r<sub>k</sub></code>, whitened within <code>(group_index, trial_pos)</code>.</p>
<h2>WRITE reward signals — reward for memory M<sub>k</sub></h2>
<p>(whitened within <code>(group_index, k+1)</code>; the last trial's memory has no downstream trial → not trained.)</p>
<ul>
 <li><b>transition_acc</b> (default): does <code>M<sub>k</sub></code> predict the next trial's observed (stone, potion) → result transitions.</li>
 <li><b>downstream</b> (sig 3): <code>R(M<sub>k</sub>) = r<sub>k+1</sub></code></li>
 <li><b>downstream_improve</b> (sig 4): <code>R(M<sub>k</sub>) = r<sub>k+1</sub> − r<sub>k</sub></code></li>
 <li><b>downstream_norm_improve</b> (<span class=hl>sig4norm</span>, used here):
   <div class=eq><code>R(M<sub>k</sub>) = r<sub>k+1</sub>/oracle[k+1] − r<sub>k</sub>/oracle[k]</code></div>
   <code>oracle[t]</code> = oracle policy's score on this episode at trial t (skipped if ≤ 0) → rewards memory that raises the <i>oracle-normalized</i> next-trial score.</li>
</ul>
<h2>ACT explore reward (β) — added in expl03</h2>
<p>A deepseek judge scores the <b>memory delta</b> <code>M<sub>k-1</sub> → M<sub>k</sub></code> on 4 dims, each ∈ {0,1,2}: <code>new_discoveries · error_correction · verification_targets · non_redundant_change</code>.</p>
<div class=eq><code>explore_score = (sum of the 4 dims) / 8 ∈ [0,1]</code> (equal 1:1:1:1)</div>
<p>Mixed at the <b>advantage level</b> (not added to raw reward):</p>
<div class=eq><code>explore_adv<sub>i</sub> = (explore_score<sub>i</sub> − mean)/(std + ε)</code> within <code>(group_index, trial_pos)</code>; &nbsp; <code>act_adv<sub>i</sub> ← act_adv<sub>i</sub> + β · explore_adv<sub>i</sub></code>, <span class=hl>β = 0.3</span> (β=0 ⇒ no-op).</p>
<p class=note>From our analysis: replaying a known optimal recipe yields zero memory-delta ⇒ zero explore bonus, so β=0.3 over-optimized this term — the agent keeps experimenting in late trials instead of cashing the known +15, lowering task score (see Group 3).</p>
"""
_JS = open(JUDGE_FILE).read()
_m = re.search(r'SYSTEM_PROMPT\s*=\s*"""(.*?)"""', _JS, re.S)
DEFS += ('<h2>explore judge — exact prompt (deepseek, live from alchemy_judge.py)</h2>'
         '<details open><summary>system prompt</summary><pre class=prompt>'
         + _html.escape(_m.group(1).strip() if _m else "(not found)") + '</pre></details>'
         '<details><summary>user prompt template</summary><pre class=prompt>'
         + _html.escape("Previous memory M_(k-1):\n{prev}\n\nUpdated memory M_k:\n{cur}\n") + '</pre></details>')

payload = json.dumps(dict(instances=DATA, groups=GROUPS, common=COMMON, defs=DEFS), ensure_ascii=False).replace("</", "<\\/")

HTML = """<!doctype html><html lang=en><head><meta charset=utf-8>
<title>Alchemy trajectory comparison viewer</title>
<style>
 body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:0;background:#0f1115;color:#e6e6e6}
 header{padding:12px 18px;background:#171a21;border-bottom:1px solid #2a2f3a}
 h1{font-size:17px;margin:0}.sub{color:#9aa4b2;font-size:12px;margin-top:4px}
 .tabs{display:flex;gap:8px;padding:10px 18px;background:#13161c;border-bottom:1px solid #2a2f3a;flex-wrap:wrap}
 .tab{padding:7px 12px;border:1px solid #2a2f3a;border-radius:7px;cursor:pointer;font-size:13px;background:#1b1f27}
 .tab.active{background:#2d6cdf;border-color:#2d6cdf;color:#fff}
 .wrap{padding:14px 18px}.q{color:#cbd5e1;font-size:13px;margin:6px 0 12px}
 table.params{border-collapse:collapse;font-size:12px;margin-bottom:14px;width:100%;max-width:980px}
 table.params td,table.params th{border:1px solid #2a2f3a;padding:5px 9px;text-align:left}
 table.params th{background:#1b1f27;color:#9aa4b2}
 td.diff{background:#3a2a12;color:#ffd479;font-weight:600}td.same{color:#cbd5e1}
 .controls{margin:8px 0 14px}
 select{background:#1b1f27;color:#e6e6e6;border:1px solid #2a2f3a;border-radius:6px;padding:6px 8px;font-size:13px}
 .cols{display:grid;grid-template-columns:1fr 1fr;gap:14px}
 .pane{background:#13161c;border:1px solid #2a2f3a;border-radius:9px;padding:10px;overflow:auto;max-height:80vh}
 .pane h3{margin:2px 0 8px;font-size:13px;color:#7fb1ff}
 .trial{border:1px solid #242a35;border-radius:8px;margin-bottom:10px;overflow:hidden}
 .trial>summary{cursor:pointer;padding:7px 10px;background:#1b1f27;font-size:13px;font-weight:600;list-style:none}
 .trial>summary::-webkit-details-marker{display:none}
 .rew-pos{color:#5bd97e}.rew-neg{color:#ff6b6b}.rew-zero{color:#9aa4b2}
 .mem{white-space:pre-wrap;background:#101822;border-left:3px solid #2d6cdf;padding:8px 10px;margin:8px;font-size:12px;border-radius:4px;color:#bfe0ff}
 .turn{border-top:1px solid #20262f;padding:7px 10px;font-size:12px}
 .role{font-size:11px;color:#9aa4b2;text-transform:uppercase;letter-spacing:.04em}
 .user{white-space:pre-wrap;color:#c8cdd6}.asst{white-space:pre-wrap;color:#e6e6e6}
 .act{display:inline-block;background:#15202b;border:1px solid #2a3a4a;border-radius:5px;padding:1px 7px;margin-top:4px;font-size:11px;color:#9fd0ff}
 .badge{font-size:11px;color:#9aa4b2;font-weight:400}
 .defsbox{max-width:920px;font-size:13px;line-height:1.6}
 .defsbox h2{font-size:15px;color:#7fb1ff;border-bottom:1px solid #2a2f3a;padding-bottom:4px;margin:18px 0 8px}
 .defsbox code{background:#101822;border:1px solid #20303f;border-radius:4px;padding:1px 5px;font-size:12.5px;color:#bfe0ff}
 .defsbox .eq{background:#101822;border-left:3px solid #2d6cdf;padding:7px 10px;margin:7px 0;border-radius:4px}
 .defsbox .hl{background:#3a2a12;color:#ffd479;padding:1px 5px;border-radius:4px;font-weight:600}
 .defsbox .note{background:#1b1410;border:1px solid #5a3a1a;border-radius:6px;padding:8px 11px;color:#ffcaa0;font-size:12.5px}
 .defsbox summary{cursor:pointer;color:#7fb1ff;font-size:13px}
 .defsbox pre.prompt{white-space:pre-wrap;background:#0d1117;border:1px solid #20303f;border-radius:6px;padding:10px;font-size:12px;color:#cbd5e1;max-height:440px;overflow:auto;margin-top:6px}
</style></head><body>
<header><h1>Symbolic Alchemy — trajectory comparison viewer (config-driven)</h1><div class=sub id=common></div></header>
<div class=tabs id=tabs></div>
<div class=wrap>
 <div class=defsbox id=defsbox style=display:none></div>
 <div id=cmpbox>
  <div class=q id=q></div><div id=paramsbox></div>
  <div class=controls>Episode: <select id=epsel></select> <span class=badge id=epinfo></span></div>
  <div class=cols><div class=pane id=paneA></div><div class=pane id=paneB></div></div>
 </div>
</div>
<script>
const PAYLOAD=__PAYLOAD__;const INST=PAYLOAD.instances,GROUPS=PAYLOAD.groups;
document.getElementById('common').textContent=PAYLOAD.common;
const PKEYS=[['run','run id'],['iter','iter'],['window','memory window'],['train_act','train ACT'],
 ['train_write','train WRITE'],['write_signal','WRITE signal'],['beta','ACT explore β'],
 ['eval_score','eval norm_score (median seed)'],['eval_improve','eval norm_improve (median seed)'],['seed','median seed (of n_reps)']];
let cur=GROUPS[0];
const rc=r=>r>0?'rew-pos':r<0?'rew-neg':'rew-zero', esc=s=>s==null?'':String(s);
function renderParams(g){
 const a=INST[g.A].params,b=INST[g.B].params;
 let h='<table class=params><tr><th>parameter</th><th>'+a.run.split('-r120')[0]+' (A)</th><th>'+b.run.split('-r120')[0]+' (B)</th></tr>';
 for(const [k,label] of PKEYS){let av=esc(a[k]),bv=esc(b[k]);
  if(k==='seed'){av+=' / '+a.n_reps;bv+=' / '+b.n_reps;}
  const cls=(g.diff_keys.includes(k)||av!==bv)?'diff':'same';
  h+='<tr><td>'+label+'</td><td class='+cls+'>'+av+'</td><td class='+cls+'>'+bv+'</td></tr>';}
 h+='</table><div class=sub>Common: '+PAYLOAD.common+' &nbsp;|&nbsp; highlighted = difference between the two runs</div>';
 document.getElementById('paramsbox').innerHTML=h;}
function renderPane(el,key,epi){
 const inst=INST[key],ep=inst.eps[epi];
 if(!ep){el.innerHTML='<h3>'+inst.params.run+'</h3><div class=sub>no data for this episode</div>';return;}
 let h='<h3>'+inst.params.run.split('-r120')[0]+' · iter'+inst.params.iter+' · win'+inst.params.window+' <span class=badge>('+ep.n_trials+' trials)</span></h3>';
 for(const t of ep.trials){
  h+='<details class=trial'+(t.trial<3?' open':'')+'><summary>Trial '+t.trial+' · <span class="'+rc(t.reward)+'">reward '+(t.reward>0?'+':'')+t.reward+'</span></summary>';
  for(const tn of t.turns){
   h+='<div class=turn><div class=role>turn '+esc(tn.turn)+' · env</div><div class=user>'+esc(tn.user)+'</div>';
   h+='<div class=role style="margin-top:5px">agent</div><div class=asst>'+esc(tn.assistant)+'</div>';
   if(tn.parsed)h+='<div class=act>action: '+esc((tn.parsed.desc||tn.parsed.kind))+'</div>';
   h+=' <span class="badge '+rc(tn.reward||0)+'">reward '+esc(tn.reward)+'</span></div>';}
  if(t.memory)h+='<div class=mem>📝 Memory (written AFTER this trial):\\n'+esc(t.memory)+'</div>';
  else h+='<div class=mem style="border-color:#444;color:#888">(last trial — no memory written after it)</div>';
  h+='</details>';}
 el.innerHTML=h;}
function commonEpisodes(g){const a=Object.keys(INST[g.A].eps).map(Number),b=new Set(Object.keys(INST[g.B].eps).map(Number));return a.filter(x=>b.has(x)).sort((x,y)=>x-y);}
function selectGroup(g){cur=g;document.getElementById('q').textContent='Q: '+g.question;renderParams(g);
 const eps=commonEpisodes(g),sel=document.getElementById('epsel');
 sel.innerHTML=eps.map(e=>'<option value='+e+'>ep'+e+'</option>').join('');sel.onchange=()=>showEp(+sel.value);showEp(eps[0]);}
function showEp(epi){document.getElementById('epinfo').textContent='(left = A, right = B, same episode '+epi+')';
 renderPane(document.getElementById('paneA'),cur.A,epi);renderPane(document.getElementById('paneB'),cur.B,epi);}
document.getElementById('defsbox').innerHTML=PAYLOAD.defs;
const TABS=[{id:'defs',title:'📖 Definitions & formulas'}].concat(GROUPS);
document.getElementById('tabs').innerHTML=TABS.map(g=>'<div class=tab data-id='+g.id+'>'+g.title+'</div>').join('');
function selectTab(id){document.querySelectorAll('.tab').forEach(t=>t.classList.toggle('active',t.dataset.id===id));
 const d=(id==='defs');document.getElementById('defsbox').style.display=d?'':'none';document.getElementById('cmpbox').style.display=d?'none':'';
 if(!d)selectGroup(GROUPS.find(g=>g.id===id));}
document.querySelectorAll('.tab').forEach(t=>t.onclick=()=>selectTab(t.dataset.id));
selectTab('g1');
</script></body></html>"""

open(OUT, "w").write(HTML.replace("__PAYLOAD__", payload))
print("wrote", OUT, "| real size", os.path.getsize(OUT), "bytes")
for k, v in DATA.items():
    p = v["params"]
    print(f"  {k}: seed=rep{p['seed']}/{p['n_reps']}  score={p['eval_score']}  improve={p['eval_improve']}  win={p['window']}  ({len(v['eps'])} eps)")
