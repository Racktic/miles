#!/usr/bin/env python3
"""Minimal side-by-side trajectory viewer for eyeballing two eval rollouts of the SAME run.
No params / no algorithm / no definitions — just: pick an episode, compare left vs right.
Reads the live eval-traj dump format (turns/raw_act/action/summary_in/summaries).

Edit the CONFIG block (two dirs + labels), run, then open the served HTML.
"""
import json, os, glob, html as _html, sys

# ================================================================ CONFIG
# CLI override:  traj_diff_viewer.py  A_DIR  A_LABEL  B_DIR  B_LABEL  OUT
if len(sys.argv) >= 6:
    A_DIR, A_LABEL, B_DIR, B_LABEL, OUT = sys.argv[1:6]
else:
    RUN = "/home/qixinx/miles/examples/alchemy/logs/qwen3-4b-curr950-actonly-w3-expl03-budgetv3-r120-e10-20260630/traj/eval/hard20"
    A_DIR, A_LABEL = f"{RUN}/rollout_9",  "rollout_9"
    B_DIR, B_LABEL = f"{RUN}/rollout_79", "rollout_79"
    OUT = "/home/qixinx/miles/examples/alchemy/notes/traj_diff_9_vs_79.html"
# ============================================================ end CONFIG

def load_dir(d):
    """episode_index -> dict(per_trial_scores, trials=[{trial, score, mem_in, mem_out, turns:[...]}])"""
    eps = {}
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        j = json.load(open(f))
        epi = j.get("episode_index")
        summaries = j.get("summaries", []) or []
        pts = j.get("per_trial_scores", []) or []
        # group turns by trial
        by_trial = {}
        for t in j.get("turns", []):
            k = int(t.get("trial", 0))
            by_trial.setdefault(k, []).append(t)
        trials = []
        for k in sorted(by_trial):
            ts = by_trial[k]
            mem_in = next((x.get("summary_in") for x in ts if x.get("summary_in")), "") or ""
            turns = [dict(turn=x.get("turn"), step=x.get("step"), user=x.get("user") or "",
                          raw_act=x.get("raw_act") or "", action=x.get("action"),
                          action_int=x.get("action_int"), valid=x.get("valid"),
                          reward=x.get("reward"), finish=x.get("act_finish")) for x in ts]
            trials.append(dict(trial=k, score=(pts[k] if k < len(pts) else None),
                               mem_in=mem_in, mem_out=(summaries[k] if k < len(summaries) else None),
                               turns=turns))
        eps[epi] = dict(per_trial=pts, n_turns=j.get("num_turns"), trials=trials)
    return eps

DATA = {"A": dict(label=A_LABEL, eps=load_dir(A_DIR)),
        "B": dict(label=B_LABEL, eps=load_dir(B_DIR))}
payload = json.dumps(DATA, ensure_ascii=False).replace("</", "<\\/")

HTML = r"""<!doctype html><html lang=en><head><meta charset=utf-8>
<title>Alchemy traj diff — __A__ vs __B__</title>
<style>
 body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:0;background:#0f1115;color:#e6e6e6}
 header{padding:11px 18px;background:#171a21;border-bottom:1px solid #2a2f3a}
 h1{font-size:16px;margin:0}.sub{color:#9aa4b2;font-size:12px;margin-top:3px}
 .controls{padding:10px 18px;background:#13161c;border-bottom:1px solid #2a2f3a}
 select{background:#1b1f27;color:#e6e6e6;border:1px solid #2a2f3a;border-radius:6px;padding:6px 9px;font-size:13px}
 .badge{font-size:11px;color:#9aa4b2}
 .wrap{padding:12px 18px}
 .cols{display:grid;grid-template-columns:1fr 1fr;gap:14px}
 .pane{background:#13161c;border:1px solid #2a2f3a;border-radius:9px;padding:10px;overflow:auto;max-height:86vh}
 .pane h3{margin:2px 0 8px;font-size:13px;color:#7fb1ff;position:sticky;top:0;background:#13161c;padding:4px 0}
 .scores{font-size:11px;color:#9aa4b2;margin-bottom:8px;white-space:pre-wrap}
 .trial{border:1px solid #242a35;border-radius:8px;margin-bottom:10px;overflow:hidden}
 .trial>summary{cursor:pointer;padding:7px 10px;background:#1b1f27;font-size:13px;font-weight:600;list-style:none}
 .trial>summary::-webkit-details-marker{display:none}
 .rew-pos{color:#5bd97e}.rew-neg{color:#ff6b6b}.rew-zero{color:#9aa4b2}
 .mem{white-space:pre-wrap;background:#101822;border-left:3px solid #2d6cdf;padding:8px 10px;margin:8px;font-size:11.5px;border-radius:4px;color:#bfe0ff}
 .mem.in{border-left-color:#7a5cff;color:#cdbfff}
 .turn{border-top:1px solid #20262f;padding:7px 10px;font-size:12px}
 .role{font-size:10.5px;color:#9aa4b2;text-transform:uppercase;letter-spacing:.04em;margin-top:5px}
 .user{white-space:pre-wrap;color:#c8cdd6;font-size:11.5px}.asst{white-space:pre-wrap;color:#e6e6e6;font-size:11.5px}
 .act{display:inline-block;background:#15202b;border:1px solid #2a3a4a;border-radius:5px;padding:1px 7px;margin-top:4px;font-size:11px;color:#9fd0ff}
 details.mono>summary{cursor:pointer;color:#7fb1ff;font-size:11px;padding:2px 0}
</style></head><body>
<header><h1>Symbolic Alchemy — trajectory diff: <span style=color:#7fb1ff>__A__</span> (left) vs <span style=color:#7fb1ff>__B__</span> (right)</h1>
<div class=sub>same episode on both sides · eval hard20 · collapse/expand trials · faint = raw agent text</div></header>
<div class=controls>Episode: <select id=epsel></select> <span class=badge id=epinfo></span></div>
<div class=wrap><div class=cols><div class=pane id=paneA></div><div class=pane id=paneB></div></div></div>
<script>
const D=__PAYLOAD__, esc=s=>s==null?'':String(s);
const rc=r=>r>0?'rew-pos':r<0?'rew-neg':'rew-zero';
function renderPane(el,side,epi){
 const inst=D[side],ep=inst.eps[epi];
 if(!ep){el.innerHTML='<h3>'+inst.label+'</h3><div class=sub>no data for episode '+epi+'</div>';return;}
 const tot=ep.per_trial.reduce((a,b)=>a+(b||0),0);
 let h='<h3>'+inst.label+' · episode '+epi+' <span class=badge>('+ep.trials.length+' trials · '+ep.n_turns+' turns · total '+tot+')</span></h3>';
 h+='<div class=scores>per-trial score: ['+ep.per_trial.map(x=>x).join(', ')+']</div>';
 for(const t of ep.trials){
  h+='<details class=trial'+(t.trial<2?' open':'')+'><summary>Trial '+t.trial+' · <span class="'+rc(t.score)+'">score '+(t.score>0?'+':'')+esc(t.score)+'</span> <span class=badge>('+t.turns.length+' turns)</span></summary>';
  if(t.mem_in)h+='<details class=mono><summary>🧠 memory shown to agent (window in)</summary><div class="mem in">'+esc(t.mem_in)+'</div></details>';
  for(const tn of t.turns){
   h+='<div class=turn><div class=role>turn '+esc(tn.turn)+' · env</div><div class=user>'+esc(tn.user)+'</div>';
   h+='<div class=role>agent</div><div class=asst>'+esc(tn.raw_act)+'</div>';
   h+='<div class=act>action: '+esc(tn.action)+' (int '+esc(tn.action_int)+', '+(tn.valid?'valid':'INVALID')+', finish='+esc(tn.finish)+')</div>';
   h+=' <span class="badge '+rc(tn.reward||0)+'">reward '+esc(tn.reward)+'</span></div>';}
  if(t.mem_out)h+='<div class=mem>📝 memory written AFTER this trial:\n'+esc(t.mem_out)+'</div>';
  else h+='<div class=mem style="border-color:#444;color:#888">(last trial — no memory written after)</div>';
  h+='</details>';}
 el.innerHTML=h;}
const epsA=Object.keys(D.A.eps).map(Number),epsB=new Set(Object.keys(D.B.eps).map(Number));
const eps=epsA.filter(x=>epsB.has(x)).sort((x,y)=>x-y);
const sel=document.getElementById('epsel');
sel.innerHTML=eps.map(e=>'<option value='+e+'>episode '+e+'</option>').join('');
function show(epi){document.getElementById('epinfo').textContent='(left='+D.A.label+', right='+D.B.label+', episode '+epi+')';
 renderPane(document.getElementById('paneA'),'A',epi);renderPane(document.getElementById('paneB'),'B',epi);}
sel.onchange=()=>show(+sel.value);show(eps[0]);
</script></body></html>"""

out = HTML.replace("__PAYLOAD__", payload).replace("__A__", A_LABEL).replace("__B__", B_LABEL)
open(OUT, "w").write(out)
print("wrote", OUT, "|", os.path.getsize(OUT), "bytes")
print(f"  A={A_LABEL}: {len(DATA['A']['eps'])} eps   B={B_LABEL}: {len(DATA['B']['eps'])} eps")
