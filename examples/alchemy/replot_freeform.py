"""离线从 traj 重画 free-form 的干净 DP4 曲线到一个全新 wandb run。
不训练、不占卡:只读已落盘的 traj json,重算每步 metrics,wandb.log(step=rollout_id)。
traj 全是 DP4(babel原 + n9-20),H100 的 traj 在它本地盘、不在 babel → 天然不含 H100。"""
import wandb, json, glob, statistics, collections, os
B = "/home/qixinx/miles/examples/alchemy"
oracle = json.load(open(f"{B}/eval/oracle_cache.json"))
RUN = "qwen3-4b-curr950-freeform-r120-e10-20260621-022402"
base = f"{B}/logs/{RUN}/traj/train"
rollouts = sorted(int(d.split("rollout_")[1]) for d in glob.glob(f"{base}/rollout_*"))
m = lambda xs: statistics.mean(xs) if xs else 0.0

wandb.init(project="miles-alchemy", entity="ivanracktic-tsinghua-university",
           name="freeform-DP4-clean-v2", group="qwen3-4B-curr950-freeform-rb8-n8-r120-e10",
           config={"note": "offline replot from traj; DP4 only (babel原+n9-20); no H100 contamination"})
# 用 rollout/step 作 x 轴(和 k38s7to1 的 define_metric 一致,这样两条 run 才对得齐)
wandb.define_metric("rollout/step")
wandb.define_metric("*", step_metric="rollout/step")
n_done = 0
for X in rollouts:
    eps = []
    for f in glob.glob(f"{base}/rollout_{X}/ep*.json"):
        try: eps.append(json.load(open(f)))
        except: pass
    if not eps: continue
    raw=[]; norm=[]; bt=collections.defaultdict(list); pot=cau=end=other=0; fk=[]; accs=[]
    for ep in eps:
        pts = ep.get("per_trial_scores") or []
        o = oracle.get(str(ep.get("episode_index")))
        for k, r in enumerate(pts):
            raw.append(float(r))
            if o and k < len(o) and o[k] > 0:
                nv = float(r)/o[k]; norm.append(nv); bt[k].append(nv)
        for t in (ep.get("turns") or []):
            a = str(t.get("action") or "").lower()
            if "potion" in a: pot += 1
            elif "cauldron" in a: cau += 1
            elif "end" in a: end += 1
            else: other += 1
        for w in (ep.get("write_audit") or []):
            fk.append(int(w.get("n_fk", 0) or 0))
            accs += [float(a) for a in (w.get("accs") or []) if a is not None]
    sv = pot+cau+end  # valid actions
    e = [x for k in range(5) for x in bt.get(k, [])]
    l = [x for k in range(5, 10) for x in bt.get(k, [])]
    log = {
        "alchemy_score/act_norm_mean": m(norm),
        "alchemy_score/act_raw_mean": m(raw),
        "alchemy_score/norm_improve": (m(l)-m(e)) if (e and l) else 0.0,
        "alchemy_action/potion_frac": pot/sv if sv else 0.0,
        "alchemy_write/fk_mean": m(fk),
        "alchemy_write/write_mean": m(accs),
        "rollout/step": X,
        "n_episodes": len(eps),
    }
    wandb.log(log)   # 不传 step;x 轴由 rollout/step(define_metric)决定,和 k38s7to1 对齐
    n_done += 1
wandb.finish()
print(f"DONE replot: {n_done} steps, rollout range {rollouts[0]}-{rollouts[-1]}")
