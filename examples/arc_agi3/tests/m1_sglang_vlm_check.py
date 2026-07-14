"""M1 gate: does SGLang serve Qwen3.5-4B as a VLM and return output logprobs?

Mirrors the rollout path exactly: processor builds input_ids (with image placeholders) from a
chat message containing a rendered grid image; we POST input_ids + image_data + return_logprob to
the local SGLang /generate and check for coherent text and meta_info.output_token_logprobs.

Run INSIDE the miles container, after the server is up on port 30000:
  apptainer exec --nv --bind /data,/home/qixinx/miles <sif> \
    python3 /home/qixinx/miles/examples/arc_agi3/m1_sglang_vlm_check.py
"""
import base64
import io
import json
import sys
import urllib.request

sys.path.insert(0, "/home/qixinx/miles")
from examples.arc_agi3.grid_render import render_grid

CKPT = "/data/hf_cache/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
URL = "http://127.0.0.1:30000/generate"


def post(payload, timeout=180):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(URL, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def b64_png(im):
    buf = io.BytesIO()
    im.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def main():
    from transformers import AutoProcessor, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(CKPT, trust_remote_code=True)

    # ---- 1) text-only logprob sanity ----
    r0 = post({
        "text": "The capital of France is",
        "sampling_params": {"max_new_tokens": 8, "temperature": 0},
        "return_logprob": True,
    })
    t_lp = r0["meta_info"].get("output_token_logprobs")
    print("[TEXT] text:", repr(r0["text"]))
    print("[TEXT] logprobs:", "present" if t_lp else "MISSING", "n=", len(t_lp or []))

    # ---- 2) image path (mirrors rollout: input_ids + image_data + return_logprob) ----
    grid = [[0] * 64 for _ in range(64)]
    for rr in range(10, 20):
        for cc in range(10, 20):
            grid[rr][cc] = 9  # a blue square block
    img = render_grid(grid)

    messages = [
        {"role": "system", "content": "You are a careful grid-game player."},
        {"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": "This image is a 64x64 grid. In one short sentence: what colour value forms the solid square block, and roughly where is it (x,y)?"},
        ]},
    ]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    from qwen_vl_utils import process_vision_info
    images, videos = process_vision_info(messages)
    proc_kwargs = {"text": [text], "images": images, "return_tensors": "pt"}
    if videos:
        proc_kwargs["videos"] = videos
    proc_out = proc(**proc_kwargs)
    input_ids = proc_out["input_ids"][0].tolist()
    image_data = [b64_png(im) for im in images]

    print(f"[IMG] input_ids len={len(input_ids)}  n_images={len(image_data)}  "
          f"mm_keys={[k for k in proc_out.keys() if k not in ('input_ids','attention_mask')]}")

    r = post({
        "input_ids": input_ids,
        "image_data": image_data,
        "sampling_params": {"max_new_tokens": 64, "temperature": 0},
        "return_logprob": True,
    })
    mi = r["meta_info"]
    i_lp = mi.get("output_token_logprobs")
    print("[IMG] text:", repr(r["text"])[:400])
    print("[IMG] logprobs:", "present" if i_lp else "MISSING", "n=", len(i_lp or []))
    print("[IMG] prompt_tokens:", mi.get("prompt_tokens"))

    ok = bool(r["text"]) and bool(i_lp) and bool(t_lp)
    print("\nM1 RESULT:", "PASS" if ok else "FAIL",
          "(VLM serves image+text and returns output logprobs)" if ok else "(see above)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
