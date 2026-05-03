"""
Standalone test script to verify SGLang returns expert routing data for MoE models.

Usage:
    python tests/proxy/test_sglang_experts.py \
        --model-path /root/miles_local/checkpoints/Qwen3-30B-A3B \
        --tp-size 8 \
        --port 30000

What it does:
    1. Launches an SGLang server with enable_return_routed_experts=True
    2. Sends a /generate request with return_routed_experts=True
    3. Decodes the base64-encoded expert routing data
    4. Validates shape, range, and prints sample expert assignments
"""

import argparse
import multiprocessing
import signal
import sys
import time

import numpy as np
import pybase64
import requests


def launch_server(model_path: str, tp_size: int, port: int) -> multiprocessing.Process:
    """Launch SGLang server in a subprocess with expert routing enabled."""
    from sglang.srt.entrypoints.http_server import launch_server as _launch
    from sglang.srt.server_args import ServerArgs

    server_args = ServerArgs(
        model_path=model_path,
        tp_size=tp_size,
        port=port,
        host="127.0.0.1",
        enable_return_routed_experts=True,
        trust_remote_code=True,
    )

    multiprocessing.set_start_method("spawn", force=True)
    p = multiprocessing.Process(target=_launch, args=(server_args,))
    p.start()
    return p


def wait_for_server(base_url: str, timeout: int = 600) -> None:
    """Poll /health until the server is ready."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{base_url}/health", timeout=5)
            if r.status_code == 200:
                return
        except requests.ConnectionError:
            pass
        time.sleep(5)
    raise TimeoutError(f"Server not ready after {timeout}s")


def test_expert_output(base_url: str, num_layers: int, topk: int, num_experts: int) -> bool:
    """Send a generate request and validate expert routing output."""
    payload = {
        "text": "What is 2+2? Let me think step by step.",
        "sampling_params": {
            "max_new_tokens": 64,
            "temperature": 0.7,
        },
        "return_logprob": True,
        "return_routed_experts": True,
    }

    print(f"  Request: POST {base_url}/generate")
    print(f"  Payload: return_routed_experts=True, max_new_tokens=64")
    r = requests.post(f"{base_url}/generate", json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()

    # Check generated text
    text = data.get("text", "")
    print(f"\n  Generated text: {text[:200]}{'...' if len(text) > 200 else ''}")

    # Check meta_info for routed_experts
    meta_info = data.get("meta_info", {})
    experts_b64 = meta_info.get("routed_experts")

    if experts_b64 is None:
        print("\n  FAIL: 'routed_experts' not found in meta_info!")
        print(f"  Available meta_info keys: {list(meta_info.keys())}")
        return False

    # Decode base64 -> numpy array
    raw = np.frombuffer(pybase64.b64decode(experts_b64.encode("ascii")), dtype=np.int32)
    print(f"\n  Raw expert data: {len(raw)} int32 values")

    # Figure out num_tokens from the data
    expected_per_token = num_layers * topk
    if len(raw) % expected_per_token != 0:
        print(f"  FAIL: Data length {len(raw)} not divisible by (num_layers={num_layers} x topk={topk} = {expected_per_token})")
        return False

    num_tokens = len(raw) // expected_per_token
    experts = raw.reshape(num_tokens, num_layers, topk)

    print(f"  Expert array shape: {experts.shape}  (num_tokens={num_tokens}, num_layers={num_layers}, topk={topk})")
    print(f"  Expert ID range: [{experts.min()}, {experts.max()}]  (expected: [0, {num_experts - 1}])")

    # Validate range
    if experts.min() < 0 or experts.max() >= num_experts:
        print(f"  FAIL: Expert IDs out of range [0, {num_experts - 1}]!")
        return False

    # Print sample assignments
    print(f"\n  Sample expert assignments (first 3 tokens, first 4 layers):")
    for t in range(min(3, num_tokens)):
        print(f"    Token {t}:")
        for layer in range(min(4, num_layers)):
            print(f"      Layer {layer:2d}: experts {experts[t, layer].tolist()}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Test SGLang expert routing output")
    parser.add_argument("--model-path", required=True, help="Path to MoE model weights")
    parser.add_argument("--tp-size", type=int, default=8, help="Tensor parallel size (default: 8)")
    parser.add_argument("--port", type=int, default=30000, help="Server port (default: 30000)")
    parser.add_argument("--num-layers", type=int, default=48, help="Number of MoE layers (default: 48 for Qwen3-30B-A3B)")
    parser.add_argument("--topk", type=int, default=8, help="Router top-k (default: 8 for Qwen3-30B-A3B)")
    parser.add_argument("--num-experts", type=int, default=128, help="Total experts (default: 128 for Qwen3-30B-A3B)")
    parser.add_argument("--skip-launch", action="store_true", help="Skip server launch (connect to existing server)")
    args = parser.parse_args()

    base_url = f"http://127.0.0.1:{args.port}"
    server_proc = None

    def cleanup(signum=None, frame=None):
        if server_proc and server_proc.is_alive():
            print("\nCleaning up: terminating server...")
            server_proc.terminate()
            server_proc.join(timeout=10)
            if server_proc.is_alive():
                server_proc.kill()
        if signum is not None:
            sys.exit(1)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    try:
        # Step 1: Launch server
        if not args.skip_launch:
            print(f"[1/4] Launching SGLang server on port {args.port} with TP={args.tp_size}...")
            print(f"       Model: {args.model_path}")
            print(f"       enable_return_routed_experts=True")
            server_proc = launch_server(args.model_path, args.tp_size, args.port)
        else:
            print(f"[1/4] Skipping server launch (--skip-launch). Connecting to {base_url}...")

        # Step 2: Wait for server
        print(f"[2/4] Waiting for server to be ready...")
        wait_for_server(base_url)
        print(f"       Server is healthy!")

        # Step 3 & 4: Test expert output
        print(f"[3/4] Sending generate request with return_routed_experts=True...")
        passed = test_expert_output(base_url, args.num_layers, args.topk, args.num_experts)

        # Step 5: Summary
        print(f"\n[4/4] {'PASS: Expert routing output is working correctly!' if passed else 'FAIL: Expert routing output test failed.'}")
        sys.exit(0 if passed else 1)

    finally:
        cleanup()


if __name__ == "__main__":
    main()
