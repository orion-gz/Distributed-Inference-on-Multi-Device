"""
run_node_b.py — Device B (뒷단 레이어)

역할:
  1. Device A의 TCP 연결 대기
  2. Activation 수신
  3. Layer SPLIT..end + norm + lm_head 실행
  4. 다음 토큰 ID를 Device A로 반환

실행:
  python run_node_b.py --split 16 --port 9000
"""

import argparse
import inspect
import socket
import struct
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from transport import recv_tensor, send_tensor

# ── 설정 ──────────────────────────────────────────────────────────────────────
DEFAULT_MODEL = "K-intelligence/Mi-dm-2.0"
LISTEN_HOST = "0.0.0.0"
LISTEN_PORT = 9000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_id: str):
    print(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    return tokenizer, model


def build_causal_mask(seq_len: int, dtype: torch.dtype, device) -> torch.Tensor:
    mask = torch.full((seq_len, seq_len), float("-inf"), dtype=dtype, device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)


def call_layer(layer, hidden: torch.Tensor, position_ids: torch.Tensor,
               causal_mask: torch.Tensor) -> torch.Tensor:
    sig = inspect.signature(layer.forward)
    kwargs = dict(position_ids=position_ids, use_cache=False)
    if "cache_position" in sig.parameters:
        kwargs["cache_position"] = torch.arange(
            position_ids.shape[-1], device=hidden.device
        )
    else:
        kwargs["attention_mask"] = causal_mask
    return layer(hidden, **kwargs)[0]


def device_b_forward(model, activation: torch.Tensor, split_layer: int,
                     seq_len: int) -> int:
    """뒷단 레이어 실행 → 다음 토큰 ID 반환"""
    device = activation.device
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    causal_mask = build_causal_mask(seq_len, activation.dtype, device)

    h = activation
    with torch.no_grad():
        for i in range(split_layer, len(model.model.layers)):
            h = call_layer(model.model.layers[i], h, position_ids, causal_mask)
        h = model.model.norm(h)
        logits = model.lm_head(h)

    next_token_id = logits[0, -1].argmax().item()
    return next_token_id


def serve(tokenizer, model, split_layer: int, host: str, port: int):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen(1)
        print(f"Device B listening on {host}:{port} (split_layer={split_layer})")

        while True:
            conn, addr = server.accept()
            print(f"\nDevice A connected: {addr}")
            step = 0

            with conn:
                while True:
                    # seq_len 수신 (0이면 세션 종료)
                    raw = conn.recv(4)
                    if len(raw) < 4:
                        break
                    seq_len = struct.unpack(">I", raw)[0]
                    if seq_len == 0:
                        print("Session ended by Device A")
                        break

                    # Activation 수신
                    t0 = time.perf_counter()
                    activation = recv_tensor(conn, device=DEVICE)
                    recv_ms = (time.perf_counter() - t0) * 1000

                    # 뒷단 레이어 실행
                    t0 = time.perf_counter()
                    next_token_id = device_b_forward(model, activation, split_layer, seq_len)
                    infer_ms = (time.perf_counter() - t0) * 1000

                    token_str = tokenizer.decode([next_token_id])
                    print(f"[{step+1:3d}] '{token_str}' "
                          f"| recv={recv_ms:.0f}ms infer={infer_ms:.0f}ms")
                    step += 1

                    # 토큰 ID 반환
                    conn.sendall(struct.pack(">I", next_token_id))

            print("Connection closed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--split", type=int, default=16)
    parser.add_argument("--host", default=LISTEN_HOST)
    parser.add_argument("--port", type=int, default=LISTEN_PORT)
    args = parser.parse_args()

    tokenizer, model = load_model(args.model)
    serve(tokenizer, model, args.split, args.host, args.port)