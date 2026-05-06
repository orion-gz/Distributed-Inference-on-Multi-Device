"""
==============
run_node_b.py
==============

recv_step():
    Receive [token ids] [activation] sent by A. 
    If n_tokens=0, treat the session as terminated.
    
device_b_forward():
    This uses the same hook injection method as 'run_device_b' in 'split_inference.py'. 
    It injects the activation received from A into 'layers[split_layer]' and executes 'model.model(input_ids)'. 
    It obtains the logit by applying 'norm' and 'lm_head' to the last hidden state and returns the next token ID using 'argmax'.

generate():
    autoregressive decoding. 
    At every step, execute the front layer of A with the current input_ids, send the activation and token IDs to B, receive the next token from B, and append it to input_ids.
    Session termination is signaled by sending n_tokens=0.
"""
import argparse
import socket
import struct
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from transport import recv_tensor

DEFAULT_MODEL = "K-intelligence/Midm-2.0-Base-Instruct"
LISTEN_HOST = "0.0.0.0"
LISTEN_PORT = 9000

PHASE_PREFILL = 0
PHASE_DECODE = 1
PHASE_END = 255


def load_model(model_id: str):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb, device_map="auto"
    )
    model.eval()
    return tokenizer, model

def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Connection closed")
        buf.extend(chunk)
    return bytes(buf)

def recv_step(conn: socket.socket, device: str):
    raw = _recv_exact(conn, 1)
    phase = struct.unpack(">B", raw)[0]

    if phase == PHASE_END:
        return PHASE_END, None, None

    raw_n = _recv_exact(conn, 4)
    n = struct.unpack(">I", raw_n)[0]
    
    raw_ids = _recv_exact(conn, n * 4)
    ids = list(struct.unpack(f">{n}i", raw_ids))
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    
    activation = recv_tensor(conn, device=device)
    return phase, input_ids, activation

def device_b_forward(model, activation: torch.Tensor, split_layer: int, input_ids: torch.Tensor, past_kv=None) -> int:
    layer_dtype = next
    injected = [False]

    def _inject(module, args, kwargs):
        if injected[0]:
            return
        injected[0] = True
        if args:
            return (activation,) + args[1:], kwargs
        kwargs["hidden_states"] = activation
        return args, kwargs

    handle = model.model.layers[split_layer].register_forward_pre_hook(
        _inject, with_kwargs=True
    )
    try:
        with torch.no_grad():
            outputs = model.model(input_ids, past_key_values=past_kv, use_cache=True)
        lm_dtype = model.lm_head.weight.dtype
        hidden = model.model.norm(outputs.last_hidden_state).to(lm_dtype)
        logits = model.lm_head(hidden)
    finally:
        handle.remove()

    next_id = logits[0, -1].argmax().item()
    return next_id, outputs.past_key_values


def serve(tokenizer, model, split_layer: int, host: str, port: int):
    device = str(model.device)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen(1)
        print(f"Device B ready  {host}:{port}  split_layer={split_layer}")

        while True:
            conn, addr = server.accept()
            print(f"\nDevice A connected: {addr}")

            past_kv = None
            step = 0

            with conn:
                while True:
                    t0 = time.perf_counter()
                    phase, input_ids, activation = recv_step(conn, device)

                    if phase == PHASE_END:
                        print("Session ended")
                        break

                    recv_ms = (time.perf_counter() - t0) * 1000

                    if phase == PHASE_PREFILL:
                        past_kv = None
                        
                    t1 = time.perf_counter()
                    next_id, past_kv = device_b_forward(model, activation, split_layer, input_ids, past_kv)
                    infer_ms = (time.perf_counter() - t1) * 1000

                    label = "prefill" if phase == PHASE_PREFILL else "decode"
                    tok_str = tokenizer.decode([next_id])
                    print(f"[{step+1:3d}][{label}] '{tok_str}'  "
                          f"recv={recv_ms:.0f}ms  infer={infer_ms:.0f}ms")
                    step += 1

                    conn.sendall(struct.pack(">I", next_id))

            print("Connection closed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default=DEFAULT_MODEL)
    parser.add_argument("--split",  type=int, default=24)
    parser.add_argument("--host",   default=LISTEN_HOST)
    parser.add_argument("--port",   type=int, default=LISTEN_PORT)
    args = parser.parse_args()

    tokenizer, model = load_model(args.model)
    serve(tokenizer, model, args.split, args.host, args.port)