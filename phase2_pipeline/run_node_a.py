"""
==============
run_node_a.py
==============

device_a_forward():
    device_a_forward executes the full forward with 'output_hidden_states=True' and extracts 'hidden_states[split_layer]' as the activation.
    This is the same method as run_baseline in split_inference.py
    
send_step():
    The reason for sending token IDs is that Node B uses a hook injection method, so input_ids are needed to run the entire forward.

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

from transport import recv_tensor, send_tensor

DEFAULT_MODEL = "K-intelligence/Midm-2.0-Base-Instruct"
NODE_B_HOST   = "127.0.0.1"
NODE_B_PORT   = 9000


def load_model(model_id: str):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb, device_map="auto"
    )
    model.eval()
    return tokenizer, model


def device_a_forward(model, input_ids: torch.Tensor, split_layer: int) -> torch.Tensor:
    """Embedding + Layer 0..split_layer-1 실행, activation 반환"""
    with torch.no_grad():
        outputs = model.model(input_ids, output_hidden_states=True, use_cache=False)
    # hidden_states[k] = output after layer k-1
    return outputs.hidden_states[split_layer]


def send_step(sock: socket.socket, input_ids: torch.Tensor, activation: torch.Tensor):
    ids = input_ids[0].cpu().tolist()
    n   = len(ids)
    sock.sendall(struct.pack(f">I{n}i", n, *ids))
    sent = send_tensor(sock, activation)
    return sent


def generate(tokenizer, model, text: str, split_layer: int,
             max_new_tokens: int, node_b_addr: tuple) -> str:

    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    print(f"\nInput: '{text}'  ({input_ids.shape[1]} tokens)")
    print(f"Connecting to Device B {node_b_addr} ...")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(node_b_addr)
        print("Connected.\n")

        for step in range(max_new_tokens):
            t0 = time.perf_counter()

            activation = device_a_forward(model, input_ids, split_layer)
            sent_b     = send_step(s, input_ids, activation)

            raw = s.recv(4)
            if len(raw) < 4:
                print("Device B disconnected")
                break
            next_id = struct.unpack(">I", raw)[0]

            ms      = (time.perf_counter() - t0) * 1000
            tok_str = tokenizer.decode([next_id])
            print(f"[{step+1:3d}] '{tok_str}'  A={ms:.0f}ms  payload={sent_b/1024:.1f}KB")

            if next_id == tokenizer.eos_token_id:
                print("\n[EOS]")
                break

            input_ids = torch.cat(
                [input_ids, torch.tensor([[next_id]], device=model.device)], dim=1
            )

        s.sendall(struct.pack(">I", 0))  

    prompt_len = tokenizer(text, return_tensors="pt").input_ids.shape[1]
    return tokenizer.decode(input_ids[0, prompt_len:], skip_special_tokens=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default=DEFAULT_MODEL)
    parser.add_argument("--text",       default="인공지능이란 무엇인가요?")
    parser.add_argument("--split",      type=int, default=24)
    parser.add_argument("--max_tokens", type=int, default=50)
    parser.add_argument("--host",       default=NODE_B_HOST)
    parser.add_argument("--port",       type=int, default=NODE_B_PORT)
    args = parser.parse_args()

    tokenizer, model = load_model(args.model)
    result = generate(tokenizer, model, args.text, args.split,
                      args.max_tokens, (args.host, args.port))
    print(f"\n=== 생성 결과 ===\n{result}")