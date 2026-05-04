"""
run_node_a.py — Device A (오케스트레이터 겸 앞단 레이어)

역할:
  1. 텍스트 입력 받기
  2. Embedding + Layer 0..SPLIT-1 실행
  3. Activation을 Device B로 TCP 전송
  4. Device B로부터 다음 토큰 수신
  5. 자동 회귀 루프 (max_new_tokens)

실행:
  python run_node_a.py --text "안녕하세요" --split 16 --max_tokens 50
"""

import argparse
import socket
import struct
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from transport import recv_tensor, send_tensor

# ── 설정 ──────────────────────────────────────────────────────────────────────
DEFAULT_MODEL = "K-intelligence/Mi-dm-2.0"
NODE_B_HOST = "127.0.0.1"  # Phase 3에서 WebRTC P2P IP로 교체
NODE_B_PORT = 9000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── 모델 로드 ─────────────────────────────────────────────────────────────────
def load_model(model_id: str):
    print(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    return tokenizer, model


# ── Device A 추론 ─────────────────────────────────────────────────────────────
def device_a_forward(
    model, input_ids: torch.Tensor, split_layer: int
) -> torch.Tensor:
    """Embedding + Layer 0..split_layer-1 실행, activation 반환"""
    with torch.no_grad():
        outputs = model.model(
            input_ids,
            output_hidden_states=True,
            use_cache=False,
        )
    # hidden_states[k] = output after layer k-1
    # hidden_states[split_layer] = Device B로 넘길 activation
    return outputs.hidden_states[split_layer]


def generate(
    tokenizer,
    model,
    text: str,
    split_layer: int,
    max_new_tokens: int,
    node_b_addr: tuple,
) -> str:
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    generated = input_ids.clone()
    print(f"\nInput: '{text}'")
    print(f"Generating (max {max_new_tokens} tokens)...\n")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(node_b_addr)
        print(f"Connected to Device B at {node_b_addr[0]}:{node_b_addr[1]}")

        for step in range(max_new_tokens):
            t0 = time.perf_counter()

            # Device A: 앞단 레이어 실행
            activation = device_a_forward(model, generated, split_layer)

            # seq_len 전송 (Device B의 position_ids 계산에 필요)
            seq_len = generated.shape[1]
            s.sendall(struct.pack(">I", seq_len))

            # Activation 전송
            sent_bytes = send_tensor(s, activation)

            # Device B로부터 다음 토큰 수신
            raw = s.recv(4)
            if len(raw) < 4:
                print("Device B disconnected")
                break
            next_token_id = struct.unpack(">I", raw)[0]

            elapsed = (time.perf_counter() - t0) * 1000
            token_str = tokenizer.decode([next_token_id])
            print(f"[{step+1:3d}] token={next_token_id:6d} '{token_str}' "
                  f"| A={elapsed:.0f}ms | payload={sent_bytes/1024:.1f}KB")

            # EOS 체크
            if next_token_id == tokenizer.eos_token_id:
                print("\n[EOS]")
                break

            # 생성 토큰 추가
            next_id_tensor = torch.tensor([[next_token_id]], device=model.device)
            generated = torch.cat([generated, next_id_tensor], dim=1)

        # 종료 신호 (seq_len=0)
        s.sendall(struct.pack(">I", 0))

    return tokenizer.decode(generated[0, input_ids.shape[1]:], skip_special_tokens=True)


# ── 엔트리포인트 ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--text", default="인공지능이란 무엇인가요?")
    parser.add_argument("--split", type=int, default=16)
    parser.add_argument("--max_tokens", type=int, default=50)
    parser.add_argument("--host", default=NODE_B_HOST)
    parser.add_argument("--port", type=int, default=NODE_B_PORT)
    args = parser.parse_args()

    tokenizer, model = load_model(args.model)

    result = generate(
        tokenizer=tokenizer,
        model=model,
        text=args.text,
        split_layer=args.split,
        max_new_tokens=args.max_tokens,
        node_b_addr=(args.host, args.port),
    )

    print(f"\n=== 생성 결과 ===\n{result}")