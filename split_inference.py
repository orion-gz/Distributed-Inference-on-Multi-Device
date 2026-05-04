"""
Phase 0: 레이어 분할 추론 검증

목표:
  1. Mi:dm 2.0 로드 확인
  2. 아키텍처 정보 출력 (레이어 수, hidden_dim)
  3. 단일 기기 vs 분할 추론 출력이 동일한지 검증
  4. Activation payload 크기 및 전송 시간 측정

실행:
  python split_inference.py
  python split_inference.py --model Qwen/Qwen2-1.5B  # 소형 모델로 먼저 테스트
"""

import argparse
import inspect
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "K-intelligence/Midm-2.0-Base-Instruct"  # HuggingFace ID 확인 필요
FALLBACK_MODEL = "Qwen/Qwen2-1.5B"          # 접근 불가 시 폴백
TEST_TEXT = "인공지능이란 무엇인가요?"

def build_causal_mask(seq_len: int, dtype: torch.dtype, device) -> torch.Tensor:
    """4D causal attention mask (일부 레이어 직접 호출 시 필요)"""
    mask = torch.full((seq_len, seq_len), float("-inf"), dtype=dtype, device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, seq]


def call_layer(layer, hidden, position_ids, causal_mask):
    """transformers 버전별 레이어 호출 인터페이스 통일"""
    sig = inspect.signature(layer.forward)
    kwargs = dict(position_ids=position_ids, use_cache=False)

    # transformers 4.43+ 는 cache_position 사용
    if "cache_position" in sig.parameters:
        kwargs["cache_position"] = torch.arange(
            position_ids.shape[-1], device=hidden.device
        )
    else:
        kwargs["attention_mask"] = causal_mask

    out = layer(hidden, **kwargs)
    return out[0]  # hidden_states

def run_baseline(model, input_ids) -> tuple[torch.Tensor, list]:
    """단일 기기 전체 추론 — 정답 기준값 생성"""
    with torch.no_grad():
        outputs = model.model(
            input_ids,
            output_hidden_states=True,
            use_cache=False,
        )
    logits = model.lm_head(model.model.norm(outputs.last_hidden_state))
    return logits, outputs.hidden_states


def run_device_b(model, activation: torch.Tensor, split_layer: int) -> torch.Tensor:
    """
    Device B 시뮬레이션: split_layer 이후 레이어만 실행.
    실제 구현에서는 activation을 소켓으로 수신하는 부분으로 교체됩니다.
    """
    device = activation.device
    seq_len = activation.shape[1]
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    causal_mask = build_causal_mask(seq_len, activation.dtype, device)

    h = activation.clone()
    with torch.no_grad():
        for i in range(split_layer, len(model.model.layers)):
            h = call_layer(model.model.layers[i], h, position_ids, causal_mask)
        h = model.model.norm(h)
        logits = model.lm_head(h)
    return logits

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--split", type=int, default=None,
                        help="분할 레이어 인덱스 (기본: 전체 절반)")
    args = parser.parse_args()

    print(f"모델 로드 중: {args.model}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            device_map="auto",
        )
    except Exception as e:
        print(f"로드 실패: {e}")
        print(f"폴백 모델 사용: {FALLBACK_MODEL}")
        tokenizer = AutoTokenizer.from_pretrained(FALLBACK_MODEL)
        model = AutoModelForCausalLM.from_pretrained(
            FALLBACK_MODEL,
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            device_map="auto",
        )

    model.eval()

    # 2. 아키텍처 정보 출력
    total_layers = len(model.model.layers)
    hidden_dim = model.config.hidden_size
    split_layer = args.split if args.split else total_layers // 2

    print(f"\n{'='*50}")
    print(f"총 레이어 수:  {total_layers}")
    print(f"hidden_dim:   {hidden_dim}")
    print(f"분할 지점:    Layer {split_layer} ({split_layer}/{total_layers})")
    print(f"Device A:     Layer 0 ~ {split_layer - 1}")
    print(f"Device B:     Layer {split_layer} ~ {total_layers - 1}")
    print(f"{'='*50}\n")

    # 3. 입력 준비
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(TEST_TEXT, return_tensors="pt").to(model.device)
    input_ids = inputs.input_ids
    seq_len = input_ids.shape[1]
    print(f"입력: '{TEST_TEXT}' ({seq_len} tokens)")

    # 4. Baseline 추론
    t0 = time.perf_counter()
    logits_baseline, all_hidden = run_baseline(model, input_ids)
    t_baseline = (time.perf_counter() - t0) * 1000
    token_baseline = tokenizer.decode(logits_baseline[0, -1].argmax())

    # 5. Activation payload 분석
    # hidden_states[k] = output after layer k-1
    # → hidden_states[split_layer] = Device A가 Device B로 전송할 텐서
    activation = all_hidden[split_layer]
    payload_bytes = activation.element_size() * activation.nelement()
    payload_kb = payload_bytes / 1024
    wifi_ms = payload_bytes * 8 / (100e6) * 1000  # 100Mbps 기준

    print(f"\nActivation shape:   {tuple(activation.shape)}")
    print(f"Payload per token:  {payload_kb / seq_len:.1f} KB")
    print(f"Total payload:      {payload_kb:.1f} KB ({seq_len} tokens)")
    print(f"예상 WiFi 전송:     {wifi_ms:.2f} ms (100Mbps 기준)")

    # 6. Split 추론 검증
    t0 = time.perf_counter()
    logits_split = run_device_b(model, activation, split_layer)
    t_split = (time.perf_counter() - t0) * 1000
    token_split = tokenizer.decode(logits_split[0, -1].argmax())

    max_diff = (logits_baseline.float() - logits_split.float()).abs().max().item()
    ok = max_diff < 0.01

    print(f"\n{'='*50}")
    print(f"baseline 추론:  '{token_baseline}'  ({t_baseline:.0f}ms)")
    print(f"split 추론:     '{token_split}'  ({t_split:.0f}ms)")
    print(f"최대 logit 오차: {max_diff:.8f}  {'✓ 정확' if ok else '✗ 확인 필요'}")
    print(f"{'='*50}")

    if not ok:
        print("\n[주의] 오차가 큽니다. 원인:")
        print("  - attention mask 포맷이 모델 버전과 다를 수 있음")
        print("  - call_layer() 내 kwargs 조정 필요")
        print("  - 모델의 forward() 시그니처 직접 확인: model.model.layers[0].forward.__doc__")


if __name__ == "__main__":
    main()