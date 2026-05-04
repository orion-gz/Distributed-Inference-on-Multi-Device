"""
===================
split_inference.py
===================
phase0: validation for "Does running with split layers yield the same results as running on a single device?".

run_baseline(): 
    executes the entire forward by passing 'output_hidden_states=True'.
    If this option is provided, it receives a list of the output hidden states of all layers.
    'hidden_states[k]' is the output of layer 'k-1' (i.e. the input of layer k),
    'hidden_states[24]' becomes the tensor that Device A will send to B.
    
run_device_b():
    using the hook injection method, insert 'register_forward_pre_hook' just before 'layers[split_layer]' to replace the hidden_states with the captured activation,
    and let the model handle position_embeddings, attention_mask, causal mask, etc. as usual.
"""
import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

DEFAULT_MODEL = "K-intelligence/Midm-2.0-Base-Instruct"
FALLBACK_MODEL = "Qwen/Qwen2-1.5B"
TEST_TEXT = "인공지능이란 무엇인가요?"

def load_model(model_id: str):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.eval()
    return tokenizer, model

def run_baseline(model, input_ids):
    captured = {}

    def _capture_pe(module, args, kwargs):
        pe = kwargs.get("position_embeddings")
        if pe is not None and "pe" not in captured:
            captured["pe"] = pe  

    hook = model.model.layers[0].register_forward_pre_hook(
        _capture_pe, with_kwargs=True
    )

    with torch.no_grad():
        outputs = model.model(
            input_ids,
            output_hidden_states=True,
            use_cache=False,
        )

    hook.remove()

    logits = model.lm_head(model.model.norm(outputs.last_hidden_state))
    return logits, outputs.hidden_states, captured.get("pe")

def run_device_b(model, activation: torch.Tensor,
                 split_layer: int, input_ids: torch.Tensor) -> torch.Tensor:
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
            outputs = model.model(input_ids, use_cache=False)
        logits = model.lm_head(model.model.norm(outputs.last_hidden_state))
    finally:
        handle.remove()
 
    return logits

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--split", type=int, default=None)
    args = parser.parse_args()

    print(f"model loading: {args.model}")
    try:
        tokenizer, model = load_model(args.model)
    except Exception as e:
        print(f"load fail: {e}")
        print(f"using fallback model: {FALLBACK_MODEL}")
        tokenizer, model = load_model(FALLBACK_MODEL)

    total_layers = len(model.model.layers)
    hidden_dim = model.config.hidden_size
    split_layer = args.split if args.split else total_layers // 2

    print(f"\n{'='*50}")
    print(f"total layer number:  {total_layers}")
    print(f"hidden_dim:   {hidden_dim}")
    print(f"split layer:    Layer {split_layer} ({split_layer}/{total_layers})")
    print(f"Device A:     Layer 0 ~ {split_layer - 1}")
    print(f"Device B:     Layer {split_layer} ~ {total_layers - 1}")
    print(f"{'='*50}\n")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(TEST_TEXT, return_tensors="pt").to(model.device)
    input_ids = inputs.input_ids
    seq_len = input_ids.shape[1]
    print(f"Input: '{TEST_TEXT}' ({seq_len} tokens)")

    t0 = time.perf_counter()
    logits_baseline, all_hidden, position_embeddings = run_baseline(model, input_ids)
    t_baseline = (time.perf_counter() - t0) * 1000
    token_baseline = tokenizer.decode(logits_baseline[0, -1].argmax())

    if position_embeddings is not None:
        cos, sin = position_embeddings
        print(f"position_embeddings capture: cos={tuple(cos.shape)}, sin={tuple(sin.shape)}")
    else:
        print("no position_embeddings")

    activation = all_hidden[split_layer]
    payload_bytes = activation.element_size() * activation.nelement()
    payload_kb = payload_bytes / 1024
    wifi_ms = payload_bytes * 8 / (100e6) * 1000

    print(f"\nActivation shape:   {tuple(activation.shape)}")
    print(f"Payload per token:  {payload_kb / seq_len:.1f} KB")
    print(f"Total payload:      {payload_kb:.1f} KB ({seq_len} tokens)")
    print(f"Expected WiFi Transmit:     {wifi_ms:.2f} ms (100Mbps)")

    t0 = time.perf_counter()
    logits_split = run_device_b(model, activation, split_layer, input_ids)
    t_split = (time.perf_counter() - t0) * 1000
    token_split = tokenizer.decode(logits_split[0, -1].argmax())

    max_diff = (logits_baseline.float() - logits_split.float()).abs().max().item()
    ok = max_diff < 0.01

    print(f"\n{'='*50}")
    print(f"baseline inference:  '{token_baseline}'  ({t_baseline:.0f}ms)")
    print(f"split inference:     '{token_split}'  ({t_split:.0f}ms)")
    print(f"maximum logit error: {max_diff:.8f}  {'ok' if ok else 'check needed'}")
    print(f"{'='*50}")

    if not ok:
        print("\n[디버그] 레이어 시그니처 확인:")
        import inspect
        sig = inspect.signature(model.model.layers[split_layer].forward)
        print(f"  layers[{split_layer}].forward 파라미터: {list(sig.parameters.keys())}")


if __name__ == "__main__":
    main()