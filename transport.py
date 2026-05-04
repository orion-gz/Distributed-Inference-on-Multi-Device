"""
transport.py — Activation tensor TCP 전송 모듈

프로토콜 (헤더 + 데이터):
  [4B: total_len] [1B: dtype_id] [1B: ndim] [ndim×4B: shape] [data]

Phase 3 (Android)에서 동일 프로토콜로 WebRTC DataChannel 사용 예정.
"""

import socket
import struct

import numpy as np
import torch

# dtype id ↔ numpy dtype 매핑
_DTYPE_TO_ID = {"float16": 0, "bfloat16": 1, "float32": 2}
_ID_TO_DTYPE = {v: k for k, v in _DTYPE_TO_ID.items()}
_ID_TO_NP = {0: np.float16, 1: np.float32, 2: np.float32}  # bfloat16 → fp32 for numpy


def encode(tensor: torch.Tensor) -> bytes:
    """텐서 → bytes 직렬화"""
    dtype_str = str(tensor.dtype).replace("torch.", "")
    dtype_id = _DTYPE_TO_ID.get(dtype_str, 2)

    # bfloat16은 numpy 미지원 → float32 변환
    arr = tensor.detach().cpu()
    if tensor.dtype == torch.bfloat16:
        arr = arr.float()
    arr = arr.numpy()

    shape = arr.shape
    header = struct.pack(f">BB{len(shape)}I", dtype_id, len(shape), *shape)
    return header + arr.tobytes()


def decode(data: bytes, device: str = "cuda") -> torch.Tensor:
    """bytes → 텐서 역직렬화"""
    dtype_id = struct.unpack(">B", data[0:1])[0]
    ndim = struct.unpack(">B", data[1:2])[0]
    shape = struct.unpack(f">{ndim}I", data[2 : 2 + ndim * 4])
    payload = data[2 + ndim * 4 :]

    np_dtype = _ID_TO_NP[dtype_id]
    arr = np.frombuffer(payload, dtype=np_dtype).reshape(shape).copy()
    tensor = torch.from_numpy(arr).to(device)

    # 원래 dtype 복원
    original = _ID_TO_DTYPE[dtype_id]
    if original == "float16":
        tensor = tensor.half()
    elif original == "bfloat16":
        tensor = tensor.bfloat16()

    return tensor


def send_tensor(sock: socket.socket, tensor: torch.Tensor) -> int:
    """소켓으로 텐서 전송, 전송 바이트 수 반환"""
    data = encode(tensor)
    total = len(data)
    sock.sendall(struct.pack(">I", total))  # 4B 길이 prefix
    sock.sendall(data)
    return total + 4


def recv_tensor(sock: socket.socket, device: str = "cuda") -> torch.Tensor:
    """소켓에서 텐서 수신"""
    raw_len = _recv_exact(sock, 4)
    total = struct.unpack(">I", raw_len)[0]
    data = _recv_exact(sock, total)
    return decode(data, device)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    """정확히 n바이트 수신"""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Connection closed unexpectedly")
        buf.extend(chunk)
    return bytes(buf)