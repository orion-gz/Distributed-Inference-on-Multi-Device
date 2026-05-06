"""
=============
transport.py 
=============
phase2: Serialization module for exchanging activation tensors between two processes via TCP.

Protocol Structure: 
    [4B: Total Length] [1B: dtype] [1B: Number of Dimensions] [Number of Dimensions × 4B: Shape] [Actual Data]
    A 4-byte prefix is ​​required at the beginning so that the receiving end knows "how much to read".

encode()/decode(): 
    responsible for tensor↔byte conversion.
    
send_tensor()/recv_tensor(): 
    responsible for socket I/O.
    
_recv_exact(): 
    _recv_exact is a helper function that loops until exactly n bytes are received, because due to the nature of TCP, 
    the desired amount may not be received in a single recv.
"""

import socket
import struct

import numpy as np
import torch

# dtype id ↔ numpy dtype mapping
_DTYPE_TO_ID = {"float16": 0, "bfloat16": 1, "float32": 2}
_ID_TO_DTYPE = {v: k for k, v in _DTYPE_TO_ID.items()}

def encode(tensor: torch.Tensor) -> bytes:
    """tensor → bytes"""
    dtype_str = str(tensor.dtype).replace("torch.", "")
    dtype_id = _DTYPE_TO_ID.get(dtype_str, 2)
    t = tensor.detach().cpu().contiguous()
    
    if tensor.dtype == torch.bfloat16:
        arr = t.view(torch.int16).numpy()
    else:
        arr = t.numpy()

    shape = arr.shape
    header = struct.pack(f">BB{len(shape)}I", dtype_id, len(shape), *shape)
    return header + arr.tobytes()

def decode(data: bytes, device: str = "cuda") -> torch.Tensor:
    """bytes → tensor"""
    dtype_id = struct.unpack(">B", data[0:1])[0]
    ndim = struct.unpack(">B", data[1:2])[0]
    shape = struct.unpack(f">{ndim}I", data[2 : 2 + ndim * 4])
    payload = data[2 + ndim * 4 :]
    original = _ID_TO_DTYPE[dtype_id]

    if original == "bfloat16":
        arr = np.frombuffer(payload, dtype=np.int16).reshape(shape).copy()
        tensor = torch.from_numpy(arr).view(torch.bfloat16).to(device)
    elif original == "float16":
        arr = np.frombuffer(payload, dtype=np.float16).reshape(shape).copy()
        tensor = torch.from_numpy(arr).to(device).half()
    else:
        arr    = np.frombuffer(payload, dtype=np.float32).reshape(shape).copy()
        tensor = torch.from_numpy(arr).to(device)

    return tensor


def send_tensor(sock: socket.socket, tensor: torch.Tensor) -> int:
    """send tensor using socket"""
    data = encode(tensor)
    total = len(data)
    sock.sendall(struct.pack(">I", total))  # 4B 길이 prefix
    sock.sendall(data)
    return total + 4


def recv_tensor(sock: socket.socket, device: str = "cuda") -> torch.Tensor:
    """receive tensor by socket"""
    raw_len = _recv_exact(sock, 4)
    total = struct.unpack(">I", raw_len)[0]
    data = _recv_exact(sock, total)
    return decode(data, device)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Connection closed unexpectedly")
        buf.extend(chunk)
    return bytes(buf)