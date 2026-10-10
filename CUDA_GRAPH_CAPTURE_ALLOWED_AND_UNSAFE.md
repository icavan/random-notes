# CUDA Graph Capture: Allowed and Unsafe Patterns

CUDA Graph replay requires the same kernels, memory addresses, tensor shapes,
and control flow on every iteration. A graph-private memory pool makes many
temporary CUDA tensors capturable, but it does not make arbitrary dynamic
behavior safe.

## Usually allowed

| Pattern | Requirement |
| --- | --- |
| `empty_like`, `zeros`, and other temporary CUDA tensors | Shape, dtype, device, and allocation order stay fixed |
| Static padding and contiguous copies | Output shapes are known during capture |
| GEMM, attention, and library kernels | Compilation, autotuning, plans, and handles are warmed up first |
| CUDA RNG operations | Graph-safe RNG state is used |
| Caller-provided workspace | Address and size remain valid for every replay |
| Multi-stream execution | Dependencies are represented by captured streams and events |

## Not allowed or unsafe

- GPU-to-CPU synchronization such as `item()` or `synchronize()` during capture.
- Data-dependent tensor shapes or Python branches.
- Raw synchronous `cudaMalloc` or `cudaFree` inside custom operators.
- Lazy JIT compilation, autotuning, plan creation, or handle initialization.
- CPU or pinned-memory allocation that is required by the captured path.
- Uncaptured CUDA work that interferes with an active capture.
- Replaying graphs concurrently when they overwrite the same private-pool storage.

## Temporary tensors are capturable, not free

A temporary tensor created during capture receives a fixed address from the
graph-private pool. Replay reuses that address instead of allocating a new
tensor. The GPU operations that produce and consume the tensor still execute.

For example, caller-side H32-to-H64 padding can be captured:

```text
Capture once:
  q64 = pad(q32)  -> fixed address P1
  w64 = pad(w32)  -> fixed address P2

Replay:
  run both padding kernels
  write q64 and w64 to P1 and P2
  run the H64 compute kernel
```

CUDA Graph removes repeated Python dispatch and allocation management, but it
does not remove the padding kernels, memory traffic, persistent H64 storage, or
optional H64-to-H32 output copies. A native compact-H32 kernel therefore still
has a real advantage under graph replay.

## Practical capture checklist

1. Use static input and output buffers with long-lived addresses.
2. Warm up compilation, plans, handles, and allocator paths on a side stream.
3. Preallocate outputs and workspace when practical.
4. Keep shapes, strides, dtypes, and execution order fixed.
5. Reset any input that the captured operation overwrites in place.
6. Update static input buffers, replay the graph, and read the static outputs.

References: [PyTorch CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs) and the [CUDA Graphs programming guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html).
