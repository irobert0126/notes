# Maglev zero-bubble I/W split under torch.compile: before vs after D124884069 (GPU)

The unit test's single-stage Rail pass (`test_split_backward_compile`'s model and stage path), scaled up and run on one NVIDIA GB200.

- 2 Maglev layers, 1 stage, batch 16384 per pass = 2 microbatches of 8192, layer dim 4096, 256 float features, embedding dim 64, fp32 (no TF32)
- I = `dense_backward_act_micro` (`stage_backward_input`), W = `dense_backward_weight_micro` (`stage_backward_weight`)
- before = plain `torch.compile` (Inductor) on each dense half; after = `torch.compile(..., backend=SplitBackwardBackend())` (Inductor); eager = no compile
- 3 warm-up passes (compile excluded), then 10 timed passes, then 1 profiled pass

## Table 1 -- op counts per pass (I and W summed over both microbatches)

| phase | | eager | before fix | after fix |
|---|---|---|---|---|
| I | matmuls (aten) | 8 | 20 | 8 |
| I | matmul GFLOP (aten) | 1133.9 | 2336.5 | 1133.9 |
| I | compute ops (aten) | 22 | 60 | 28 |
| I | GPU kernels (Inductor) | 20 | 62 | 20 |
| W | matmuls (aten) | 12 | 20 | 12 |
| W | matmul GFLOP (aten) | 1202.6 | 2336.5 | 1202.6 |
| W | compute ops (aten) | 60 | 82 | 60 |
| W | GPU kernels (Inductor) | 64 | 86 | 70 |

Before the fix, I and W each run the whole backward (20 matmuls, 2336.5 GFLOP each). After the fix the matmul work splits exactly like eager: I 1133.9 + W 1202.6 = 2336.5 GFLOP, one backward's worth.

## Table 2 -- GPU latency per pass (CUDA events around each phase, median of 10 passes)

| | eager | before fix | after fix |
|---|---|---|---|
| I | 17.34 ms | 36.71 ms | 17.06 ms |
| W | 20.32 ms | 36.89 ms | 19.93 ms |
| **I+W** | **37.66 ms** | **73.60 ms** | **37.00 ms** |

The fix halves I+W (73.60 -> 37.00 ms, -49.7%) and lands at eager (37.66 ms). GPU kernel busy time agrees: I 17.35 / 36.58 / 17.14 ms and W 20.17 / 36.72 / 19.79 ms (eager / before / after).

## Op counts by op, per pass (eager / before / after, aten level)

I:

| op | eager | before | after |
|---|---|---|---|
| `mm[8192x4096 @ 4096x4096]` (dX, block) | 4 | 4 | 4 |
| `mm[8192x4096 @ 4096x128]` (dX, to seam) | 4 | 4 | 4 |
| `mm[4096x8192 @ 8192x4096]` (dW, block) | 0 | 4 | 0 |
| `mm[4096x8192 @ 8192x256]` (dW, float_proj) | 0 | 4 | 0 |
| `mm[4096x8192 @ 8192x128]` (dW, input_proj) | 0 | 4 | 0 |
| `sum` (bias grads) | 0 | 16 | 0 |
| `mul` | 4 | 8 | 4 |
| `threshold_backward` | 4 | 4 | 4 |
| `add` | 0 | 4 | 4 |
| `copy_` | 0 | 2 | 2 |
| `mse_loss_backward`, `fill_`, `zero_` | 2, 2, 2 | 2, 2, 2 | 2, 2, 2 |

W:

| op | eager | before | after |
|---|---|---|---|
| `mm[4096x8192 @ 8192x4096]` (dW, block) | 4 | 4 | 4 |
| `mm[4096x8192 @ 8192x256]` (dW, float_proj) | 4 | 4 | 4 |
| `mm[4096x8192 @ 8192x128]` (dW, input_proj) | 4 | 4 | 4 |
| `mm[8192x4096 @ 4096x4096]` (dX, recomputed) | 0 | 4 | 0 |
| `mm[8192x4096 @ 4096x128]` (dX, recomputed) | 0 | 4 | 0 |
| `threshold_backward` (recomputed) | 0 | 4 | 0 |
| `mul` | 4 | 8 | 4 |
| `sum` | 16 | 16 | 16 |
| `add_` (grad accumulation) | 28 | 28 | 28 |
| `add`, `copy_` | 0, 0 | 4, 2 | 0, 0 |

Before the fix, I also computes every weight gradient and throws it away, and W recomputes every input gradient. After the fix, W matches eager op for op; I matches eager's matmuls, `mul` and `threshold_backward`, plus 4 small `add` and 2 `copy_` from the traced graph that the before arm has too.

## Traces (GPU, profiled pass)

- eager: [Perfetto](https://www.internalfb.com/intern/kernelhub/perfetto?trace_path=tree/permanent_traces/DIFF/D124884069/gpu_eager.json.gz&bucket=torchrec_benchmark_traces) · [Perf Doctor](https://www.internalfb.com/intern/perfdoctor/trace_view?filepath=tree/permanent_traces/DIFF/D124884069/gpu_eager.json.gz&bucket=torchrec_benchmark_traces)
- before fix (`torch.compile`): [Perfetto](https://www.internalfb.com/intern/kernelhub/perfetto?trace_path=tree/permanent_traces/DIFF/D124884069/gpu_before_fix_torch_compile_inductor.json.gz&bucket=torchrec_benchmark_traces) · [Perf Doctor](https://www.internalfb.com/intern/perfdoctor/trace_view?filepath=tree/permanent_traces/DIFF/D124884069/gpu_before_fix_torch_compile_inductor.json.gz&bucket=torchrec_benchmark_traces)
- after fix (`SplitBackwardBackend`): [Perfetto](https://www.internalfb.com/intern/kernelhub/perfetto?trace_path=tree/permanent_traces/DIFF/D124884069/gpu_after_fix_split_backward_inductor.json.gz&bucket=torchrec_benchmark_traces) · [Perf Doctor](https://www.internalfb.com/intern/perfdoctor/trace_view?filepath=tree/permanent_traces/DIFF/D124884069/gpu_after_fix_split_backward_inductor.json.gz&bucket=torchrec_benchmark_traces)
- [manifold folder](https://www.internalfb.com/manifold/explorer/torchrec_benchmark_traces/tree/permanent_traces/DIFF/D124884069)

In each trace, search for `## dense_backward_act mb` (I) and `## dense_backward_weight mb` (W). Before the fix both scopes contain a `CompiledFunctionBackward` running the whole backward; after the fix they contain `_SplitBackwardFunctionBackward` running the dI graph and the dW graph respectively.

## Method notes

- Aten-level op counts (matmuls, GFLOP, compute ops) come from arms whose graphs run op by op: before = AOTAutograd with the same min-cut partitioner, after = `SplitBackwardBackend(eager_compiler)`. That makes every op visible and alignable with eager. GPU kernel counts and latency come from the Inductor arms.
- Compute ops exclude view/allocation/metadata ops and autograd node wrappers; a matmul counts as one op. Ops are attributed to a phase by time window across all threads (autograd runs CUDA backwards on its device thread).
- Inductor's extern-matmul shapes do not always record cleanly in the profiler, so GFLOP is reported from the aten-level arms; Inductor kernel busy time matches them.
- Single GPU, single stage: this measures each half's compute, not the pipeline bubble.
- Same pass on CPU at batch 2048 / dim 1024 gives the same op counts in proportion; at the unit test's toy shapes (4x6 matrices) the after-fix CPU time was ~40% higher, since that run is pure host overhead.
