---
template: home.html
hide:
  - navigation
  - toc
---

<div class="hero" markdown>
  <p class="hero__eyebrow">ICAVAN / SYSTEMS NOTES</p>
  <h1>Notes from the edge<br>of GPU systems.</h1>
  <p class="hero__lede">Deep dives into GPU kernels, sparse attention, low-precision training, and the communication layers that connect them.</p>
  <div class="hero__actions">
    <a class="md-button md-button--primary" href="FAST_HADAMARD_TRANSFORM_GPU_OPTIMIZATION/">Start with FHT</a>
    <a class="md-button" href="DEEPSEEK_V41_CED_CSA2_ARCHITECTURE/">Explore attention</a>
  </div>
</div>

<p class="section-kicker">FEATURED NOTES</p>

<div class="grid cards" markdown>

-   :material-butterfly:{ .lg .middle } __Engineering a Fast Hadamard Transform__

    ---

    Butterfly factorization, hierarchical GPU communication, and CTA-dispatch amortization on modern NVIDIA GPUs.

    [:octicons-arrow-right-24: Read the note](FAST_HADAMARD_TRANSFORM_GPU_OPTIMIZATION.md)

-   :material-chart-timeline-variant-shimmer:{ .lg .middle } __DeepSeek V4.1: CED and CSA2__

    ---

    A visual explanation of causal encoding, compressed sparse attention, and cross-layer KV reuse.

    [:octicons-arrow-right-24: Read the note](DEEPSEEK_V41_CED_CSA2_ARCHITECTURE.md)

-   :material-memory:{ .lg .middle } __CUDA VMM and Communication Layers__

    ---

    How virtual memory mappings, fabric handles, NCCL, NVSHMEM, and MoonEP fit into one system view.

    [:octicons-arrow-right-24: Read the note](CUDA_VMM_COMMUNICATION_LAYERS.md)

</div>

<div class="topic-intro" markdown>
<p class="section-kicker">BROWSE BY TOPIC</p>
<h2>Three threads, one systems view.</h2>
<p>The notes move between algorithms, kernel implementation, and system-level data movement—the three layers that usually have to be understood together.</p>
</div>

<div class="topic-grid" markdown>
  <div class="topic-card" markdown>
  <span class="topic-card__number">01</span>
  <h3>GPU Kernels</h3>
  <p>FHT butterfly networks, tensor-core trade-offs, block scaling, memory hierarchy, and launch geometry.</p>
  <a href="FAST_HADAMARD_TRANSFORM_GPU_OPTIMIZATION/">View kernel notes →</a>
  </div>

  <div class="topic-card" markdown>
  <span class="topic-card__number">02</span>
  <h3>Attention Systems</h3>
  <p>DSA backward, CSA2, CAKE KDA, sparse selection, tensor shapes, and production optimization paths.</p>
  <a href="dsa-backward-formula-derivation-and-tensor-shapes/">View attention notes →</a>
  </div>

  <div class="topic-card" markdown>
  <span class="topic-card__number">03</span>
  <h3>Distributed Systems</h3>
  <p>CUDA VMM, GPU-initiated networking, expert-parallel communication, and zero-copy buffer layouts.</p>
  <a href="CUDA_VMM_COMMUNICATION_LAYERS/">View systems notes →</a>
  </div>
</div>

<p class="section-kicker latest-kicker">ALL NOTES</p>

| Note | Topic | Focus |
| --- | --- | --- |
| [B200 Locality Domain Probe](B200_LOCALITY_DOMAIN_PROBE.md) | Kernels | Per-SM pointer chasing, topology classification, and domain-aware scheduling |
| [DSA Backward Formula and Tensor Shapes](dsa-backward-formula-derivation-and-tensor-shapes.md) | Attention | Forward-to-backward derivation and layout map |
| [Adjacent-Query KV-Index Overlap](DSA_256K_QPAIR_KV_OVERLAP_OBSERVATIONS.md) | Attention | Measured locality at 256K context |
| [Block-Scaled FP8 and FP4 Quantization](fp8-fp4-weight-quantization.md) | Kernels | Layout, transpose consistency, 1D vs. 2D scaling |
| [CAKE KDA Prefill Highlights](FLASHINFER_CAKE_KDA_PREFILL_HIGHLIGHTS.md) | Attention | Numerical safety, SMEM reuse, ring buffers |
| [Scaling K1 Parallelism for Small B × H](FLASHINFER_CAKE_KDA_SMALL_BH_K1_PARALLELISM.md) | Attention | Owner-helper scheduling and mailboxes |
| [FlashInfer PR #4262 Core Optimizations](FLASHINFER_PR_4262_CORE_OPTIMIZATIONS.md) | Attention | End-to-end optimization summary |
| [MoonEP Public API Visual Guide](MOONEP_PUBLIC_API_VISUAL_GUIDE.md) | Systems | Buffer layouts and lifecycle |
