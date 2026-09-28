# About these notes

This site is a working notebook for systems questions that are easier to understand with diagrams, equations, and concrete tensor layouts.

The recurring subjects are GPU kernel design, low-precision computation, sparse and linear attention, and distributed communication. Most articles begin with a narrow implementation question and then trace it across the layers that matter: mathematical structure, hardware mapping, memory movement, synchronization, and observed performance.

The notes favor explicit assumptions over polished generalities. Diagrams are used to make dataflow and ownership visible; equations are included when they remove ambiguity; implementation details stay close to the claims they support.

Source files and revisions live in the [random-notes repository](https://github.com/icavan/random-notes). Corrections and technical discussion are welcome through GitHub issues.

## Topics

- **GPU kernels** — butterfly transforms, block scaling, tensor-core trade-offs, and launch amortization.
- **Attention systems** — DSA, CSA2, KDA, backward derivations, cache structure, and scheduling.
- **Distributed systems** — CUDA VMM, expert parallelism, GPU-initiated networking, and buffer APIs.

