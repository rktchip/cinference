# buun-llama-cpp — format docs only

- Source: https://github.com/spiritbuun/buun-llama-cpp.git at `a2fd78181`
- Role in this tree: independent cross-check of the trellis layout
  (`ggml/src/llama/llama-hdf5.h`, `ggml-cuda/exl3*.cuh`, format docs).

## Borrowed

- Layout confirmation: `[k/16, n/16, 16K]`, dim0 = k-side, dim1 = n-side;
  suh always k-side `[k]`, svh always n-side `[n]`; verified
  independently on all 409 checkpoint groups.

## Not borrowed

- Everything else — same stack class as HyperQwen (mostly Python, a few
  kernels), no EXL3 C++/CUDA vehicle. Criterion 2 stays satisfied by the
  exllamav3 reference + cuda-exl3 vehicle.
