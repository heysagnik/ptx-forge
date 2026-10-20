# ptx-forge

Parses PTX, builds a CFG, runs liveness and dead-code elimination, and simulates Chaitin-Briggs register
allocation for a given register limit, including spill code. It then compiles the same PTX with the CUDA
driver JIT (or `ptxas`) and compares the prediction with what ends up in the cubin.

<img src="docs/architecture.svg" alt="ptx-forge architecture" width="100%">

## Build

```bash
cmake -S . -B build && cmake --build build -j && ctest --test-dir build
make -j8 && make test        # without CMake
```

Building needs only a C++20 compiler. The CUDA driver library is loaded at run time, so no SDK or GPU is
required; without a GPU, ground truth comes from `ptxas` if it is installed, or from `--cubin=file`.
Tests use GoogleTest when available and a small built-in shim otherwise.

## Usage

```bash
ptx-forge analyze kernel.ptx --maxrregcount=32 --export-cfg=cfg.dot --heatmap --emit-spilled=spilled.ptx
ptx-forge analyze kernel.ptx --ground-truth=ptxas     # auto (default) | jit | ptxas | none
ptx-forge analyze kernel.ptx --cubin=prebuilt.cubin
ptx-forge inspect kernel.cubin
```

Output for `test/kernels/pressure_loop.ptx` (48 loop-carried accumulators) on an RTX 2050:

```
[2] Chaitin-Briggs Register Allocation Simulation (Target Limit: K = 32)
  Allocation Status          : SPILLED (54 peak demand > 32 limit)
  Predicted Registers Used   : 32 physical registers (+1 predicate)
  Spilled by Select          : 22 registers: %f17, %f16, %f15, %f14, ...
  Estimated .local Frame     : 96 bytes (41 spill / 62 fill instructions inserted)
[3] Ground-Truth Verification via CUDA Driver JIT & ELF Cubin Inspection
  ELF Header                 : ELF64, ABI=0x41 (CUDA), Machine=0xBE (EM_CUDA), SM=80
  ptxas Actual Reg Count     : 32 physical registers (via EIATTR_REGCOUNT)
  Prediction vs ptxas        : 32 predicted, 32 actual (+0)
  ptxas Actual Spill Frame   : 160 bytes stack frame (predicted 96)
```

## Design notes

- A predicated write is a conditional definition. It reads the old value and kills nothing, so in
  `mov %r1,5; @%p mov %r1,7; use %r1` the first `mov` stays live.
- Register pressure counts 32-bit slots (`.b64` and `.f64` take two). Predicates are a separate file of 7.
  A source that dies at an instruction can share its register with the result.
- DCE runs mark-and-sweep first, which removes values that only feed themselves around a loop, then a
  liveness sweep, repeated until nothing changes.
- Interference edges join a definition to everything live right after it. Nodes have widths of 1, 2 or 4
  slots and 64-bit values get aligned pairs. Potential spills are picked by
  `sum(10^loop_depth over defs and uses) / degree`. Select is optimistic.
- Spilled registers are replaced by short-lived temporaries with `st.local`/`ld.local` against
  `__spill_depot`, and allocation reruns until it converges.

## What the cubins actually contain (CUDA 13.4, sm_80 and sm_86)

- `EI_OSABI` is `0x41` on current toolchains and `0x33` on older ones. Both are accepted.
- The register count is in the high byte of `sh_info` on `.text.<kernel>`, and also as `EIATTR_REGCOUNT`
  in the global `.nv.info` section, keyed by symbol index. `EIATTR_FRAME_SIZE` and `EIATTR_MIN_STACK_SIZE`
  are there too, not in `.nv.info.<kernel>`.
- The fixtures in `test/data` agree with `cuobjdump --dump-resource-usage`.
- The driver JIT and the toolkit `ptxas` are different builds and can disagree (`gemm_tiled`: 29 vs 30
  registers). The report says which one it used.
- PTX written by `--emit-spilled` assembles with `ptxas`, including spills inside `mma.sync` operands.

## Limits

Only PTX virtual registers are modelled. `ptxas` also schedules, rematerializes, allocates address
registers and has per-instruction minimums (it needs 28 registers for the `mma` in `mma_tile.ptx`), so
predictions are close but not exact: 24 vs 29 for `gemm_tiled`, 10 vs 12 for `vectorAdd`, and exact only
when the cap binds. Spill sizes differ more.

Not handled: spilling predicate or 128-bit registers, indirect branches (`brx.idx`), `call` results written
to registers, and nested `{}` register scopes (flattened, so same-name redeclarations share a register).
`.loc`, `.file` and debug sections are dropped when PTX is re-emitted.
