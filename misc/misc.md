# Relevant github issues

WGSL changes

- Proposal: add WGSL instruction workgroupUniformAtomicLoad
  - https://github.com/gpuweb/gpuweb/issues/5124
- builtin variables should be automatically available inside kernels (WGSL proposed language change)
  - https://github.com/gpuweb/gpuweb/issues/5153
- builtins with a 3D variant should all have 1D variants (global_invocation_index, workgroup_index) (WGSL proposed language change)
  - https://github.com/gpuweb/gpuweb/issues/5154
- allow declaring workgroup memory within function scope (WGSL proposed language change)
  - https://github.com/gpuweb/gpuweb/issues/5155
- function workgroup-memory args should be more flexible (WGSL proposed language change)
  - https://github.com/gpuweb/gpuweb/issues/5156
- Last 4: discussed at 2025-05-13 WGSL meeting

WebGPU Inspector

- https://github.com/brendan-duncan/webgpu_inspector/issues/24

Deno:

- https://github.com/denoland/deno/issues/26769
- https://github.com/gfx-rs/wgpu/pull/7113
- https://github.com/denoland/deno/issues/26766
- https://github.com/denoland/deno/pull/28192

WebGPU Projects

- https://github.com/kishimisu/WebGPU-Radix-Sort
- https://github.com/gpuweb/gpuweb/issues/2229 (atomics concerns)

Other GPU projects

- https://github.com/facebookresearch/dietgpu (ANS)
- Futhark
  - https://futhark-lang.org/blog/2024-07-17-opencl-cuda-hip.html scan
  - https://futhark-lang.org/publications/two-things-robert-did.pdf Parallel Differentiation and Rank Polymorphism
- https://dl.acm.org/doi/10.1145/3318.3478 Optimal parallel generation of a computation tree form
- https://github.com/Snektron/pareas Snektron GPU compiler

---

# Capturing Metal WebGPU Traces

`METAL_CAPTURE_ENABLED=1 DAWN_TRACE_FILE_BASE=/Users/jdowens/Downloads/ xctrace record --output sort-225-250327.trace --template "CPU Profiler" --time-limit 10s --launch -- /Users/jdowens/.nvm/versions/node/v23.0.0/bin/node benchmarking_node.mjs`

Workaround for crashes:

```js
async function main() {
  const navigator = { gpu: create([]) };

  // ... do my webgpu stuff ...

  return WeakRef(navigator);
}

const ref = await main();
while (ref.deref()) {
  console.log("wait for gc of gpu");
  await new Promise((r) => setTimeout(r, 1000));
}

// wait for capture to save
//await new Promise(r => setTimeout(r, 1000));   // not sure this is needed. It worked for me commented out
```

> This is a short coming of dawn.node at the moment in that it doesn't clean up well. It appears the GPU process/thread is killed while the capture is being saved. I'm not sure if waiting for it via this method is robust but for my small test it worked.

---

# Deno in-progress command-line

```
webgpu-benchmarking % ~/Documents/src/deno/target/release/deno run --allow-env --allow-import --allow-ffi standalone.mjs
/Users/runner/work/node-webgpu/node-webgpu/third_party/dawn/out/cmake-release/gen/node/NapiSymbols.h:26: UNREACHABLE: napi_create_string_utf8() napi_create_string_utf8 is a weak stub, and should have been runtime replaced by the node implementation
```

`const isDeno = typeof Deno !== "undefined";`
