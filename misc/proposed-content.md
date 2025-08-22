# Proposed content of webgpu-benchmarking repo

- https://jowens.github.io/webgpu-benchmarking/
- Currently unpopulated

Documentation on using the library, which is:

- For each of {reduce, scan, sort}:
  - Standalone example in JS that is the minimum possible to compute a result
  - Configurable example in JS that has pulldowns that shows different examples (e.g., sort ascending vs. descending, different datatypes, different input sizes)
  - More expensive examples that produce results plots
  - Also screenshots of Apple M3 plots
  - Text descriptions of algorithms with citations (brief)

Documentation for writing a new Primitive

Documentation on structure of my framework

Articles on:

- How I compute timing and what’s good and bad about it
- WebGPU object caching strategy
- What I do with string pasting vs. what I can do at runtime
- Array sizing
- I would like to specialize on hardware, what are the issues with that
- Strategy for supporting non-subgroup-capable devices
- Challenges with not knowing subgroup size
- Supporting task parallelism
- Supporting a single WebGPU program in Chrome/node/deno
- My “Buffer” JS object and what it needs to do
- My WGSL requests and why (“John Writes a Workgroup Reduce”)
- Forward progress concerns/issues
