import { range, arrayProd, datatypeToTypedArray, datatypeToBytes } from "./util.mjs";
import {
    BasePrimitive,
    Kernel,
    // InitializeMemoryBlock,
    AllocateBuffer,
} from "./primitive.mjs";
import { BaseTestSuite } from "./testsuite.mjs";
import { BinOpAddU32, BinOpAddF32 } from "./binop.mjs";



export class BaseHistogram extends BasePrimitive {
    constructor(args) {
        super(args);

        // Required parameters (like your BaseScan)
        for (const required of ["datatype", "numBins"]) {
            if (!this[required]) {
                throw new Error(`${this.constructor.name}: ${required} is required`);
            }
        }

        // Histogram always outputs u32 (counts), regardless of input datatype
        // The binop should always be u32 addition
        this.binop = args.binop ?? new BinOpAddU32();

        if (this.binop.datatype !== "u32") {
            throw new Error(
                `${this.constructor.name}: binop datatype must be u32 (histogram output is always u32), but got ${this.binop.datatype}.`
            );
        }
        /*default bin range*/
        this.minValue = args.minValue ?? 0.0;
        this.maxValue = args.maxValue ?? 1.0;


        this.knownBuffers = ["inputBuffer", "outputBuffer"];

        for (const knownBuffer of this.knownBuffers) {
            /* we passed an existing buffer into the constructor */
            if (knownBuffer in args) {
                this.registerBuffer({ label: knownBuffer, buffer: args[knownBuffer] });
                delete this[knownBuffer]; // let's make sure it's in one place only
            }
        }
        /* by default, delegate to simple call from BasePrimitive */
        this.getDispatchGeometry = this.getSimpleDispatchGeometry;
    }

    get bytesTransferred() {
        return (
            this.getBuffer("inputBuffer").size + this.getBuffer("outputBuffer").size
        );
    }

    validate = (args = {}) => {
        const memsrc = args.inputBuffer ?? this.getBuffer("inputBuffer").cpuBuffer;
        const memdest = args.outputBuffer ?? this.getBuffer("outputBuffer").cpuBuffer;
        let referenceOutput;
        try {
            // Histogram output is ALWAYS u32 (counts), not the input datatype
            referenceOutput = new Uint32Array(this.numBins);
        } catch (error) {
            console.error(error, "Tried to allocate array of length", this.numBins);
        }
        /*Intialize histogram bins*/
        for (let bin = 0; bin < this.numBins; bin++) {
            referenceOutput[bin] = 0;
        }
        //build reference for histogram
        for (let i = 0; i < memsrc.length; i++) {
            const value = memsrc[i];

            //map value to bin's index
            const normalized = (value - this.minValue) / (this.maxValue - this.minValue);
            let binIndex = Math.floor(normalized * this.numBins);
            //clamp
            binIndex = Math.max(0, Math.min(binIndex, this.numBins - 1));
            referenceOutput[binIndex] = referenceOutput[binIndex] + 1;
        }
        function validates(args) {
            return args.cpu == args.gpu;
        }

        let returnString = "";
        let allowedErrors = 5;

        for (let bin = 0; bin < memdest.length; bin++) {
            if (allowedErrors == 0) {
                break;
            }
            if (
                !validates({
                    cpu: referenceOutput[bin],
                    gpu: memdest[bin],
                    datatype: this.datatype,
                })
            ) {
                const ref = referenceOutput[bin];
                const gpu = memdest[bin];
                const diff = ref === 0 ? Math.abs(gpu - ref) : Math.abs((ref - gpu) / ref);
                returnString += `\nBin ${bin}: expected ${ref}, instead saw ${gpu} (diff: ${diff}).`;
                if (this.getBuffer("debugBuffer")) {
                    returnString += ` debug[${bin}] = ${this.getBuffer("debugBuffer").cpuBuffer[bin]}.`;
                }
                if (this.getBuffer("debug2Buffer")) {
                    returnString += ` debug2[${bin}] = ${this.getBuffer("debug2Buffer").cpuBuffer[bin]}.`;
                }
                allowedErrors--;
            }
        }

        // ALWAYS log validation output, just like scan/reduce do
        console.log(
            this.label,
            "histogram",
            "with input",
            memsrc,
            "should validate to",
            referenceOutput,
            "and actually validates to",
            memdest,
            this.getBuffer("debugBuffer") ? "\ndebugBuffer" : "",
            this.getBuffer("debugBuffer")
                ? this.getBuffer("debugBuffer").cpuBuffer
                : "",
            this.getBuffer("debug2Buffer") ? "\ndebug2Buffer" : "",
            this.getBuffer("debug2Buffer")
                ? this.getBuffer("debug2Buffer").cpuBuffer
                : "",
            this.binop.constructor.name,
            this.binop.datatype,
            "identity is",
            this.binop.identity,
            "bins:",
            this.numBins,
            "range:",
            this.minValue,
            "to",
            this.maxValue,
            "input length:",
            memsrc.length
        );

        return returnString;
    };
}

export const histogramBandwidthPlot = {
    x: { field: "inputBytes", label: "Input array size (B)" },
    y: { field: "bandwidth", label: "Achieved bandwidth (GB/s)" },
    stroke: { field: "timing" },  // Lines colored by GPU vs CPU
    text_br: "gpuinfo.description",
    caption: "Histogram Bandwidth (GPU vs CPU)",
};

/* needs to be a function if we do string interpolation */
// eslint-disable-next-line no-unused-vars
function histogramWGCountFnPlot() {
    return {
        x: { field: "inputBytes", label: "Input array size (B)" },
        /* y.field is just showing off that we can have an embedded function */
        /* { field: "bandwidth", ...} would do the same */
        y: { field: (d) => d.bandwidth, label: "Achieved bandwidth (GB/s)" },
        stroke: { field: "workgroupCount" },
        text_br: (d) => `${d.gpuinfo.description}`,
        caption: `${this.category} | ${this.testSuite} | Lines are workgroup count`,
    };
}

// eslint-disable-next-line no-unused-vars
const histogramWGSizeBinOpPlot = {
    x: { field: "inputBytes", label: "Input array size (B)" },
    y: { field: "bandwidth", label: "Achieved bandwidth (GB/s)" },
    fy: { field: "binop" },
    stroke: { field: "workgroupSize" },
    text_br: "gpuinfo.description",
    caption: "Lines are workgroup size",
};

//https://developer.nvidia.com/blog/gpu-pro-tip-fast-histograms-using-shared-atomics-maxwell/

/*Step 1: clear private bins (like setting to 0).
Step 2: each thread counts elements into its workgroup’s private histogram. 
Step 3: merge your private histogram into the final global result. */

export class WGHistogram extends BaseHistogram {
    constructor(args) {
        super(args);
    }
    finalizeRuntimeParameters() {
        /* tunable parameters*/
        this.workgroupSize = this.workgroupSize ?? 256;
        this.maxGSLWorkgroupCount = this.maxGSLWorkgroupCount ?? 256;

        /*Compute settings based on tunable parameters */
        this.workgroupCount = Math.min(Math.ceil(this.getBuffer("inputBuffer").size / this.workgroupSize), this.maxGSLWorkgroupCount);
        this.numPartials = this.workgroupCount;

        // CREATE THE UNIFORM DATA
        const inputLength = this.getBuffer("inputBuffer").size / datatypeToBytes(this.datatype);
        this.histogramUniforms = new Uint32Array([
            inputLength,
            this.numBins,
            this.workgroupCount, // Add workgroup count
            0 // padding for alignment
        ]);
        const histogramUniformsFloat = new Float32Array([
            this.minValue,
            this.maxValue
        ]);

        // Create 32-byte buffer with proper padding
        const ub = new ArrayBuffer(32);
        const dv = new DataView(ub);

        dv.setUint32(0, inputLength, true);           // offset 0
        dv.setUint32(4, this.numBins, true);          // offset 4
        dv.setUint32(8, this.workgroupCount, true);   // offset 8
        dv.setUint32(12, 0, true);                    // padding0 at 12

        dv.setFloat32(16, this.minValue, true);       // offset 16
        dv.setFloat32(20, this.maxValue, true);       // offset 20
        // bytes 24..31 left as zero padding (padding1, padding2)

        this.histogramUniformsBuffer = new Uint8Array(ub);

    }
    histogramKernelDefinition = () => {
        return /*wgsl*/ `
    // input buffer
    @group(0) @binding(0) var<storage, read> inputBuffer: array<${this.datatype}>;
    //output buffer
    @group(0) @binding(1) var<storage,read_write>outputBuffer:array<atomic<u32>>;
    
    struct HistogramUniforms{
        inputLength:u32,
        numBins:u32,
        numWorkgroups:u32,
        padding0:u32,
        minValue:f32,
        maxValue:f32,
        padding1:f32,
        padding2:f32,
    }
    @group(0) @binding(2) var<uniform> uniforms:HistogramUniforms;

    //privatized workgroup
    var<workgroup>privateHistogram:array<atomic<u32>,${this.numBins}>;

    @compute @workgroup_size(${this.workgroupSize})
    fn main(
        @builtin(global_invocation_id)globalId:vec3<u32>,
        @builtin(local_invocation_id)localId:vec3<u32>,
        @builtin(workgroup_id)wg_id:vec3<u32>
    ) {

        let gwIndex:u32 =globalId.x;
        let localIndex:u32=localId.x;
        let wgIndex:u32=wg_id.x;

        //zero private histograms
        var i:u32=localIndex;
        let WGS: u32=${this.workgroupSize}u;
        let NB:u32=uniforms.numBins;

        loop{
            if(i>=NB){break;}
            atomicExchange(&privateHistogram[i],0u);
            i=i+WGS;
        }
        workgroupBarrier();

        // Each thread processes a strided subset of input elements:
        var idx: u32 = gwIndex;
    let inputLen: u32 = uniforms.inputLength;
    while (idx < inputLen) {
      // read input value
      let value: ${this.datatype} = inputBuffer[idx];

      // compute bin index (map to 0..numBins-1) with divide-by-zero protection
      let range: f32 = uniforms.maxValue - uniforms.minValue;
      var normalized: f32 = select((f32(value) - uniforms.minValue) / range, 0.0, range == 0.0);
      normalized = clamp(normalized, 0.0, 0.999999); // keep in [0,1) to avoid out-of-range
      var binIndex: i32 = i32(floor(normalized * f32(NB)));

      //clampping to be in range of bins
      if (binIndex < 0) { binIndex = 0; }
      if (binIndex >= i32(NB)) { binIndex = i32(NB) - 1; }

      // increment private histogram
      atomicAdd(&privateHistogram[u32(binIndex)], 1u);

    // stride by total threads (workgroups * WG size)
      idx = idx + uniforms.numWorkgroups * WGS; 
    }

    workgroupBarrier();

    // Now each thread will merge a subset of bins into the global outputBuffer
    // Partition bins across threads to parallelize the merge
    var b: u32 = localIndex;
    while (b < NB) {
      let partialCount: u32 = atomicLoad(&privateHistogram[b]); // read local atomic
      if (partialCount > 0u) {
        atomicAdd(&outputBuffer[b], partialCount);
      }
      b = b + WGS;
    }
    }
    `;
    }

    compute() {
        this.finalizeRuntimeParameters();

        return [
            new AllocateBuffer({
                label: "histogramUniforms",
                size: this.histogramUniformsBuffer.byteLength,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
                populateWith: this.histogramUniformsBuffer,
            }),

            new Kernel({
                kernel: this.histogramKernelDefinition,
                bufferTypes: [["read-only-storage", "storage", "uniform"]],
                bindings: [["inputBuffer", "outputBuffer", "histogramUniforms"]],
                label: "histogram kernel",
                getDispatchGeometry: () => {
                    return [this.workgroupCount];
                },
            }),
        ];
    }
}

/*https://developer.nvidia.com/blog/gpu-pro-tip-fast-histograms-using-shared-atomics-maxwell/ */

export class HierarchicalHistogram extends BaseHistogram {
    constructor(args) {
        super(args);
    }
    finalizeRuntimeParameters() {
        this.workgroupSize = this.workgroupSize ?? 256;
        this.maxGSLWorkgroupCount = this.maxGSLWorkgroupCount ?? 512;  // Optimal balance based on benchmarks
        this.numThreadsPerWorkgroup = arrayProd(this.workgroupSize);

        // Calculate workgroup count with hardware limit
        const inputLength = this.getBuffer("inputBuffer").size / datatypeToBytes(this.datatype);
        const idealWorkgroupCount = Math.ceil(inputLength / this.numThreadsPerWorkgroup);
        this.workgroupCount = Math.min(idealWorkgroupCount, this.maxGSLWorkgroupCount);
        this.numPartials = this.workgroupCount;

        // CREATE THE HISTOGRAM UNIFORM DATA
        this.histogramUniforms = new Uint32Array([
            inputLength,
            this.numBins,
            this.workgroupCount, // Add workgroup count to uniforms
            0 // padding for alignment
        ]);
        const histogramUniformsFloat = new Float32Array([
            this.minValue,
            this.maxValue
        ]);

        // Create 32-byte buffer with proper padding
        const ub = new ArrayBuffer(32);
        const dv = new DataView(ub);

        dv.setUint32(0, inputLength, true);           // offset 0
        dv.setUint32(4, this.numBins, true);          // offset 4
        dv.setUint32(8, this.workgroupCount, true);   // offset 8
        dv.setUint32(12, 0, true);                    // padding0 at 12

        dv.setFloat32(16, this.minValue, true);       // offset 16
        dv.setFloat32(20, this.maxValue, true);       // offset 20
        // bytes 24..31 left as zero padding (padding1, padding2)

        this.histogramUniformsBuffer = new Uint8Array(ub);

        // CREATE CLEAR UNIFORMS (4 bytes)
        const clearUB = new ArrayBuffer(4);
        const clearDV = new DataView(clearUB);
        clearDV.setUint32(0, this.numBins, true);
        this.clearUniformsBuffer = new Uint8Array(clearUB);

        // CREATE ACCUMULATE UNIFORMS (8 bytes - removed dispatch count, using numBins workgroups)
        this.accumulateDispatchCount = this.numBins;  // ONE workgroup per bin!
        const accUB = new ArrayBuffer(8);
        const accDV = new DataView(accUB);
        accDV.setUint32(0, this.numBins, true);
        accDV.setUint32(4, this.workgroupCount, true);
        this.accumulateUniformsBuffer = new Uint8Array(accUB);
    }
    // Kernel 1: Each workgroup builds its own local histogram using workgroup atomics
    histogramPerWorkgroupKernel = () => {
        return /*wgsl*/`
        //input buffer
        @group(0) @binding(0) var<storage, read>inputBuffer:array<${this.datatype}>;
        //partial bufferr
        @group(0) @binding(1) var<storage,read_write>partials:array<u32>;

        struct HistogramUniforms{
            inputLength:u32,
            numBins:u32,
            numWorkgroups:u32,
            padding0:u32,
            minValue:f32,
            maxValue:f32,
            padding1:f32,
            padding2:f32,
        }
        @group(0) @binding(2) var<uniform>uniforms:HistogramUniforms;

        // Private workgroup-local histogram
        var<workgroup> privateHistogram: array<atomic<u32>, ${this.numBins}>;
        
        @compute @workgroup_size(${this.workgroupSize})
        fn histogramPerWorkgroupKernel(
            @builtin(global_invocation_id) globalId: vec3<u32>,
            @builtin(local_invocation_id) localId: vec3<u32>,
            @builtin(workgroup_id) wgId: vec3<u32>
        ) {
            let gwIndex: u32 = globalId.x;
            let localIndex: u32 = localId.x;
            let wgIndex: u32 = wgId.x;
            
            let WGS: u32 = ${this.workgroupSize}u;
            let NB: u32 = uniforms.numBins;
            
            // Zero out private histogram
            var i: u32 = localIndex;
            loop {
                if (i >= NB) { break; }
                atomicStore(&privateHistogram[i], 0u);
                i = i + WGS;
            }
            workgroupBarrier();
            
            // CRITICAL FIX: Process CONTIGUOUS BLOCKS per workgroup (like scan does!)
            // This ensures each workgroup has INDEPENDENT data, reducing overlap
            // Each workgroup processes: [wgIndex * chunkSize, (wgIndex+1) * chunkSize)
            let inputLen: u32 = uniforms.inputLength;
            let elementsPerWorkgroup: u32 = (inputLen + uniforms.numWorkgroups - 1u) / uniforms.numWorkgroups;
            let workgroupStart: u32 = wgIndex * elementsPerWorkgroup;
            let workgroupEnd: u32 = min(workgroupStart + elementsPerWorkgroup, inputLen);
            
            // Each thread processes elements within THIS workgroup's chunk
            var idx: u32 = workgroupStart + localIndex;
            
            while (idx < workgroupEnd) {
                //Read input value
                let value: ${this.datatype} = inputBuffer[idx];
                //Compute bin index (map to 0..numBins-1) with divide-by-zero protection
                let range: f32 = uniforms.maxValue - uniforms.minValue;
                var normalized: f32 = select((f32(value) - uniforms.minValue) / range, 0.0, range == 0.0);
                normalized = clamp(normalized, 0.0, 0.999999); // keep in [0,1) to avoid out-of-range
                var binIndex: i32 = i32(floor(normalized * f32(NB)));
                
                //Clamp to valid bin range
                binIndex = clamp(binIndex, 0, i32(NB) - 1);
                
                // Increment workgroup-local histogram (NO cross-workgroup conflicts!)
                atomicAdd(&privateHistogram[u32(binIndex)], 1u);
                // Stride within THIS workgroup's chunk only
                idx = idx + WGS;
            }
            
            workgroupBarrier();
            
            // Write local histogram to partials buffer with COALESCED writes
            // Transpose the write pattern: partials[binIndex * numWorkgroups + wgIndex]
            // This makes accumulation much faster with sequential reads
            var b: u32 = localIndex;
            while (b < NB) {
                let partialCount: u32 = atomicLoad(&privateHistogram[b]);
                // Transposed layout for better accumulation performance
                let partialIndex: u32 = b * uniforms.numWorkgroups + wgIndex;
                partials[partialIndex] = partialCount;
                b = b + WGS;
            }
        }`;
    };
    // kernel 2: for accumulated partials - SIMPLE ACCUMULATION, NO BARRIERS!
    accumulateHistogramsKernel = () => {
        return /* wgsl */ `
        // Partials buffers
        @group(0) @binding(0) var<storage, read> partials: array<u32>;
        // Output buffer with atomics for final sum
        @group(0) @binding(1) var<storage, read_write> outputBuffer: array<atomic<u32>>;

        struct AccumulateUniforms {
            numBins: u32,
            numWorkgroups: u32,
        }
        @group(0) @binding(2) var<uniform> uniforms: AccumulateUniforms;
        
        @compute @workgroup_size(${this.workgroupSize})
        fn accumulateHistogramsKernel(
            @builtin(global_invocation_id) globalId: vec3<u32>,
            @builtin(local_invocation_id) localId: vec3<u32>,
            @builtin(workgroup_id) workgroupId: vec3<u32>
        ) {
            let localIdx: u32 = localId.x;
            let binIdx: u32 = workgroupId.x;  // ONE workgroup per bin!
            let NB: u32 = uniforms.numBins;
            let numWG: u32 = uniforms.numWorkgroups;
            let WGS: u32 = ${this.workgroupSize}u;
            
            // Guard against out-of-bounds
            if (binIdx >= NB) {
                return;
            }
            
            // Simple accumulation: each thread processes its subset and atomically adds
            // With only 512 workgroups, this is MUCH faster than tree reduction with barriers
            var wgIdx: u32 = localIdx;
            
            // Each thread directly accumulates its subset to output
            // Sequential memory access pattern: partials[binIdx * numWG + wgIdx]
            while (wgIdx < numWG) {
                let partialIndex: u32 = binIdx * numWG + wgIdx;
                let value: u32 = partials[partialIndex];
                atomicAdd(&outputBuffer[binIdx], value);
                wgIdx = wgIdx + WGS;
            }
            
            // NO BARRIERS, NO SHARED MEMORY - just simple atomic accumulation!
            // 256 threads * 2 iterations = 512 atomic adds per bin
            // This is WAY cheaper than 8 barrier operations!
        }`;
    };

    // Kernel to clear/initialize the output buffer
    clearOutputBufferKernel = () => {
        return /* wgsl */ `
        @group(0) @binding(0) var<storage, read_write> outputBuffer: array<atomic<u32>>;
        
        struct ClearUniforms {
            numBins: u32,
        }
        @group(0) @binding(1) var<uniform> uniforms: ClearUniforms;
        
        @compute @workgroup_size(${this.workgroupSize})
        fn clearOutputBufferKernel(
            @builtin(global_invocation_id) globalId: vec3<u32>
        ) {
            let idx: u32 = globalId.x;
            if (idx < uniforms.numBins) {
                atomicStore(&outputBuffer[idx], 0u);
            }
        }`;
    };

    compute() {
        this.finalizeRuntimeParameters();

        return [
            // Allocate partials buffer: transposed layout [numBins][numWorkgroups]
            new AllocateBuffer({
                label: "partials",
                size: this.numBins * this.workgroupCount * 4, // 4 bytes per u32
            }),

            // Allocate and populate histogram uniforms
            new AllocateBuffer({
                label: "histogramUniforms",
                size: this.histogramUniformsBuffer.byteLength,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
                populateWith: this.histogramUniformsBuffer,
            }),

            // Allocate and populate accumulate uniforms
            new AllocateBuffer({
                label: "accumulateUniforms",
                size: this.accumulateUniformsBuffer.byteLength,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
                populateWith: this.accumulateUniformsBuffer,
            }),

            // Kernel 1: Each workgroup builds local histogram from contiguous chunks
            new Kernel({
                kernel: this.histogramPerWorkgroupKernel,
                bufferTypes: [["read-only-storage", "storage", "uniform"]],
                bindings: [["inputBuffer", "partials", "histogramUniforms"]],
                label: "histogram per workgroup",
                logKernelCodeToConsole: false,
                getDispatchGeometry: () => {
                    return [this.workgroupCount];
                },
            }),

            // Kernel 2: Accumulate all per-workgroup histograms (NO ATOMICS!)
            new Kernel({
                kernel: this.accumulateHistogramsKernel,
                bufferTypes: [["read-only-storage", "storage", "uniform"]],
                bindings: [["partials", "outputBuffer", "accumulateUniforms"]],
                label: "accumulate histograms",
                logKernelCodeToConsole: false,
                getDispatchGeometry: () => {
                    // One workgroup per bin = numBins workgroups
                    return [this.numBins];
                },
            }),
        ];
    }

}

const HistogramParams = {
    inputLength: [2 ** 20, 2 ** 22, 2 ** 24, 2 ** 26, 2 ** 27],  // 5 sizes: 1M, 4M, 16M, 64M, 128M
    numBins: [64, 256],
    workgroupSize: [256],  // Single workgroup size like scan
    maxGSLWorkgroupCount: [256, 512, 1024],  // Test different workgroup counts to find optimal
    minValue: [-1024.0],  // Can add more ranges like [-1024.0, 0.0, -512.0]
    maxValue: [1024.0],   // Can add more ranges like [1024.0, 1.0, 512.0]
};

const HistogramParamsSingleton = {
    inputLength: [2 ** 10],
    numBins: [64],
    workgroupSize: [256],
    maxGSLWorkgroupCount: [64],
    minValue: [-1024.0],
    maxValue: [1024.0],
};

export const WGHistogramTestSuite = new BaseTestSuite({
    category: "histogram",
    testSuite: "workgroup histogram",
    trials: 10,
    params: HistogramParams,  // Use full parameter sweep
    uniqueRuns: ["inputLength", "numBins", "workgroupSize"],  // Add "minValue", "maxValue" if you want different ranges as separate tests
    primitive: WGHistogram,
    primitiveArgs: {
        datatype: "f32",
        binop: BinOpAddU32,  // Histogram output is always u32 (counts)
        gputimestamps: true,
    },
    plots: [
        histogramBandwidthPlot
    ],
});

export const HierarchicalHistogramTestSuite = new BaseTestSuite({
    category: "histogram",
    testSuite: "hierarchical histogram",
    trials: 10,
    params: HistogramParams,  // Use full parameter sweep
    uniqueRuns: ["inputLength", "numBins", "maxGSLWorkgroupCount"],  // Include workgroup count to see performance impact
    primitive: HierarchicalHistogram,
    primitiveArgs: {
        datatype: "f32",
        binop: BinOpAddU32,  // Histogram output is always u32 (counts)
        gputimestamps: true,
    },
    plots: [
        histogramBandwidthPlot
    ],
});











