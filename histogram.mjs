import { range, arrayProd } from "./util.mjs";
import {
    BasePrimitive,
    Kernel,
    // InitializeMemoryBlock,
    AllocateBuffer,
} from "./primitive.mjs";
import { BaseTestSuite } from "./testsuite.mjs";
import { BinOpAddU32, BinOpAddF32 } from "./binop.mjs";
import { datatypeToTypedArray, datatypeToBytes } from "./util.mjs";



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
                returnString += `\nBin ${bin}: expected ${referenceOutput[bin]}, instead saw ${memdest[bin]} (diff: ${Math.abs(
                    (referenceOutput[bin] - memdest[bin]) / referenceOutput[bin]
                )}).`;
                if (this.getBuffer("debugBuffer")) {
                    returnString += ` debug[${bin}] = ${this.getBuffer("debugBuffer").cpuBuffer[bin]}.`;
                }
                if (this.getBuffer("debug2Buffer")) {
                    returnString += ` debug2[${bin}] = ${this.getBuffer("debug2Buffer").cpuBuffer[bin]}.`;
                }
                allowedErrors--;
            }
        }

        if (returnString !== "") {
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
        }
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
        
        // Combine into single buffer
        this.histogramUniformsBuffer = new Uint8Array(
            this.histogramUniforms.byteLength + histogramUniformsFloat.byteLength
        );
        this.histogramUniformsBuffer.set(new Uint8Array(this.histogramUniforms.buffer), 0);
        this.histogramUniformsBuffer.set(new Uint8Array(histogramUniformsFloat.buffer), this.histogramUniforms.byteLength);

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
    };
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

      // compute bin index (map to 0..numBins-1)
      //to get range between [1,0]
      let normalized: f32 = (f32(value) - uniforms.minValue) / (uniforms.maxValue - uniforms.minValue);
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
        this.maxGSLWorkgroupCount = this.maxGSLWorkgroupCount ?? 256;
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
        
        // Combine into single buffer
        this.histogramUniformsBuffer = new Uint8Array(
            this.histogramUniforms.byteLength + histogramUniformsFloat.byteLength
        );
        this.histogramUniformsBuffer.set(new Uint8Array(this.histogramUniforms.buffer), 0);
        this.histogramUniformsBuffer.set(new Uint8Array(histogramUniformsFloat.buffer), this.histogramUniforms.byteLength);
        
        // CREATE ACCUMULATE UNIFORMS
        this.accumulateUniforms = new Uint32Array([
            this.numBins,
            this.workgroupCount
        ]);
    }
    // Kernel 1: Each workgroup builds its own local histogram
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
        }
        @group(0) @binding(2) var<uniform>uniforms:HistogramUniforms;

        //private workgroup 
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
            
            // Each thread processes elements with grid-stride loop
            var idx:u32 = gwIndex;
            let inputLen: u32 = uniforms.inputLength;
            let totalThreads: u32 = WGS * uniforms.numWorkgroups;
            
            while (idx < inputLen) {
                //Read input value
                let value: ${this.datatype} = inputBuffer[idx];
                //Compute bin index (map to 0..numBins-1)
                let normalized: f32 = (f32(value) - uniforms.minValue) / 
                                      (uniforms.maxValue - uniforms.minValue);
                var binIndex: i32 = i32(floor(normalized * f32(NB)));
                
                //Clamp to valid bin range
                binIndex = clamp(binIndex, 0, i32(NB) - 1);
                
                // Increment private histogram
                atomicAdd(&privateHistogram[u32(binIndex)], 1u);
                // Grid-stride loop
                idx = idx + totalThreads;
            }
            
            workgroupBarrier();
            
            // Write local histogram to partials buffer
            // Each workgroup writes to partials[wgIndex * numBins : (wgIndex+1) * numBins]
            var b: u32 = localIndex;
            while (b < NB) {
                let partialCount: u32 = atomicLoad(&privateHistogram[b]);
                let partialIndex: u32 = wgIndex * NB + b;
                partials[partialIndex] = partialCount;
                b = b + WGS;
            }
        }`;
    };
    // kernel 2: for accumulated partials
    accumulateHistogramsKernel = () => {
        return /* wgsl */ `
        // Partials buffers
        @group(0) @binding(0) var<storage, read> partials: array<u32>;
        // Output buffer: final accumulated histogram of partial buffer
        @group(0) @binding(1) var<storage, read_write> outputBuffer: array<atomic<u32>>;
        

        struct AccumulateUniforms {
            numBins: u32,
            numWorkgroups: u32,
        }
        @group(0) @binding(2) var<uniform> uniforms: AccumulateUniforms;
        
        @compute @workgroup_size(${this.workgroupSize})
        fn accumulateHistogramsKernel(
            @builtin(global_invocation_id) globalId: vec3<u32>
        ) {
            let binIndex: u32 = globalId.x;
            let NB: u32 = uniforms.numBins;
            
            // Only process valid bins
            if (binIndex >= NB) {
                return;
            }
            

            // Sum across all workgroup histograms for this bin
            var sum: u32 = 0u;

            for (var wg: u32 = 0u; wg < uniforms.numWorkgroups; wg = wg + 1u) {
                let partialIndex: u32 = wg * NB + binIndex;
                sum = sum + partials[partialIndex];
            }
            
            //Write to output
            atomicStore(&outputBuffer[binIndex], sum);
        }`;
    };
    compute() {
        this.finalizeRuntimeParameters();

        return [
            // Allocate partials buffer: numBins × workgroupCount
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
                size: this.accumulateUniforms.byteLength,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
                populateWith: this.accumulateUniforms,
            }),

            // Kernel 1: Each workgroup builds local histogram
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

            // Kernel 2: Accumulate all per-workgroup histograms
            new Kernel({
                kernel: this.accumulateHistogramsKernel,
                bufferTypes: [["read-only-storage", "storage", "uniform"]],
                bindings: [["partials", "outputBuffer", "accumulateUniforms"]],
                label: "accumulate histograms",
                logKernelCodeToConsole: false,
                getDispatchGeometry: () => {
                    // Dispatch enough workgroups to cover all bins
                    return [Math.ceil(this.numBins / this.workgroupSize)];
                },
            }),
        ];
    }

}

const HistogramParams = {
    inputLength: range(8, 24).map((i) => 2 ** i),  // 256 to 16M elements, like DLDF scan
    numBins: [64, 256],
    workgroupSize: [256],  // Single workgroup size like scan
    maxGSLWorkgroupCount: [256],  // Single workgroup count
};

const HistogramParamsSingleton = {
    inputLength: [2 ** 10],
    numBins: [64],
    workgroupSize: [256],
    maxGSLWorkgroupCount: [64],
};

export const WGHistogramTestSuite = new BaseTestSuite({
    category: "histogram",
    testSuite: "workgroup histogram",
    trials: 10,
    params: HistogramParams,  // Use full parameter sweep
    uniqueRuns: ["inputLength", "numBins", "workgroupSize"],
    primitive: WGHistogram,
    primitiveArgs: {
        datatype: "f32",
        binop: BinOpAddU32,  // Histogram output is always u32 (counts)
        minValue: -1024.0,
        maxValue: 1024.0,
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
    uniqueRuns: ["inputLength", "numBins"],
    primitive: HierarchicalHistogram,
    primitiveArgs: {
        datatype: "f32",
        binop: BinOpAddU32,  // Histogram output is always u32 (counts)
        minValue: -1024.0,
        maxValue: 1024.0,
        gputimestamps: true,
    },
    plots: [
        histogramBandwidthPlot
    ],
});











