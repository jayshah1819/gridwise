import { range, arrayProd } from "./util.mjs";
import {
    BasePrimitive,
    Kernel,
    // InitializeMemoryBlock,
    AllocateBuffer,
} from "./primitive.mjs";
import { BaseTestSuite } from "./testsuite.mjs";
import { BinOpAddU32 } from "./binop.mjs";
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

        // Assign binop (default to addition for counting)
        this.binop = args.binop ?? new BinOpAddU32();

        if (this.datatype != this.binop.datatype) {
            throw new Error(
                `${this.constructor.name}: datatype (${this.datatype}) is incompatible with binop datatype (${this.binop.datatype}).`
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
            referenceOutput = new (datatypeToTypedArray(this.datatype))(
                memdest.length
            );
        } catch (error) {
            console.error(error, "Tried to allocate array of length", memdest.length);
        }
        /*Intialize histogram bins<--------------------------check this again */
        for (let bin = 0; bin < this.numBins; bin++) {
            referenceOutput[bin] = this.binop.identity;
        }
        //build reference for histogram
        for (let i = 0; i < memsrc.length; i++) {
            const value = memsrc[i];

            //map value to bin's index
            const normalized = (value - this.minValue) / (this.maxValue - this.minValue);
            let binIndex = Math.floor(normalized * this.numBins);
            //clamp
            binIndex = Math.max(0, Math.min(binIndex, this.numBins - 1));
            referenceOutput[binIndex] = this.binop.op(referenceOutput[binIndex], 1);
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

export const histogramWGCountPlot = {
    x: { field: "inputBytes", label: "Input array size (B)" },
    y: { field: "bandwidth", label: "Achieved bandwidth (GB/s)" },
    fx: { field: "timing" },
    stroke: { field: "workgroupCount" },
    test_br: "gpuinfo.description",
    caption: "Lines are workgroup count",
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
    test_br: "gpuinfo.description",
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
        /* tuneble parameters*/
        this.workgroupSize = this.workgroupSize ?? 256;
        this.maxGSLWorkgroupCount = this.maxGSLWorkgroupCount ?? 256;

        /*Compute settings based on tunable parameters */
        this.workgroupCount = Math.min(Math.ceil(this.getBuffer("inputBuffer").size / this.workgroupSize), this.maxGSLWorkgroupCount);
        this.numPartials = this.workgroupCount;

    }
    histogramKernelDefinition = () => {
        return /*wgsl*/ `
    enable subgroups;
    // input buffer
    @group(0) @binding(0) var<storage, read> inputBuffer: array<${this.datatype}>;
    //output buffer
    @group(0) @binding(1) var<storage,read_write>outputBuffer:array<atomic<u32>>;
    
    struct histogramUniforms{
        numBins:u32,
        minValue:f32,
        maxValue:f32

    };
    @group(0) @binding(1) var<uniform> uniforms:histogramUniforms;

    ${BasePrimitive.fnDeclarations.commondefinitions};

    //privatized workgroup
    var<workgroup>privateHistogram:array<atomic<u32>,$this.numBins>;

    @compute @workgroup_size(${this.workgroupSize})

    fn main(
        @builtin(global_invocation_id)globalId:vec3<u32>;
        @builtin(local_invocation_id)localId:vec3<u32>;
        @builtin(workgroup_id)wg_id:vec3<u32>;

        let gwIndex:u32 =global_id..x;
        let localIndex:u32=local_id.x;
        let wgIndex:u32=wg_id.x;

        //zero private histograms
        var i:u32=localIndex;
        let WGS: u32=${this.workgroupSize}u;
        let NB:u32=unigorm.numsBins;

        loop{
            if(i>=NB){break;}
            atomicExchange(&privateHistogram[i],0u);
            i=i+WGS;
        }
        workgroupBarrier();



    )
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
      idx = idx + ${this.workgroupCount}u * WGS; 
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


    
    `;

    }
}