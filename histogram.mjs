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

    /*privatized workgroup*/
    var<workgroup>privateHistogram:array<atomic<u32>,$this.numBins>;

    @compute @workgroup_size(${this.workgroupSize})

    fn main(
        @builtin(global_invocation_id)globalId:vec3<u32>;
        @builtin(local_invocation_id)localId:vec3<u32>;
        @builtin(workgroup_id)wg_id:vec3<u32>;

        let g


    )



    
    `;

    }
}