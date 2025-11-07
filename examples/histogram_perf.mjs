import { Pane } from "https://cdn.jsdelivr.net/npm/tweakpane@4.0.5/dist/tweakpane.min.js";
import { BinOpAddU32 } from "../binop.mjs";
import {
  datatypeToTypedArray,
  logspaceRounded,
  datatypeToBytes,
} from "../util.mjs";
import { WGHistogram, HierarchicalHistogram } from "../histogram.mjs";

let Plot = await import(
  "https://cdn.jsdelivr.net/npm/@observablehq/plot@0.6/+esm"
);

/* set up a WebGPU device */
const adapter = await navigator.gpu?.requestAdapter();
const hasSubgroups = adapter.features.has("subgroups");
const hasTimestampQuery = adapter.features.has("timestamp-query");
const device = await adapter?.requestDevice({
  requiredFeatures: [
    ...(hasTimestampQuery ? ["timestamp-query"] : []),
    ...(hasSubgroups ? ["subgroups"] : []),
  ],
});

if (!device) {
  console.error("Fatal error: Device does not support WebGPU.");
}

/* set up the UI, with parameters stored in the "params" object */
const pane = new Pane();
const params = {
  /* defaults */
  implementation: "workgroup",
  datatype: "f32",
  numBins: 64,
  minValue: -1024.0,
  maxValue: 1024.0,
  inputLengthStart: 2 ** 20,
  inputLengthEnd: 2 ** 24,
  inputCount: 5,
  trials: 10,
};

pane.addBinding(params, "implementation", {
  options: {
    "Workgroup Histogram": "workgroup",
    "Hierarchical Histogram": "hierarchical",
  },
});

pane.addBinding(params, "datatype", {
  options: {
    f32: "f32",
    i32: "i32",
    u32: "u32",
  },
});

pane.addBinding(params, "numBins", {
  min: 16,
  max: 1024,
  step: 16,
});

pane.addBinding(params, "minValue");
pane.addBinding(params, "maxValue");
pane.addBinding(params, "inputLengthStart", { format: (v) => Math.floor(v) });
pane.addBinding(params, "inputLengthEnd", { format: (v) => Math.floor(v) });
pane.addBinding(params, "inputCount", {
  min: 1,
  max: 20,
  step: 1,
  format: (v) => Math.floor(v)
});
pane.addBinding(params, "trials", {
  min: 1,
  max: 100,
  step: 1,
  format: (v) => Math.floor(v)
});

const button = pane.addButton({
  title: "Start",
});

button.on("click", async () => {
  params.inputLengthStart = Math.floor(params.inputLengthStart);
  params.inputLengthEnd = Math.floor(params.inputLengthEnd);
  params.inputCount = Math.floor(params.inputCount);
  params.trials = Math.floor(params.trials);
  params.numBins = Math.floor(params.numBins);

  /* refresh the pane */
  pane.refresh();

  const results = document.getElementById("webgpu-results");
  const validation = await buildAndRun();

  results.innerHTML = `<p>Histogram Performance Test Complete</p>
  <ul>
  <li>Implementation: ${params.implementation}
  <li>Datatype: ${params.datatype}
  <li>Number of bins: ${params.numBins}
  <li>Value range: [${params.minValue}, ${params.maxValue}]
  <li>Input length: ${params.inputCount} lengths from ${params.inputLengthStart} to ${params.inputLengthEnd} (items)
  <li>Trials per config: ${params.trials}
  </ul>
  <p>${validation}</p>`;
});

/* all of the work is in this function */
async function buildAndRun() {
  let returnStr = "";
  const results = new Array(); // push new rows (experiments) onto this

  for (const inputLength of logspaceRounded(
    params.inputLengthStart,
    params.inputLengthEnd,
    params.inputCount
  )) {
    /* generate an input dataset */
    const memsrc = new (datatypeToTypedArray(params.datatype))(inputLength);

    /* generate random values within the histogram range */
    for (let i = 0; i < inputLength; i++) {
      switch (params.datatype) {
        case "u32":
          memsrc[i] = Math.floor(
            Math.random() * (params.maxValue - params.minValue) + params.minValue
          );
          break;
        case "i32":
        case "f32":
          memsrc[i] =
            Math.random() * (params.maxValue - params.minValue) + params.minValue;
          break;
      }
    }
    console.log("input array (first 20)", memsrc.slice(0, 20));

    /* declare the primitive */
    let primitive;
    switch (params.implementation) {
      case "workgroup":
        primitive = new WGHistogram({
          device,
          binop: BinOpAddU32,
          datatype: params.datatype,
          inputLength: inputLength,
          numBins: params.numBins,
          minValue: params.minValue,
          maxValue: params.maxValue,
          workgroupSize: 256,
        });
        break;
      case "hierarchical":
        primitive = new HierarchicalHistogram({
          device,
          binop: BinOpAddU32,
          datatype: params.datatype,
          inputLength: inputLength,
          numBins: params.numBins,
          minValue: params.minValue,
          maxValue: params.maxValue,
          workgroupSize: 256,
          maxGSLWorkgroupCount: 256,
        });
        break;
    }

    /* histogram output is always u32 with length = numBins */
    const memdestBytes = params.numBins * 4; // u32 = 4 bytes

    /* allocate/create buffers on the GPU to hold in/out data */
    const memsrcBuffer = device.createBuffer({
      label: `histogram input buffer (${params.datatype})`,
      size: memsrc.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(memsrcBuffer, 0, memsrc);

    /* histogram output buffer must be initialized to zeros (for atomic operations) */
    const memdestBuffer = device.createBuffer({
      label: "histogram output buffer (u32 counts)",
      size: memdestBytes,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
    });

    // Zero-initialize the output buffer
    const zeros = new Uint32Array(params.numBins);
    device.queue.writeBuffer(memdestBuffer, 0, zeros);

    const mappableMemdestBuffer = device.createBuffer({
      label: "mappable histogram output buffer",
      size: memdestBytes,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    /* call the primitive once to warm up */
    await primitive.execute({
      inputBuffer: memsrcBuffer,
      outputBuffer: memdestBuffer,
    });

    /* call params.trials times */
    await primitive.execute({
      inputBuffer: memsrcBuffer,
      outputBuffer: memdestBuffer,
      trials: params.trials,
      enableGPUTiming: hasTimestampQuery,
      enableCPUTiming: true,
    });

    const { gpuTotalTimeNS, cpuTotalTimeNS } =
      await primitive.getTimingResult();

    /* copy output back to host for validation */
    const encoder = device.createCommandEncoder({
      label: "copy result GPU->CPU encoder",
    });
    encoder.copyBufferToBuffer(
      memdestBuffer,
      0,
      mappableMemdestBuffer,
      0,
      mappableMemdestBuffer.size
    );
    const commandBuffer = encoder.finish();
    device.queue.submit([commandBuffer]);

    await mappableMemdestBuffer.mapAsync(GPUMapMode.READ);
    const memdest = new Uint32Array(
      mappableMemdestBuffer.getMappedRange().slice()
    );
    mappableMemdestBuffer.unmap();

    console.log("histogram output (bin counts):", memdest);

    /* validate */
    if (primitive.validate) {
      const errorstr = primitive.validate({
        inputBuffer: memsrc,
        outputBuffer: { cpuBuffer: memdest },
      });
      if (errorstr === "") {
        console.info("✓ Validation passed for inputLength =", inputLength);
      } else {
        console.error(`✗ Validation failed for inputLength = ${inputLength}:\n${errorstr}`);
        returnStr += `✗ Validation failed for inputLength = ${inputLength}\n`;
      }
    }

    /* record results */
    const inputBytes = inputLength * datatypeToBytes(params.datatype);
    const gpuAvgTime = gpuTotalTimeNS / params.trials;
    const cpuAvgTime = cpuTotalTimeNS / params.trials;

    // bytes transferred = input + output
    const bytesTransferred = inputBytes + (params.numBins * 4);
    const bandwidthGPU = bytesTransferred / gpuAvgTime; // GB/s
    const bandwidthCPU = bytesTransferred / cpuAvgTime; // GB/s

    results.push({
      inputLength,
      inputBytes,
      numBins: params.numBins,
      timing: "GPU",
      time: gpuAvgTime,
      bandwidth: bandwidthGPU,
    });
    results.push({
      inputLength,
      inputBytes,
      numBins: params.numBins,
      timing: "CPU",
      time: cpuAvgTime,
      bandwidth: bandwidthCPU,
    });

    console.info(`InputLength: ${inputLength}, GPU: ${(gpuAvgTime / 1e6).toFixed(3)}ms (${bandwidthGPU.toFixed(2)} GB/s), CPU: ${(cpuAvgTime / 1e6).toFixed(3)}ms (${bandwidthCPU.toFixed(2)} GB/s)`);
  }

  if (returnStr === "") {
    returnStr = "✓ All validations passed";
  }

  /* plot the results */
  console.log("Results:", results);
  console.table(results);

  const plotDiv = document.getElementById("plot");
  plotDiv.innerHTML = ""; // clear previous plots

  // Bandwidth plot
  const bandwidthPlot = Plot.plot({
    marks: [
      Plot.lineY(results, {
        x: "inputBytes",
        y: "bandwidth",
        stroke: "timing",
        tip: true,
      }),
      Plot.text(
        results,
        Plot.selectLast({
          x: "inputBytes",
          y: "bandwidth",
          z: "timing",
          text: "timing",
          textAnchor: "start",
          dx: 3,
        })
      ),
    ],
    x: { type: "log", label: "Input array size (bytes)" },
    y: { type: "log", label: "Achieved bandwidth (GB/s)" },
    color: { legend: true },
    width: 1280,
    title: `Histogram Bandwidth (${params.implementation})`,
    caption: `${params.numBins} bins, range [${params.minValue}, ${params.maxValue}], ${params.trials} trials`,
  });
  plotDiv.append(bandwidthPlot);

  // Time plot
  const timePlot = Plot.plot({
    marks: [
      Plot.lineY(results, {
        x: "inputBytes",
        y: "time",
        stroke: "timing",
        tip: true,
      }),
      Plot.text(
        results,
        Plot.selectLast({
          x: "inputBytes",
          y: "time",
          z: "timing",
          text: "timing",
          textAnchor: "start",
          dx: 3,
        })
      ),
    ],
    x: { type: "log", label: "Input array size (bytes)" },
    y: { type: "log", label: "Runtime (nanoseconds)" },
    color: { legend: true },
    width: 1280,
    title: `Histogram Runtime (${params.implementation})`,
    caption: `${params.numBins} bins, range [${params.minValue}, ${params.maxValue}], ${params.trials} trials`,
  });
  plotDiv.append(timePlot);

  return returnStr;
}
