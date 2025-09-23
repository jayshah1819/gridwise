import { Pane } from "https://cdn.jsdelivr.net/npm/tweakpane@4.0.5/dist/tweakpane.min.js";
import { BinOpAdd, BinOpMax, BinOpMin, makeBinOp } from "../binop.mjs";
import { datatypeToTypedArray } from "../util.mjs";
import { DLDFScan } from "../scandldf.mjs";

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

const pane = new Pane();
const params = {
  scanType: "exclusive",
  datatype: "u32",
  binop: "add",
  inputLength: 2 ** 20,
};
pane.addBinding(params, "scanType", {
  options: {
    // what it shows : what it returns
    exclusive: "exclusive",
    inclusive: "inclusive",
    reduce: "reduce",
  },
});
pane.addBinding(params, "datatype", {
  options: {
    // what it shows : what it returns
    u32: "u32",
    i32: "i32",
    f32: "f32",
  },
});
pane.addBinding(params, "binop", {
  options: {
    // what it shows : what it returns
    add: "add",
    max: "max",
    min: "min",
  },
});
pane.addBinding(params, "inputLength", { format: (v) => Math.floor(v) });
pane.on("change", async (ev) => {
  if (params.inputLength % 4 !== 0) {
    params.inputLength = Math.floor(params.inputLength / 4) * 4;
  }
  pane.refresh();
  buildAndRun();
});

async function buildAndRun() {
  /* generate an input dataset */
  if (params.inputLength % 4 !== 0) {
    console.warn(
      "Input length (currently: ",
      params.inputLength,
      ") must be divisible by 4 (output is likely to be incorrect) "
    );
  }
  const memsrc = new (datatypeToTypedArray(params.datatype))(
    params.inputLength
  );
  /* the gymnastics below are to try to generate GPU-native {i,u,f}32
   * datatypes, there's probably an easier/faster way to do it */
  for (let i = 0; i < params.inputLength; i++) {
    switch (params.datatype) {
      case "u32":
      case "i32":
        memsrc[i] = i === 0 ? 11 : memsrc[i - 1] + 1; // trying to get u32s
        break;
      case "f32":
        /* attempt to evenly distribute ints between [-1023, 1023] */
        memsrc[i] =
          (Math.random() < 0.5 ? 1 : -1) *
          Math.floor(Math.random() * Math.pow(2, 10));
        break;
    }
  }
  console.log("input array", memsrc);

  /* declare the primitive */
  const dldfscanPrimitive = new DLDFScan({
    device,
    binop: makeBinOp({ op: params.binop, datatype: params.datatype }),
    type: params.scanType,
    datatype: params.datatype,
  });

  const primitive = dldfscanPrimitive;

  /* size the output */
  let memdestBytes;
  if (
    primitive.constructor.name === "DLDFScan" &&
    primitive.type === "reduce"
  ) {
    memdestBytes = 4;
  } else {
    memdestBytes = memsrc.byteLength;
  }

  /* allocate/create buffers on the GPU to hold in/out data */
  const memsrcBuffer = device.createBuffer({
    label: `memory source buffer (${params.datatype})`,
    size: memsrc.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(memsrcBuffer, 0, memsrc);

  const memdestBuffer = device.createBuffer({
    label: "memory destination buffer",
    size: memdestBytes,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST /* COPY_DST necessary for initialization */,
  });

  const mappableMemdestBuffer = device.createBuffer({
    label: "mappable memory destination buffer",
    size: memdestBytes,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  /* actually run the primitive */
  await primitive.execute({
    inputBuffer: memsrcBuffer,
    outputBuffer: memdestBuffer,
  });

  /* copy output back to host */
  const encoder = device.createCommandEncoder({
    label: "copy result CPU->GPU encoder",
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
  const memdest = new (datatypeToTypedArray(params.datatype))(
    mappableMemdestBuffer.getMappedRange().slice()
  );
  mappableMemdestBuffer.unmap();

  console.log("output array", memdest);

  if (primitive.validate) {
    const errorstr = primitive.validate({
      inputBuffer: memsrc,
      outputBuffer: memdest,
    });
    if (errorstr === "") {
      console.info("Validation passed");
    } else {
      console.error(`Validation failed:\n${errorstr}`);
    }
  }
}

buildAndRun();
