import { OneSweepSort } from "../onesweep.mjs";
import { datatypeToTypedArray } from "../util.mjs";

// Set up WebGPU device
const adapter = await navigator.gpu?.requestAdapter();
const device = await adapter?.requestDevice({
  requiredLimits: {
    maxComputeWorkgroupStorageSize: 32768,
  },
  requiredFeatures: adapter.features.has("subgroups") ? ["subgroups"] : [],
});

if (!device) {
  alert("WebGPU is not supported on this browser/device.");
  throw new Error("WebGPU not supported");
}

// Canvas setup
const canvas = document.getElementById("barChart");
const ctx = canvas.getContext("2d");

// Set canvas resolution
function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);
  canvas.style.width = rect.width + 'px';
  canvas.style.height = rect.height + 'px';
}
resizeCanvas();
window.addEventListener("resize", resizeCanvas);

// UI Elements
const codeEditor = document.getElementById("code-editor");
const arraySizeInput = document.getElementById("arraySize");
const directionSelect = document.getElementById("direction");
const newArrayBtn = document.getElementById("newArrayBtn");
const runBtn = document.getElementById("runBtn");
const statusDiv = document.getElementById("status");

// Default sort code template
const defaultCode = `async function runSort(device, array, direction) {
  const sorter = new OneSweepSort({
    device: device,
    datatype: "u32",
    direction: direction,
    copyOutputToTemp: true,
  });
  
  const inputBuffer = device.createBuffer({
    size: array.byteLength,
    usage: GPUBufferUsage.STORAGE | 
           GPUBufferUsage.COPY_DST | 
           GPUBufferUsage.COPY_SRC,
  });
  
  const outputBuffer = device.createBuffer({
    size: array.byteLength,
    usage: GPUBufferUsage.STORAGE | 
           GPUBufferUsage.COPY_SRC | 
           GPUBufferUsage.COPY_DST,
  });
  
  device.queue.writeBuffer(inputBuffer, 0, array);
  
  await sorter.execute({
    keysInOut: inputBuffer,
    keysTemp: outputBuffer,
  });
  
  const mappableBuffer = device.createBuffer({
    size: array.byteLength,
    usage: GPUBufferUsage.MAP_READ | 
           GPUBufferUsage.COPY_DST,
  });
  
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(
    outputBuffer, 0, 
    mappableBuffer, 0, 
    array.byteLength
  );
  device.queue.submit([encoder.finish()]);
  
  await mappableBuffer.mapAsync(GPUMapMode.READ);
  const result = new Uint32Array(
    mappableBuffer.getMappedRange().slice()
  );
  mappableBuffer.unmap();
  
  return result;
}

return runSort;`;

// Initialize code editor with default code
codeEditor.value = defaultCode;

// Generate random array
function generateArray(size) {
  const arr = new Uint32Array(size);
  for (let i = 0; i < size; i++) {
    arr[i] = Math.floor(Math.random() * 100) + 1;
  }
  return arr;
}

// Draw bars on canvas
function drawBars(array) {
  const rect = canvas.getBoundingClientRect();
  const width = rect.width;
  const height = rect.height;
  
  // Enable anti-aliasing
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  
  ctx.clearRect(0, 0, width, height);
  
  const barWidth = width / array.length;
  const maxValue = Math.max(...array);
  const padding = Math.max(2, barWidth * 0.15);
  const bottomMargin = 40;
  const topMargin = 20;
  
  // Draw grid lines
  ctx.strokeStyle = '#f0f0f0';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 5; i++) {
    const y = topMargin + (height - bottomMargin - topMargin) * (i / 5);
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }
  
  for (let i = 0; i < array.length; i++) {
    const barHeight = (array[i] / maxValue) * (height - bottomMargin - topMargin);
    const x = i * barWidth;
    const y = height - barHeight - bottomMargin;
    const actualBarWidth = barWidth - padding * 2;
    const cornerRadius = Math.min(4, actualBarWidth / 2);
    
    // Draw rounded rectangle bar
    ctx.beginPath();
    ctx.moveTo(x + padding + cornerRadius, y);
    ctx.lineTo(x + padding + actualBarWidth - cornerRadius, y);
    ctx.quadraticCurveTo(x + padding + actualBarWidth, y, x + padding + actualBarWidth, y + cornerRadius);
    ctx.lineTo(x + padding + actualBarWidth, y + barHeight);
    ctx.lineTo(x + padding, y + barHeight);
    ctx.lineTo(x + padding, y + cornerRadius);
    ctx.quadraticCurveTo(x + padding, y, x + padding + cornerRadius, y);
    ctx.closePath();
    
    // Create gradient
    const gradient = ctx.createLinearGradient(x, y, x, y + barHeight);
    gradient.addColorStop(0, '#4fd1c5');
    gradient.addColorStop(1, '#38b2ac');
    
    ctx.fillStyle = gradient;
    ctx.fill();
    
    // Add shadow effect
    ctx.shadowColor = 'rgba(0, 0, 0, 0.1)';
    ctx.shadowBlur = 4;
    ctx.shadowOffsetY = 2;
    ctx.fill();
    ctx.shadowColor = 'transparent';
    
    // Draw value on top for smaller arrays
    if (array.length <= 64) {
      ctx.fillStyle = "#1d1d1f";
      ctx.font = `bold ${Math.max(9, Math.min(12, barWidth * 0.4))}px -apple-system, BlinkMacSystemFont, sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "bottom";
      ctx.fillText(array[i], x + barWidth / 2, y - 4);
    }
  }
}

// Animate transition
async function animateTransition(beforeArray, afterArray) {
  const steps = 30;
  const delay = 20;
  
  for (let step = 0; step <= steps; step++) {
    const progress = step / steps;
    const interpolated = new Uint32Array(beforeArray.length);
    
    for (let i = 0; i < beforeArray.length; i++) {
      interpolated[i] = Math.round(
        beforeArray[i] + (afterArray[i] - beforeArray[i]) * progress
      );
    }
    
    drawBars(interpolated);
    await new Promise(resolve => setTimeout(resolve, delay));
  }
}

// Show status message
function showStatus(message, isError = false) {
  statusDiv.textContent = message;
  statusDiv.className = isError ? 'error' : 'success';
  statusDiv.style.display = 'block';
  
  setTimeout(() => {
    statusDiv.style.display = 'none';
  }, 3000);
}

// State to track current array
let currentArray = generateArray(64);

// New Array button handler
newArrayBtn.addEventListener("click", () => {
  const arraySize = parseInt(arraySizeInput.value);
  currentArray = generateArray(arraySize);
  drawBars(currentArray);
  showStatus(`✓ Generated new array with ${arraySize} elements`);
  console.log("New array:", currentArray);
});

// Run the user's code
runBtn.addEventListener("click", async () => {
  const arraySize = parseInt(arraySizeInput.value);
  const direction = directionSelect.value;
  
  runBtn.disabled = true;
  newArrayBtn.disabled = true;
  
  try {
    // Use current array instead of generating new one
    const inputArray = new Uint32Array(currentArray);
    drawBars(inputArray);
    
    await new Promise(resolve => setTimeout(resolve, 500));
    
    // Execute user's code
    const userCode = codeEditor.value;
    const runSortFunction = new Function(
      'device', 'OneSweepSort', 'GPUBufferUsage', 'GPUMapMode', 'Uint32Array',
      userCode
    );
    
    const sortFn = runSortFunction(
      device, OneSweepSort, GPUBufferUsage, GPUMapMode, Uint32Array
    );
    
    const sortedArray = await sortFn(device, inputArray, direction);
    
    // Update current array with sorted result
    currentArray = sortedArray;
    
    // Animate the result
    await animateTransition(inputArray, sortedArray);
    
    console.log("Input:", inputArray);
    console.log("Sorted:", sortedArray);
    
    showStatus("✓ Sort completed successfully!");
    
  } catch (error) {
    console.error("Error running code:", error);
    showStatus(`✗ Error: ${error.message}`, true);
  } finally {
    runBtn.disabled = false;
    newArrayBtn.disabled = false;
  }
});

// Validate array size input
arraySizeInput.addEventListener("change", () => {
  let value = parseInt(arraySizeInput.value);
  value = Math.round(value / 16) * 16;
  value = Math.max(16, Math.min(256, value));
  arraySizeInput.value = value;
});

// Initialize with current array
drawBars(currentArray);
