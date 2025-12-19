import { DLDFScan } from "../scandldf.mjs";
import { BinOpAdd, BinOpMax, BinOpMin } from "../binop.mjs";
import { datatypeToTypedArray } from "../util.mjs";

// Set up WebGPU device
const adapter = await navigator.gpu?.requestAdapter();
const hasSubgroups = adapter?.features.has("subgroups");
const device = await adapter?.requestDevice({
  requiredFeatures: [
    ...(hasSubgroups ? ["subgroups"] : []),
  ],
});

if (!device) {
  alert("WebGPU is not supported on this browser/device.");
  throw new Error("WebGPU not supported");
}

// Canvas setup
const canvas = document.getElementById("scanCanvas");
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
const scanTypeSelect = document.getElementById("scanType");
const operationSelect = document.getElementById("operation");
const newArrayBtn = document.getElementById("newArrayBtn");
const runBtn = document.getElementById("runBtn");
const statusDiv = document.getElementById("status");

// Default scan code template
const defaultCode = `async function runScan(device, array, scanType, binop) {
  const scanner = new DLDFScan({
    device: device,
    binop: binop,
    type: scanType,
    datatype: "u32",
  });
  
  const inputBuffer = device.createBuffer({
    size: array.byteLength,
    usage: GPUBufferUsage.STORAGE | 
           GPUBufferUsage.COPY_DST,
  });
  
  const outputBuffer = device.createBuffer({
    size: array.byteLength,
    usage: GPUBufferUsage.STORAGE | 
           GPUBufferUsage.COPY_SRC | 
           GPUBufferUsage.COPY_DST,
  });
  
  device.queue.writeBuffer(inputBuffer, 0, array);
  
  await scanner.execute({
    inputBuffer: inputBuffer,
    outputBuffer: outputBuffer,
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

return runScan;`;

// Initialize code editor with default code
codeEditor.value = defaultCode;

// Generate random array with smaller values for better visualization
function generateArray(size) {
  const arr = new Uint32Array(size);
  for (let i = 0; i < size; i++) {
    arr[i] = Math.floor(Math.random() * 9) + 1; // Values from 1-9
  }
  return arr;
}

// Animation state
let animationFrameId = null;

// Draw the scan visualization with wave animation
function drawScanVisualization(inputArray, outputArray, progress = 0) {
  const rect = canvas.getBoundingClientRect();
  const width = rect.width;
  const height = rect.height;
  
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.clearRect(0, 0, width, height);
  
  const padding = 40;
  const usableWidth = width - padding * 2;
  const usableHeight = height - padding * 2;
  
  // Calculate cell size to be proper big squares
  const spacing = 8;
  const cellSize = Math.min(120, (usableWidth - spacing * (inputArray.length - 1)) / inputArray.length);
  const cellWidth = cellSize;
  const cellHeight = cellSize;
  const barWidth = cellSize + spacing;
  
  // Calculate max value for scaling
  const maxInput = Math.max(...inputArray);
  const maxOutput = Math.max(...outputArray);
  const maxValue = Math.max(maxInput, maxOutput, 1);
  
  // Draw input row
  const inputY = padding + 40;
  ctx.font = 'bold 16px -apple-system, BlinkMacSystemFont, sans-serif';
  ctx.fillStyle = '#1d1d1f';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'bottom';
  ctx.fillText('Input:', padding, inputY - 15);
  
  for (let i = 0; i < inputArray.length; i++) {
    const x = padding + i * barWidth;
    
    // Draw cell border (box style)
    ctx.fillStyle = '#ffffff';
    ctx.strokeStyle = '#333333';
    ctx.lineWidth = 3;
    
    ctx.fillRect(x, inputY, cellWidth, cellHeight);
    ctx.strokeRect(x, inputY, cellWidth, cellHeight);
    
    // Draw value
    ctx.fillStyle = '#1d1d1f';
    ctx.font = 'bold 32px -apple-system, BlinkMacSystemFont, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(inputArray[i], x + cellWidth / 2, inputY + cellHeight / 2);
  }
  
  // Draw arrows and animation effects
  const arrowY = inputY + cellHeight + 30;
  const currentIndex = Math.floor(progress * inputArray.length);
  
  for (let i = 0; i < inputArray.length; i++) {
    const x = padding + i * barWidth;
    const arrowProgress = Math.max(0, Math.min(1, progress * inputArray.length - i));
    
    if (arrowProgress > 0) {
      // Draw flowing arrow
      ctx.strokeStyle = i <= currentIndex ? '#f5576c' : '#e0e0e0';
      ctx.lineWidth = 3;
      ctx.globalAlpha = 0.3 + 0.7 * arrowProgress;
      
      const arrowStartY = inputY + cellHeight + 5;
      const arrowEndY = arrowY + 20;
      
      ctx.beginPath();
      ctx.moveTo(x + cellWidth / 2, arrowStartY);
      ctx.lineTo(x + cellWidth / 2, arrowEndY);
      ctx.stroke();
      
      // Arrow head
      ctx.beginPath();
      ctx.moveTo(x + cellWidth / 2 - 5, arrowEndY - 5);
      ctx.lineTo(x + cellWidth / 2, arrowEndY);
      ctx.lineTo(x + cellWidth / 2 + 5, arrowEndY - 5);
      ctx.stroke();
      
      ctx.globalAlpha = 1;
    }
  }
  
  // Draw output row
  const outputY = arrowY + 50;
  ctx.font = 'bold 16px -apple-system, BlinkMacSystemFont, sans-serif';
  ctx.fillStyle = '#1d1d1f';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'bottom';
  ctx.fillText('Output:', padding, outputY - 15);
  
  for (let i = 0; i < outputArray.length; i++) {
    const x = padding + i * barWidth;
    const cellProgress = Math.max(0, Math.min(1, progress * inputArray.length - i));
    
    // Only show cells that have started animating (with a threshold)
    if (cellProgress > 0.01) {
      // Determine color based on completion
      const isComplete = i < currentIndex;
      
      // Draw box with fill color
      if (isComplete) {
        ctx.fillStyle = '#e3f2fd'; // Light blue for completed
        ctx.strokeStyle = '#1976d2'; // Darker blue border
      } else {
        ctx.fillStyle = '#fff3e0'; // Light orange for computing
        ctx.strokeStyle = '#f57c00'; // Orange border
      }
      
      // Scale animation
      const scale = 0.5 + 0.5 * cellProgress;
      const scaledWidth = cellWidth * scale;
      const scaledHeight = cellHeight * scale;
      const offsetX = (cellWidth - scaledWidth) / 2;
      const offsetY = (cellHeight - scaledHeight) / 2;
      
      ctx.lineWidth = 3;
      ctx.globalAlpha = cellProgress;
      
      ctx.fillRect(x + offsetX, outputY + offsetY, scaledWidth, scaledHeight);
      ctx.strokeRect(x + offsetX, outputY + offsetY, scaledWidth, scaledHeight);
      
      // Draw value with same alpha - only if progress is significant
      if (cellProgress > 0.2) {
        ctx.fillStyle = isComplete ? '#1976d2' : '#f57c00';
        ctx.font = `bold ${32 * scale}px -apple-system, BlinkMacSystemFont, sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(outputArray[i], x + cellWidth / 2, outputY + cellHeight / 2);
      }
      
      ctx.globalAlpha = 1;
    }
  }
  
  // Draw bar chart comparison at bottom
  const chartY = outputY + cellHeight + 60;
  const chartHeight = height - chartY - padding;
  
  if (chartHeight > 50) {
    ctx.font = 'bold 16px -apple-system, BlinkMacSystemFont, sans-serif';
    ctx.fillStyle = '#1d1d1f';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'bottom';
    ctx.fillText('Accumulated Values:', padding, chartY - 15);
    
    for (let i = 0; i < outputArray.length; i++) {
      const x = padding + i * barWidth;
      const barProgress = Math.max(0, Math.min(1, progress * inputArray.length - i));
      
      if (barProgress > 0.01) {
        const barHeight = (outputArray[i] / maxValue) * chartHeight * barProgress;
        const barY = chartY + chartHeight - barHeight;
        
        // Simple fill color for bars
        ctx.fillStyle = '#64b5f6';
        ctx.strokeStyle = '#1976d2';
        ctx.lineWidth = 2;
        
        ctx.globalAlpha = barProgress;
        ctx.fillRect(x, barY, cellWidth, barHeight);
        ctx.strokeRect(x, barY, cellWidth, barHeight);
        
        // Draw value on top - only if progress is significant
        if (barHeight > 20 && barProgress > 0.3) {
          ctx.fillStyle = '#1d1d1f';
          ctx.font = 'bold 11px -apple-system, BlinkMacSystemFont, sans-serif';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'bottom';
          ctx.fillText(outputArray[i], x + cellWidth / 2, barY - 3);
        }
        
        ctx.globalAlpha = 1;
      }
    }
  }
  
  // Draw progress indicator
  ctx.fillStyle = '#39bda7';
  ctx.font = 'bold 12px -apple-system, BlinkMacSystemFont, sans-serif';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'top';
  ctx.fillText(
    `Computing: ${Math.min(currentIndex + 1, inputArray.length)} / ${inputArray.length}`,
    width - padding,
    padding - 5
  );
}

// Helper function to draw rounded rectangles
function roundRect(ctx, x, y, width, height, radius) {
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.lineTo(x + width - radius, y);
  ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
  ctx.lineTo(x + width, y + height - radius);
  ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
  ctx.lineTo(x + radius, y + height);
  ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
  ctx.lineTo(x, y + radius);
  ctx.quadraticCurveTo(x, y, x + radius, y);
  ctx.closePath();
}

// Animate scan operation
async function animateScan(inputArray, outputArray) {
  const duration = 2000; // 2 seconds
  const fps = 60;
  const frames = (duration / 1000) * fps;
  
  return new Promise((resolve) => {
    let frame = 0;
    
    function animate() {
      frame++;
      const progress = Math.min(frame / frames, 1);
      
      drawScanVisualization(inputArray, outputArray, progress);
      
      if (progress < 1) {
        animationFrameId = requestAnimationFrame(animate);
      } else {
        animationFrameId = null;
        resolve();
      }
    }
    
    animate();
  });
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
let currentArray = generateArray(12);

// Helper function to get binop based on operation
function getBinOp(operation) {
  switch (operation) {
    case 'add':
      return new BinOpAdd({ datatype: "u32" });
    case 'max':
      return new BinOpMax({ datatype: "u32" });
    case 'min':
      return new BinOpMin({ datatype: "u32" });
    default:
      return new BinOpAdd({ datatype: "u32" });
  }
}

// New Array button handler
newArrayBtn.addEventListener("click", () => {
  const arraySize = parseInt(arraySizeInput.value);
  currentArray = generateArray(arraySize);
  
  // Create dummy output for initial display
  const dummyOutput = new Uint32Array(arraySize).fill(0);
  drawScanVisualization(currentArray, dummyOutput, 0);
  
  showStatus(`Generated new array with ${arraySize} elements`);
  console.log("New array:", currentArray);
});

// Run the user's code
runBtn.addEventListener("click", async () => {
  const scanType = scanTypeSelect.value;
  const operation = operationSelect.value;
  
  runBtn.disabled = true;
  newArrayBtn.disabled = true;
  
  try {
    // Cancel any ongoing animation
    if (animationFrameId) {
      cancelAnimationFrame(animationFrameId);
      animationFrameId = null;
    }
    
    // Use current array
    const inputArray = new Uint32Array(currentArray);
    const dummyOutput = new Uint32Array(inputArray.length).fill(0);
    drawScanVisualization(inputArray, dummyOutput, 0);
    
    await new Promise(resolve => setTimeout(resolve, 300));
    
    // Execute user's code
    const userCode = codeEditor.value;
    const runScanFunction = new Function(
      'device', 'DLDFScan', 'BinOpAdd', 'BinOpMax', 'BinOpMin',
      'GPUBufferUsage', 'GPUMapMode', 'Uint32Array',
      userCode
    );
    
    const scanFn = runScanFunction(
      device, DLDFScan, BinOpAdd, BinOpMax, BinOpMin,
      GPUBufferUsage, GPUMapMode, Uint32Array
    );
    
    const binop = getBinOp(operation);
    const scannedArray = await scanFn(device, inputArray, scanType, binop);
    
    // Animate the result
    await animateScan(inputArray, scannedArray);
    
    console.log("Input:", inputArray);
    console.log("Scanned:", scannedArray);
    
    showStatus(`${scanType.charAt(0).toUpperCase() + scanType.slice(1)} scan completed successfully!`);
    
  } catch (error) {
    console.error("Error running code:", error);
    showStatus(`Error: ${error.message}`, true);
  } finally {
    runBtn.disabled = false;
    newArrayBtn.disabled = false;
  }
});

// Validate array size input
arraySizeInput.addEventListener("change", () => {
  let value = parseInt(arraySizeInput.value);
  value = Math.round(value / 4) * 4;
  value = Math.max(8, Math.min(24, value));
  arraySizeInput.value = value;
});

// Initialize with current array
const initialOutput = new Uint32Array(currentArray.length).fill(0);
drawScanVisualization(currentArray, initialOutput, 0);
