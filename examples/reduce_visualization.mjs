// Energy Flow / Particle Merge Visualization for Reduce
// Uses actual Gridwise WebGPU reduce implementation

import { BinOpAdd, BinOpMax, BinOpMin } from "../binop.mjs";
import { datatypeToTypedArray } from "../util.mjs";
import { DLDFScan } from "../scandldf.mjs";

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

// WebGPU state
let device = null;
let adapter = null;
let reducePrimitive = null;

// Visualization state
let arraySize = 16;
let data = [];
let particles = [];
let energyStreams = [];
let currentStep = 0;
let maxSteps = 0;
let isPlaying = false;
let operation = 'sum';
let animationFrame = 0;
const ANIMATION_DURATION = 90;
let finalResult = null;

// Particle class
class Particle {
  constructor(x, y, value, index, isActive = true) {
    this.x = x;
    this.y = y;
    this.targetX = x;
    this.targetY = y;
    this.value = value;
    this.index = index;
    this.isActive = isActive;
    this.opacity = isActive ? 1 : 0.3;
    this.size = isActive ? 12 : 8;
    this.glow = 0;
    this.pulsePhase = Math.random() * Math.PI * 2;
  }

  update() {
    // Smooth movement
    this.x += (this.targetX - this.x) * 0.1;
    this.y += (this.targetY - this.y) * 0.1;
    
    // Pulsing glow
    this.pulsePhase += 0.05;
    this.glow = Math.sin(this.pulsePhase) * 0.3 + 0.7;
  }

  draw() {
    ctx.save();
    
    // Glow effect
    if (this.isActive) {
      const glowSize = this.size * (2 + this.glow);
      const gradient = ctx.createRadialGradient(this.x, this.y, 0, this.x, this.y, glowSize);
      gradient.addColorStop(0, `rgba(57, 189, 167, ${0.6 * this.opacity})`);
      gradient.addColorStop(0.5, `rgba(57, 189, 167, ${0.2 * this.opacity})`);
      gradient.addColorStop(1, 'rgba(57, 189, 167, 0)');
      
      ctx.fillStyle = gradient;
      ctx.fillRect(this.x - glowSize, this.y - glowSize, glowSize * 2, glowSize * 2);
    }
    
    // Core particle
    ctx.fillStyle = this.isActive 
      ? `rgba(57, 189, 167, ${this.opacity})` 
      : `rgba(134, 134, 139, ${this.opacity})`;
    ctx.beginPath();
    ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
    ctx.fill();
    
    // Value text
    if (this.isActive || currentStep === 0) {
      ctx.fillStyle = '#0a0a0a';
      ctx.font = 'bold 11px -apple-system';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(Math.round(this.value), this.x, this.y);
    }
    
    ctx.restore();
  }
}

// Energy Stream class
class EnergyStream {
  constructor(fromParticle, toParticle) {
    this.from = fromParticle;
    this.to = toParticle;
    this.progress = 0;
    this.particles = [];
    this.complete = false;
    
    // Create energy particles along the path
    for (let i = 0; i < 8; i++) {
      this.particles.push({
        offset: i / 8,
        phase: Math.random() * Math.PI * 2
      });
    }
  }

  update() {
    if (this.complete) return;
    
    this.progress += 0.015;
    
    if (this.progress >= 1) {
      this.complete = true;
    }
    
    // Update particle phases
    this.particles.forEach(p => {
      p.phase += 0.1;
    });
  }

  draw() {
    if (this.complete) return;
    
    ctx.save();
    
    // Draw each energy particle
    this.particles.forEach(p => {
      const t = Math.min(1, Math.max(0, this.progress - p.offset * 0.3));
      
      if (t > 0 && t < 1) {
        // Bezier curve for smooth flow
        const midX = (this.from.x + this.to.x) / 2;
        const midY = (this.from.y + this.to.y) / 2 - 50;
        
        const x = Math.pow(1 - t, 2) * this.from.x + 
                  2 * (1 - t) * t * midX + 
                  Math.pow(t, 2) * this.to.x;
        const y = Math.pow(1 - t, 2) * this.from.y + 
                  2 * (1 - t) * t * midY + 
                  Math.pow(t, 2) * this.to.y;
        
        // Pulsing energy particle
        const pulse = Math.sin(p.phase) * 0.5 + 0.5;
        const size = 4 + pulse * 2;
        const alpha = (1 - Math.abs(t - 0.5) * 2) * (0.6 + pulse * 0.4);
        
        // Glow
        const gradient = ctx.createRadialGradient(x, y, 0, x, y, size * 2);
        gradient.addColorStop(0, `rgba(57, 189, 167, ${alpha})`);
        gradient.addColorStop(1, 'rgba(57, 189, 167, 0)');
        
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(x, y, size * 2, 0, Math.PI * 2);
        ctx.fill();
        
        // Core
        ctx.fillStyle = `rgba(255, 255, 255, ${alpha})`;
        ctx.beginPath();
        ctx.arc(x, y, size / 2, 0, Math.PI * 2);
        ctx.fill();
      }
    });
    
    ctx.restore();
  }
}

// Initialize WebGPU
async function initWebGPU() {
  try {
    adapter = await navigator.gpu?.requestAdapter();
    if (!adapter) {
      console.error("WebGPU not supported");
      document.getElementById('stepInfo').innerHTML = '<strong>Error:</strong> WebGPU not supported in this browser';
      return false;
    }

    const hasSubgroups = adapter.features.has("subgroups");
    const hasTimestampQuery = adapter.features.has("timestamp-query");
    
    device = await adapter.requestDevice({
      requiredFeatures: [
        ...(hasTimestampQuery ? ["timestamp-query"] : []),
        ...(hasSubgroups ? ["subgroups"] : []),
      ],
    });

    if (!device) {
      console.error("Failed to get WebGPU device");
      return false;
    }

    console.log("WebGPU initialized successfully");
    return true;
  } catch (error) {
    console.error("WebGPU initialization error:", error);
    document.getElementById('stepInfo').innerHTML = '<strong>Error:</strong> Failed to initialize WebGPU';
    return false;
  }
}

// Execute actual GPU reduce and get result
async function executeGPUReduce() {
  if (!device) {
    console.error("WebGPU device not initialized");
    return null;
  }

  try {
    const datatype = "i32";
    let binop;
    
    switch (operation) {
      case 'sum':
        binop = new BinOpAdd({ datatype });
        break;
      case 'max':
        binop = new BinOpMax({ datatype });
        break;
      case 'min':
        binop = new BinOpMin({ datatype });
        break;
      default:
        binop = new BinOpAdd({ datatype });
    }

    // Create input buffer
    const memsrc = new Int32Array(data);
    
    // Create reduce primitive
    reducePrimitive = new DLDFScan({
      device,
      binop,
      type: "reduce",
      datatype,
    });

    // Create GPU buffers
    const memsrcBuffer = device.createBuffer({
      label: `reduce input buffer`,
      size: memsrc.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(memsrcBuffer, 0, memsrc);

    const memdestBuffer = device.createBuffer({
      label: "reduce output buffer",
      size: 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    const mappableMemdestBuffer = device.createBuffer({
      label: "mappable output buffer",
      size: 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    // Execute reduce on GPU
    await reducePrimitive.execute({
      inputBuffer: memsrcBuffer,
      outputBuffer: memdestBuffer,
    });

    // Copy result back
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(memdestBuffer, 0, mappableMemdestBuffer, 0, 4);
    device.queue.submit([encoder.finish()]);

    await mappableMemdestBuffer.mapAsync(GPUMapMode.READ);
    const result = new Int32Array(mappableMemdestBuffer.getMappedRange().slice());
    mappableMemdestBuffer.unmap();

    // Cleanup
    memsrcBuffer.destroy();
    memdestBuffer.destroy();
    mappableMemdestBuffer.destroy();

    console.log(`GPU Reduce (${operation}):`, result[0]);
    return result[0];
  } catch (error) {
    console.error("GPU reduce error:", error);
    return null;
  }
}

async function init() {
  // Resize canvas
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * window.devicePixelRatio;
  canvas.height = rect.height * window.devicePixelRatio;
  ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
  canvas.style.width = rect.width + 'px';
  canvas.style.height = rect.height + 'px';
  
  // Initialize WebGPU if not already done
  if (!device) {
    const success = await initWebGPU();
    if (!success) {
      return;
    }
  }
  
  generateRandomData();
  currentStep = 0;
  maxSteps = Math.ceil(Math.log2(arraySize));
  animationFrame = 0;
  finalResult = null;
  
  // Execute GPU reduce to get the actual result
  finalResult = await executeGPUReduce();
  
  createParticles();
  updateStepInfo();
  draw();
}

function generateRandomData() {
  data = [];
  for (let i = 0; i < arraySize; i++) {
    data.push(Math.floor(Math.random() * 90) + 10);
  }
}

function createParticles() {
  particles = [];
  energyStreams = [];
  
  const rect = canvas.getBoundingClientRect();
  const padding = 60;
  const spacing = (rect.width - padding * 2) / (arraySize - 1);
  const yPos = rect.height / 2;
  
  const currentValues = getCurrentValues();
  
  for (let i = 0; i < arraySize; i++) {
    const x = padding + i * spacing;
    const isActive = isActiveAtStep(i, currentStep);
    particles.push(new Particle(x, yPos, currentValues[i], i, isActive));
  }
}

function updateStepInfo() {
  const stepInfo = document.getElementById('stepInfo');
  
  if (currentStep === 0) {
    stepInfo.innerHTML = `<strong>Step 0:</strong> Starting with ${arraySize} energy particles (GPU ready)`;
  } else if (currentStep > maxSteps) {
    const result = finalResult !== null ? finalResult : getReducedValue(arraySize);
    stepInfo.innerHTML = `<strong>Complete!</strong> GPU computed final result: <strong>${result}</strong>`;
  } else {
    const activeCount = Math.ceil(arraySize / Math.pow(2, currentStep));
    const pairCount = Math.floor(activeCount / 2);
    const opName = operation === 'sum' ? 'sum' : operation === 'max' ? 'maximum' : 'minimum';
    stepInfo.innerHTML = `<strong>Step ${currentStep} of ${maxSteps}:</strong> ${pairCount} parallel GPU merges (${opName})`;
  }
}

function getOperation() {
  switch (operation) {
    case 'sum': return (a, b) => a + b;
    case 'max': return (a, b) => Math.max(a, b);
    case 'min': return (a, b) => Math.min(a, b);
    default: return (a, b) => a + b;
  }
}

function getReducedValue(count) {
  const op = getOperation();
  let result = data[0];
  for (let i = 1; i < count && i < data.length; i++) {
    result = op(result, data[i]);
  }
  return result;
}

function getCurrentValues() {
  if (currentStep === 0) return data.slice();
  
  const op = getOperation();
  let values = data.slice();
  
  for (let step = 1; step <= currentStep; step++) {
    const newValues = [];
    const stride = Math.pow(2, step);
    
    for (let i = 0; i < values.length; i += stride) {
      if (i + stride / 2 < values.length) {
        newValues.push(op(values[i], values[i + stride / 2]));
      } else {
        newValues.push(values[i]);
      }
    }
    
    for (let i = 0; i < values.length; i++) {
      const groupIdx = Math.floor(i / stride);
      if (groupIdx < newValues.length) {
        values[i] = newValues[groupIdx];
      }
    }
  }
  
  return values;
}

function isActiveAtStep(index, step) {
  if (step === 0) return true;
  if (step > maxSteps) return index === 0;
  
  const stride = Math.pow(2, step);
  return index % stride === 0 && index + stride / 2 < arraySize;
}

function nextStep() {
  if (currentStep > maxSteps) return false;
  
  currentStep++;
  animationFrame = 0;
  
  // Create energy streams for merging pairs
  if (currentStep > 0 && currentStep <= maxSteps) {
    energyStreams = [];
    const stride = Math.pow(2, currentStep);
    
    for (let i = 0; i < arraySize; i += stride) {
      if (i + stride / 2 < arraySize) {
        const from = particles[i + stride / 2];
        const to = particles[i];
        energyStreams.push(new EnergyStream(from, to));
      }
    }
  }
  
  updateStepInfo();
  return currentStep <= maxSteps + 1;
}

function animate() {
  animationFrame++;
  
  // Update particles
  particles.forEach(p => p.update());
  
  // Update energy streams
  energyStreams.forEach(s => s.update());
  
  // Update particle states mid-animation
  if (animationFrame === Math.floor(ANIMATION_DURATION * 0.6)) {
    const currentValues = getCurrentValues();
    particles.forEach((p, i) => {
      p.value = currentValues[i];
      p.isActive = isActiveAtStep(i, currentStep);
      p.opacity = p.isActive ? 1 : 0.3;
      p.size = p.isActive ? 12 : 8;
    });
  }
  
  // Final step: make result glow stronger
  if (currentStep > maxSteps) {
    particles[0].size = 20;
    particles[0].glow = 1;
  }
  
  draw();
  
  if (animationFrame >= ANIMATION_DURATION) {
    if (isPlaying) {
      const hasMore = nextStep();
      if (hasMore) {
        requestAnimationFrame(animate);
      } else {
        stop();
      }
    }
  } else {
    requestAnimationFrame(animate);
  }
}

function draw() {
  // Clear with dark background
  ctx.fillStyle = '#0a0a0a';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  
  // Draw energy streams
  energyStreams.forEach(s => s.draw());
  
  // Draw particles
  particles.forEach(p => p.draw());
}

function play() {
  if (currentStep > maxSteps) {
    reset();
  }
  
  isPlaying = true;
  document.getElementById('playBtn').textContent = 'Pause';
  
  if (animationFrame === 0) {
    nextStep();
  }
  animate();
}

function stop() {
  isPlaying = false;
  document.getElementById('playBtn').textContent = 'Play';
}

function togglePlay() {
  if (isPlaying) {
    stop();
  } else {
    play();
  }
}

function reset() {
  stop();
  init();
}

function applySize() {
  const newSize = parseInt(document.getElementById('arraySizeInput').value);
  if (newSize >= 4 && newSize <= 64 && newSize % 4 === 0) {
    arraySize = newSize;
    reset();
  }
}

// Event listeners
document.getElementById('playBtn').addEventListener('click', togglePlay);
document.getElementById('stepBtn').addEventListener('click', () => {
  stop();
  if (currentStep > maxSteps) {
    reset();
  } else if (animationFrame === 0) {
    nextStep();
    animate();
  }
});
document.getElementById('resetBtn').addEventListener('click', reset);
document.getElementById('arraySizeInput').addEventListener('change', applySize);
document.getElementById('operationSelect').addEventListener('change', (e) => {
  operation = e.target.value;
  reset();
});

// Handle window resize
window.addEventListener('resize', () => {
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * window.devicePixelRatio;
  canvas.height = rect.height * window.devicePixelRatio;
  ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
  canvas.style.width = rect.width + 'px';
  canvas.style.height = rect.height + 'px';
  createParticles();
  draw();
});

// Initialize
init();
