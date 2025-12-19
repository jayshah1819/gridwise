import { OneSweepSort } from "../onesweep.mjs";
import { DLDFScan } from "../scandldf.mjs";
import { BinOpAdd } from "../binop.mjs";

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

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

function resizeCanvas() {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
}
resizeCanvas();
window.addEventListener("resize", resizeCanvas);

class Particle {
  constructor(x, y, colorIndex, size) {
    this.x = x;
    this.y = y;
    this.originalX = x;
    this.originalY = y;
    this.vx = (Math.random() - 0.5) * 0.5;
    this.vy = (Math.random() - 0.5) * 0.5;
    this.targetX = x;
    this.targetY = y;
    this.colorIndex = colorIndex;
    this.size = size;
    this.baseSize = size;
  }

  update() {
    if (!isOperating) {
      const dx = mouseX - this.x;
      const dy = mouseY - this.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      
      if (mouseDown && dist < 150 && dist > 0.1) {
        const force = (150 - dist) / 150 * (attractMode ? 0.4 : -0.4);
        this.vx += (dx / dist) * force;
        this.vy += (dy / dist) * force;
      }
      
      this.vx *= 0.99;
      this.vy *= 0.99;
      
      this.x += this.vx;
      this.y += this.vy;
      
      if (this.x < 0 || this.x > canvas.width) this.vx *= -0.5;
      if (this.y < 0 || this.y > canvas.height) this.vy *= -0.5;
      this.x = Math.max(0, Math.min(canvas.width, this.x));
      this.y = Math.max(0, Math.min(canvas.height, this.y));
    } else {
      this.x += (this.targetX - this.x) * 0.08;
      this.y += (this.targetY - this.y) * 0.08;
    }
  }

  draw() {
    const color = COLORS[this.colorIndex];
    const glowSize = this.size * 2.5;
    
    const gradient = ctx.createRadialGradient(this.x, this.y, 0, this.x, this.y, glowSize);
    gradient.addColorStop(0, `hsla(${color.h}, ${color.s}%, ${color.l}%, 0.9)`);
    gradient.addColorStop(0.5, `hsla(${color.h}, ${color.s}%, ${color.l}%, 0.4)`);
    gradient.addColorStop(1, `hsla(${color.h}, ${color.s}%, ${color.l}%, 0)`);
    
    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.arc(this.x, this.y, glowSize, 0, Math.PI * 2);
    ctx.fill();
    
    ctx.fillStyle = `hsl(${color.h}, ${color.s}%, ${color.l}%)`;
    ctx.beginPath();
    ctx.arc(this.x, this.y, this.size * 0.6, 0, Math.PI * 2);
    ctx.fill();
  }
}

const COLORS = [
  { h: 350, s: 85, l: 60 },
  { h: 30, s: 90, l: 55 },
  { h: 180, s: 80, l: 55 },
  { h: 280, s: 85, l: 60 },
  { h: 200, s: 80, l: 60 },
  { h: 60, s: 90, l: 60 },
  { h: 120, s: 75, l: 55 },
  { h: 240, s: 85, l: 60 },
];

let particles = [];
let particleCount = 10000;
let isOperating = false;
let currentOperation = null;
let mouseX = -1000;
let mouseY = -1000;
let mouseDown = false;
let attractMode = true;

const starSlider = document.getElementById("starSlider");
const starCountDisplay = document.getElementById("starCount");
const sortBtn = document.getElementById("sortBtn");
const scanBtn = document.getElementById("scanBtn");
const reduceBtn = document.getElementById("reduceBtn");

function generateParticles(count) {
  particles = [];
  
  for (let i = 0; i < count; i++) {
    const x = Math.random() * canvas.width;
    const y = Math.random() * canvas.height;
    const colorIndex = Math.floor(Math.random() * COLORS.length);
    const size = 0.4 + Math.random() * 1.2;
    
    const particle = new Particle(x, y, colorIndex, size);
    particle.originalX = x;
    particle.originalY = y;
    particles.push(particle);
  }
  
  return particles;
}

function render() {
  ctx.fillStyle = "rgba(10, 10, 15, 0.25)";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  
  for (const particle of particles) {
    particle.update();
    particle.draw();
  }
  
  requestAnimationFrame(render);
}


async function performSort() {
  if (currentOperation) {
    clearTimeout(currentOperation);
  }
  
  isOperating = true;
  
  try {
    const sizeValues = new Uint32Array(particles.map(p => Math.floor(p.size * 1000)));
    
    const sorter = new OneSweepSort({
      device: device,
      datatype: "u32",
      direction: "ascending",
      copyOutputToTemp: true,
    });
    
    const inputBuffer = device.createBuffer({
      size: sizeValues.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    
    const outputBuffer = device.createBuffer({
      size: sizeValues.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    
    device.queue.writeBuffer(inputBuffer, 0, sizeValues);
    
    await sorter.execute({
      keysInOut: inputBuffer,
      keysTemp: outputBuffer,
    });
    
    const mappableBuffer = device.createBuffer({
      size: sizeValues.byteLength,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(outputBuffer, 0, mappableBuffer, 0, sizeValues.byteLength);
    device.queue.submit([encoder.finish()]);
    
    await mappableBuffer.mapAsync(GPUMapMode.READ);
    mappableBuffer.unmap();
    
    particles.sort((a, b) => a.size - b.size);
    
    const padding = 100;
    const usableWidth = canvas.width - padding * 2;
    const usableHeight = canvas.height - padding * 2;
    
    particles.forEach((particle, i) => {
      const progress = i / particles.length;
      particle.targetX = padding + progress * usableWidth;
      particle.targetY = padding + (Math.random() * 0.5 + 0.25) * usableHeight;
      particle.vx = 0;
      particle.vy = 0;
    });
    
    inputBuffer.destroy();
    outputBuffer.destroy();
    
    currentOperation = setTimeout(() => {
      particles.forEach((particle) => {
        particle.targetX = particle.originalX;
        particle.targetY = particle.originalY;
      });
      
      currentOperation = setTimeout(() => {
        isOperating = false;
        currentOperation = null;
      }, 8000);
    }, 8000);
    
  } catch (error) {
    console.error("Sort error:", error);
    isOperating = false;
    currentOperation = null;
  }
}

async function performScan() {
  if (currentOperation) {
    clearTimeout(currentOperation);
  }
  
  isOperating = true;
  
  try {
    const sizeValues = new Uint32Array(particles.map(p => Math.floor(p.size * 1000)));
    
    const scanner = new DLDFScan({
      device: device,
      binop: new BinOpAdd({ datatype: "u32" }),
      type: "exclusive",
      datatype: "u32",
    });
    
    const inputBuffer = device.createBuffer({
      size: sizeValues.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    
    const outputBuffer = device.createBuffer({
      size: sizeValues.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    
    device.queue.writeBuffer(inputBuffer, 0, sizeValues);
    
    await scanner.execute({
      inputBuffer: inputBuffer,
      outputBuffer: outputBuffer,
    });
    
    const mappableBuffer = device.createBuffer({
      size: sizeValues.byteLength,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(outputBuffer, 0, mappableBuffer, 0, sizeValues.byteLength);
    device.queue.submit([encoder.finish()]);
    
    await mappableBuffer.mapAsync(GPUMapMode.READ);
    const scannedValues = new Uint32Array(mappableBuffer.getMappedRange().slice());
    mappableBuffer.unmap();
    
    const maxValue = Math.max(...scannedValues);
    
    particles.forEach((particle, i) => {
      const normalized = scannedValues[i] / maxValue;
      const waveProgress = i / particles.length;
      particle.targetX = waveProgress * canvas.width;
      particle.targetY = canvas.height / 2 + Math.sin(waveProgress * Math.PI * 12) * (canvas.height * 0.25) * (particle.size / 1.2);
      particle.vx = 0;
      particle.vy = 0;
    });
    
    inputBuffer.destroy();
    outputBuffer.destroy();
    
    currentOperation = setTimeout(() => {
      particles.forEach((particle) => {
        particle.targetX = particle.originalX;
        particle.targetY = particle.originalY;
      });
      
      currentOperation = setTimeout(() => {
        isOperating = false;
        currentOperation = null;
      }, 8000);
    }, 8000);
    
  } catch (error) {
    console.error("Scan error:", error);
    isOperating = false;
    currentOperation = null;
  }
}

async function performReduce() {
  if (currentOperation) {
    clearTimeout(currentOperation);
  }
  
  isOperating = true;
  
  try {
    const sizeValues = new Uint32Array(particles.map(p => Math.floor(p.size * 1000)));
    
    particles.sort((a, b) => a.size - b.size);
    
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const maxRadius = Math.min(canvas.width, canvas.height) * 0.45;
    
    particles.forEach((particle, i) => {
      const progress = i / particles.length;
      const angle = (i * 137.5) * (Math.PI / 180);
      const radius = Math.sqrt(progress) * maxRadius;
      
      particle.targetX = centerX + Math.cos(angle) * radius;
      particle.targetY = centerY + Math.sin(angle) * radius;
      particle.vx = 0;
      particle.vy = 0;
    });
    
    const totalSize = sizeValues.reduce((a, b) => a + b, 0);
    
    currentOperation = setTimeout(() => {
      particles.forEach((particle) => {
        particle.targetX = particle.originalX;
        particle.targetY = particle.originalY;
      });
      
      currentOperation = setTimeout(() => {
        isOperating = false;
        currentOperation = null;
      }, 8000);
    }, 8000);
    
  } catch (error) {
    console.error("Reduce error:", error);
    isOperating = false;
    currentOperation = null;
  }
}

function formatStarCount(count) {
  if (count >= 1000) {
    return (count / 1000).toFixed(0) + "K";
  }
  return count;
}

// Event Listeners
canvas.addEventListener("mousemove", (e) => {
  mouseX = e.clientX;
  mouseY = e.clientY;
});

canvas.addEventListener("mouseleave", () => {
  mouseX = -1000;
  mouseY = -1000;
});

starSlider.addEventListener("input", (e) => {
  particleCount = parseInt(e.target.value);
  starCountDisplay.textContent = formatStarCount(particleCount);
});

starSlider.addEventListener("change", (e) => {
  particleCount = parseInt(e.target.value);
  
  if (currentOperation) {
    clearTimeout(currentOperation);
    currentOperation = null;
  }
  
  isOperating = false;
  
  setTimeout(() => {
    generateParticles(particleCount);
  }, 100);
});

canvas.addEventListener("mousedown", (e) => {
  mouseDown = true;
  attractMode = !e.shiftKey;
});

canvas.addEventListener("mouseup", () => {
  mouseDown = false;
});

sortBtn.addEventListener("click", performSort);
scanBtn.addEventListener("click", performScan);
reduceBtn.addEventListener("click", performReduce);

generateParticles(particleCount);
render();
