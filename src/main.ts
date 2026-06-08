import './style.css'
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

// Hero video autoplay - wait for DOM to be ready
document.addEventListener('DOMContentLoaded', () => {
  const heroVideo = document.getElementById('hero-video') as HTMLVideoElement;
  if (heroVideo) {
    // Ensure the video loads and plays
    heroVideo.load();
    heroVideo.play().catch(err => {
      console.error('Video autoplay failed:', err);
    });
  }
});

// BibTeX Modal functionality
const bibtexBtn = document.getElementById('bibtex-btn');
const bibtexModal = document.getElementById('bibtex-modal');
const closeModal = document.getElementById('close-modal');
const copyBibtex = document.getElementById('copy-bibtex');

bibtexBtn?.addEventListener('click', () => {
  bibtexModal?.classList.remove('hidden');
});

closeModal?.addEventListener('click', () => {
  bibtexModal?.classList.add('hidden');
});

bibtexModal?.addEventListener('click', (e) => {
  if (e.target === bibtexModal) {
    bibtexModal.classList.add('hidden');
  }
});

copyBibtex?.addEventListener('click', async () => {
  const bibtexContent = document.getElementById('bibtex-content')?.textContent;
  if (bibtexContent) {
    try {
      await navigator.clipboard.writeText(bibtexContent);
      const btn = copyBibtex as HTMLButtonElement;
      const originalText = btn.textContent;
      btn.textContent = 'Copied!';
      setTimeout(() => {
        btn.textContent = originalText;
      }, 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  }
});

// Smooth scroll for internal links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
  anchor.addEventListener('click', function (this: HTMLAnchorElement, e) {
    e.preventDefault();
    const target = document.querySelector(this.getAttribute('href')!);
    target?.scrollIntoView({ behavior: 'smooth' });
  });
});

// Generate random sample from standard normal distribution
function randomNormal(): number {
  // Box-Muller transform
  const u1 = Math.random();
  const u2 = Math.random();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

// Convert quaternion to axis-angle (rotation vector) using canonical form
function quatToRotvec(quat: THREE.Quaternion): THREE.Vector3 {
  // Ensure quaternion is canonical (w >= 0)
  const canonicalQuat = quat.clone();
  if (canonicalQuat.w < 0) {
    canonicalQuat.x = -canonicalQuat.x;
    canonicalQuat.y = -canonicalQuat.y;
    canonicalQuat.z = -canonicalQuat.z;
    canonicalQuat.w = -canonicalQuat.w;
  }

  const axNorm = Math.sqrt(
    canonicalQuat.x * canonicalQuat.x +
    canonicalQuat.y * canonicalQuat.y +
    canonicalQuat.z * canonicalQuat.z
  );

  const angle = 2 * Math.atan2(axNorm, canonicalQuat.w);
  const smallAngle = angle <= 1e-3;
  const angle2 = angle * angle;
  const smallScale = 2 + angle2 / 12 + 7 * angle2 * angle2 / 2880;

  // Avoid division by zero with Taylor series approximation
  const divSin = Math.sin(angle / 2.0) + (smallAngle ? 1 : 0);
  const largeScale = angle / divSin;
  const scale = smallAngle ? smallScale : largeScale;

  return new THREE.Vector3(
    scale * canonicalQuat.x,
    scale * canonicalQuat.y,
    scale * canonicalQuat.z
  );
}

const NUM_POINTS = 3000;

const REPRESENTATION_COLORS: Record<string, THREE.Color> = {
  'tangent': new THREE.Color(0x0E4DAB),
  'matrix': new THREE.Color(0xF1750F),
  'quaternion': new THREE.Color(0xD426C2),
  'euler': new THREE.Color(0x26C706),
};

const REPRESENTATION_DIMS: Record<string, number> = {
  'tangent': 3,
  'euler': 3,
  'quaternion': 4,
  'matrix': 9,
};

type DistributionPointData = { positions: Float32Array; colors: Float32Array };

function generateDistributionPoints(representation: string, stdDev: number): DistributionPointData {
  const baseColor = REPRESENTATION_COLORS[representation] || new THREE.Color(0x3b82f6);
  const dim = REPRESENTATION_DIMS[representation] || 3;

  const positions = new Float32Array(NUM_POINTS * 3);
  const colors = new Float32Array(NUM_POINTS * 3);
  const sampleBuffer = new Float32Array(dim);
  const allSamples = new Float32Array(NUM_POINTS * dim);
  for (let i = 0; i < allSamples.length; i++) {
    allSamples[i] = randomNormal() * stdDev;
  }

  const tempQuat = new THREE.Quaternion();
  const tempEuler = new THREE.Euler();
  const tempVec = new THREE.Vector3();
  const tempMat3 = new THREE.Matrix3();
  const tempMat4 = new THREE.Matrix4();
  const v1 = new THREE.Vector3();
  const v2 = new THREE.Vector3();
  const v3 = new THREE.Vector3();

  for (let i = 0; i < NUM_POINTS; i++) {
    const offset = i * dim;
    for (let j = 0; j < dim; j++) {
      sampleBuffer[j] = Math.tanh(allSamples[offset + j]);
    }

    let axisAngle: THREE.Vector3;
    if (representation === 'tangent') {
      tempVec.set(sampleBuffer[0], sampleBuffer[1], sampleBuffer[2]);
      axisAngle = tempVec;
    } else if (representation === 'euler') {
      tempEuler.set(sampleBuffer[0], sampleBuffer[1], sampleBuffer[2], 'XYZ');
      tempQuat.setFromEuler(tempEuler);
      axisAngle = quatToRotvec(tempQuat);
    } else if (representation === 'quaternion') {
      tempQuat.set(sampleBuffer[0], sampleBuffer[1], sampleBuffer[2], sampleBuffer[3]);
      tempQuat.normalize();
      axisAngle = quatToRotvec(tempQuat);
    } else if (representation === 'matrix') {
      v1.set(sampleBuffer[0], sampleBuffer[3], sampleBuffer[6]).normalize();
      v2.set(sampleBuffer[1], sampleBuffer[4], sampleBuffer[7]);
      const dot = v2.dot(v1);
      v2.x -= v1.x * dot;
      v2.y -= v1.y * dot;
      v2.z -= v1.z * dot;
      v2.normalize();
      v3.crossVectors(v1, v2);
      tempMat3.set(
        v1.x, v2.x, v3.x,
        v1.y, v2.y, v3.y,
        v1.z, v2.z, v3.z,
      );
      tempMat4.setFromMatrix3(tempMat3);
      tempQuat.setFromRotationMatrix(tempMat4);
      axisAngle = quatToRotvec(tempQuat);
    } else {
      tempVec.set(0, 0, 0);
      axisAngle = tempVec;
    }

    const magnitude = Math.sqrt(axisAngle.x * axisAngle.x + axisAngle.y * axisAngle.y + axisAngle.z * axisAngle.z);
    if (magnitude > Math.PI) {
      const scale = (2 * Math.PI - magnitude) / magnitude;
      axisAngle.x *= -scale;
      axisAngle.y *= -scale;
      axisAngle.z *= -scale;
    }

    if (representation === 'quaternion' || representation === 'matrix') {
      const invPi = 1.0 / Math.PI;
      axisAngle.x *= invPi;
      axisAngle.y *= invPi;
      axisAngle.z *= invPi;
    } else if (representation === 'tangent') {
      const invSqrt3 = 1.0 / Math.sqrt(3);
      axisAngle.x *= invSqrt3;
      axisAngle.y *= invSqrt3;
      axisAngle.z *= invSqrt3;
    } else if (representation === 'euler') {
      const euler_cnst = 0.5;
      axisAngle.x *= euler_cnst;
      axisAngle.y *= euler_cnst;
      axisAngle.z *= euler_cnst;
    }

    const idx = i * 3;
    positions[idx] = axisAngle.x;
    positions[idx + 1] = axisAngle.y;
    positions[idx + 2] = axisAngle.z;

    const distance = Math.sqrt(axisAngle.x * axisAngle.x + axisAngle.y * axisAngle.y + axisAngle.z * axisAngle.z);
    const intensity = 0.5 + 0.5 * distance;
    colors[idx] = baseColor.r * intensity;
    colors[idx + 1] = baseColor.g * intensity;
    colors[idx + 2] = baseColor.b * intensity;
  }

  return { positions, colors };
}

// 3D Visualization for distribution warping (WebGL preferred, Canvas 2D fallback)
function createDistributionVisualization(containerId: string, representation: string) {
  const container = document.getElementById(containerId);
  if (!container) return;

  let renderer: THREE.WebGLRenderer;
  try {
    renderer = new THREE.WebGLRenderer({ antialias: true });
  } catch (err) {
    console.warn(`WebGL unavailable for ${containerId}, using Canvas 2D fallback:`, err);
    createCanvas2DDistributionVisualization(container, representation);
    return;
  }

  const width = container.clientWidth;
  const height = container.clientHeight;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf9fafb);

  const camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
  camera.position.set(1.05, 1.05, 1.05);

  renderer.setSize(width, height);
  container.appendChild(renderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;

  const sphereGeometry = new THREE.SphereGeometry(1, 32, 32);
  const sphereMaterial = new THREE.MeshBasicMaterial({
    color: 0x9ca3af,
    wireframe: true,
    transparent: true,
    opacity: 0.1,
  });
  scene.add(new THREE.Mesh(sphereGeometry, sphereMaterial));
  scene.add(new THREE.AxesHelper(1.2));

  let points: THREE.Points | null = null;
  function updatePoints(stdDev: number) {
    if (points) {
      scene.remove(points);
      points.geometry.dispose();
      (points.material as THREE.Material).dispose();
    }
    const { positions, colors } = generateDistributionPoints(representation, stdDev);
    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geom.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    const material = new THREE.PointsMaterial({
      size: 0.03,
      vertexColors: true,
      transparent: true,
      opacity: 0.8,
    });
    points = new THREE.Points(geom, material);
    scene.add(points);
  }
  updatePoints(0.3);

  const slider = document.getElementById(`slider-${representation}`) as HTMLInputElement | null;
  const stdLabel = document.getElementById(`std-${representation}`);
  if (slider && stdLabel) {
    slider.addEventListener('input', () => {
      const stdDev = parseFloat(slider.value);
      stdLabel.textContent = stdDev.toFixed(1);
      updatePoints(stdDev);
    });
  }

  function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  }
  animate();

  window.addEventListener('resize', () => {
    const newWidth = container.clientWidth;
    const newHeight = container.clientHeight;
    camera.aspect = newWidth / newHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(newWidth, newHeight);
  });
}

// Canvas 2D fallback: same data, projected and drawn by hand. No GPU required.
function createCanvas2DDistributionVisualization(container: HTMLElement, representation: string) {
  const canvas = document.createElement('canvas');
  canvas.style.display = 'block';
  canvas.style.width = '100%';
  canvas.style.height = '100%';
  canvas.style.touchAction = 'none';
  canvas.style.cursor = 'grab';
  container.appendChild(canvas);

  const ctx = canvas.getContext('2d');
  if (!ctx) {
    container.removeChild(canvas);
    const msg = document.createElement('div');
    msg.className = 'flex items-center justify-center h-full text-sm text-gray-500 px-4 text-center';
    msg.textContent = 'Canvas 2D unavailable in this browser.';
    container.appendChild(msg);
    return;
  }

  // Initial isometric-ish view to match the WebGL camera at (1.05, 1.05, 1.05).
  let theta = -Math.PI / 4;
  let phi = Math.atan(1 / Math.SQRT2);
  let data = generateDistributionPoints(representation, 0.3);

  const SEGS = 16;
  const sphereLines: Float32Array[] = [];
  for (let lat = 1; lat < SEGS; lat++) {
    const phiLat = (lat / SEGS) * Math.PI - Math.PI / 2;
    const r = Math.cos(phiLat);
    const yLat = Math.sin(phiLat);
    const ring = new Float32Array((SEGS + 1) * 3);
    for (let s = 0; s <= SEGS; s++) {
      const t = (s / SEGS) * Math.PI * 2;
      const idx = s * 3;
      ring[idx] = Math.cos(t) * r;
      ring[idx + 1] = yLat;
      ring[idx + 2] = Math.sin(t) * r;
    }
    sphereLines.push(ring);
  }
  for (let lon = 0; lon < SEGS; lon++) {
    const thetaLon = (lon / SEGS) * Math.PI * 2;
    const ring = new Float32Array((SEGS + 1) * 3);
    for (let s = 0; s <= SEGS; s++) {
      const phiLat = (s / SEGS) * Math.PI - Math.PI / 2;
      const r = Math.cos(phiLat);
      const yLat = Math.sin(phiLat);
      const idx = s * 3;
      ring[idx] = Math.cos(thetaLon) * r;
      ring[idx + 1] = yLat;
      ring[idx + 2] = Math.sin(thetaLon) * r;
    }
    sphereLines.push(ring);
  }

  let projXY: Float32Array | null = null;
  let projZ: Float32Array | null = null;
  let order: Uint32Array | null = null;

  function render() {
    const width = container.clientWidth;
    const height = container.clientHeight;
    if (width <= 0 || height <= 0) return;

    ctx!.fillStyle = '#f9fafb';
    ctx!.fillRect(0, 0, width, height);
    const cx = width / 2;
    const cy = height / 2;
    const scale = Math.min(width, height) * 0.42;

    const ct = Math.cos(theta), st = Math.sin(theta);
    const cp = Math.cos(phi), sp = Math.sin(phi);

    ctx!.strokeStyle = 'rgba(156,163,175,0.3)';
    ctx!.lineWidth = 0.5;
    for (const ring of sphereLines) {
      ctx!.beginPath();
      for (let i = 0; i < ring.length; i += 3) {
        const x = ring[i], y = ring[i + 1], z = ring[i + 2];
        const x1 = ct * x + st * z;
        const z1 = -st * x + ct * z;
        const sx = cx + x1 * scale;
        const sy = cy - (cp * y - sp * z1) * scale;
        if (i === 0) ctx!.moveTo(sx, sy);
        else ctx!.lineTo(sx, sy);
      }
      ctx!.stroke();
    }

    const axes: [number, number, number, string][] = [
      [1.2, 0, 0, '#ef4444'],
      [0, 1.2, 0, '#22c55e'],
      [0, 0, 1.2, '#3b82f6'],
    ];
    ctx!.lineWidth = 1.5;
    for (const [ax, ay, az, color] of axes) {
      const x1 = ct * ax + st * az;
      const z1 = -st * ax + ct * az;
      const sx = cx + x1 * scale;
      const sy = cy - (cp * ay - sp * z1) * scale;
      ctx!.strokeStyle = color;
      ctx!.beginPath();
      ctx!.moveTo(cx, cy);
      ctx!.lineTo(sx, sy);
      ctx!.stroke();
    }

    const N = data.positions.length / 3;
    if (!projXY || projXY.length !== N * 2) {
      projXY = new Float32Array(N * 2);
      projZ = new Float32Array(N);
      order = new Uint32Array(N);
    }
    for (let i = 0; i < N; i++) {
      const idx = i * 3;
      const x = data.positions[idx], y = data.positions[idx + 1], z = data.positions[idx + 2];
      const x1 = ct * x + st * z;
      const z1 = -st * x + ct * z;
      projXY[i * 2] = x1;
      projXY[i * 2 + 1] = cp * y - sp * z1;
      projZ![i] = sp * y + cp * z1;
      order![i] = i;
    }
    const orderArr = Array.from(order!);
    const z = projZ!;
    orderArr.sort((a, b) => z[a] - z[b]);

    const radius = 1.5;
    for (let k = 0; k < orderArr.length; k++) {
      const i = orderArr[k];
      const idx = i * 3;
      const px = cx + projXY[i * 2] * scale;
      const py = cy - projXY[i * 2 + 1] * scale;
      const r = (data.colors[idx] * 255) | 0;
      const g = (data.colors[idx + 1] * 255) | 0;
      const b = (data.colors[idx + 2] * 255) | 0;
      ctx!.fillStyle = `rgba(${r},${g},${b},0.8)`;
      ctx!.beginPath();
      ctx!.arc(px, py, radius, 0, Math.PI * 2);
      ctx!.fill();
    }
  }

  function resize() {
    const dpr = window.devicePixelRatio || 1;
    const width = container.clientWidth;
    const height = container.clientHeight;
    canvas.width = Math.max(1, Math.floor(width * dpr));
    canvas.height = Math.max(1, Math.floor(height * dpr));
    ctx!.setTransform(dpr, 0, 0, dpr, 0, 0);
    render();
  }

  let dragging = false;
  let lastX = 0, lastY = 0;
  canvas.addEventListener('pointerdown', e => {
    dragging = true;
    lastX = e.clientX;
    lastY = e.clientY;
    canvas.style.cursor = 'grabbing';
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener('pointermove', e => {
    if (!dragging) return;
    const dx = e.clientX - lastX;
    const dy = e.clientY - lastY;
    lastX = e.clientX;
    lastY = e.clientY;
    theta -= dx * 0.01;
    phi -= dy * 0.01;
    phi = Math.max(-Math.PI / 2 + 0.001, Math.min(Math.PI / 2 - 0.001, phi));
    render();
  });
  const endDrag = (e: PointerEvent) => {
    if (!dragging) return;
    dragging = false;
    canvas.style.cursor = 'grab';
    canvas.releasePointerCapture(e.pointerId);
  };
  canvas.addEventListener('pointerup', endDrag);
  canvas.addEventListener('pointercancel', endDrag);

  const slider = document.getElementById(`slider-${representation}`) as HTMLInputElement | null;
  const stdLabel = document.getElementById(`std-${representation}`);
  if (slider && stdLabel) {
    slider.addEventListener('input', () => {
      const stdDev = parseFloat(slider.value);
      stdLabel.textContent = stdDev.toFixed(1);
      data = generateDistributionPoints(representation, stdDev);
      render();
    });
  }

  window.addEventListener('resize', resize);
  resize();
}

// Initialize visualizations when DOM is loaded
function initializeVisualizations() {
  const visualizations = [
    { id: 'distribution-viz-tangent', rep: 'tangent' },
    { id: 'distribution-viz-matrix', rep: 'matrix' },
    { id: 'distribution-viz-quaternion', rep: 'quaternion' },
    { id: 'distribution-viz-euler', rep: 'euler' }
  ];

  visualizations.forEach(viz => {
    try {
      createDistributionVisualization(viz.id, viz.rep);
    } catch (err) {
      console.error(`Visualization "${viz.id}" failed to initialize:`, err);
      const container = document.getElementById(viz.id);
      if (container) {
        container.innerHTML = '';
        const msg = document.createElement('div');
        msg.className = 'flex items-center justify-center h-full text-sm text-gray-500 px-4 text-center';
        msg.textContent = '3D visualization unavailable (WebGL is disabled or unsupported in this browser).';
        container.appendChild(msg);
      }
    }
  });
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initializeVisualizations);
} else {
  initializeVisualizations();
}

console.log('SO(3) Primer website loaded');

// Benchmark switching functionality with slide animations
let currentBenchmark = 'trajectory'; // Track the currently active benchmark
let isAnimating = false; // Prevent multiple animations at once

(window as any).showBenchmark = function (benchmarkName: string) {
  // Prevent switching during animation
  if (isAnimating || benchmarkName === currentBenchmark) {
    return;
  }

  isAnimating = true;

  // Get the current and new content elements
  const currentContent = document.getElementById(`benchmark-${currentBenchmark}`);
  const newContent = document.getElementById(`benchmark-${benchmarkName}`);
  const wrapper = document.getElementById('benchmark-content-wrapper');

  if (!currentContent || !newContent || !wrapper) {
    isAnimating = false;
    return;
  }

  // STEP 1: Capture current wrapper height before any position changes
  const currentHeight = wrapper.offsetHeight;
  wrapper.style.height = `${currentHeight}px`;

  // Update button states immediately
  const allButtons = document.querySelectorAll('.benchmark-btn');
  allButtons.forEach(button => {
    button.classList.remove('active');
  });

  const selectedButton = document.querySelector(`[data-benchmark="${benchmarkName}"]`);
  if (selectedButton) {
    selectedButton.classList.add('active');
  }

  // Start the slide-out animation for current content
  currentContent.classList.remove('active');
  currentContent.classList.add('slide-out-left');

  // Start slide-in animation while slide-out is still in progress (overlapping for faster perceived transition)
  setTimeout(() => {
    // Remove slide-out animation and hide current content
    currentContent.classList.remove('slide-out-left');

    // Prepare new content for slide-in
    newContent.classList.add('slide-in-right');

    // Small delay to ensure the element is rendered
    requestAnimationFrame(() => {
      newContent.classList.add('active');

      // STEP 2: Measure new content height and transition to it
      requestAnimationFrame(() => {
        const newHeight = newContent.offsetHeight;
        wrapper.style.height = `${newHeight}px`;
      });
    });

    // After slide-in completes
    setTimeout(() => {
      // Clean up animations
      newContent.classList.remove('slide-in-right');

      // STEP 3: Reset to auto height for responsive behavior
      wrapper.style.height = 'auto';

      // Update current benchmark tracker
      currentBenchmark = benchmarkName;
      isAnimating = false;
    }, 250); // Match the animation duration
  }, 175); // Start slide-in early while slide-out is still in progress
};

// Initialize the first benchmark as active on page load
document.addEventListener('DOMContentLoaded', () => {
  const firstBenchmark = document.getElementById('benchmark-trajectory');
  if (firstBenchmark) {
    firstBenchmark.classList.add('active');
  }

  // Initialize the first distribution as active
  const firstDistribution = document.getElementById('distribution-tangent');
  if (firstDistribution) {
    firstDistribution.classList.add('active');
  }
});

// Distribution switching functionality with slide animations
let currentDistribution = 'tangent'; // Track the currently active distribution
let isAnimatingDistribution = false; // Prevent multiple animations at once

(window as any).showDistribution = function (distributionName: string) {
  // Prevent switching during animation
  if (isAnimatingDistribution || distributionName === currentDistribution) {
    return;
  }

  isAnimatingDistribution = true;

  // Get the current and new content elements
  const currentContent = document.getElementById(`distribution-${currentDistribution}`);
  const newContent = document.getElementById(`distribution-${distributionName}`);
  const wrapper = document.getElementById('distribution-content-wrapper');

  if (!currentContent || !newContent || !wrapper) {
    isAnimatingDistribution = false;
    return;
  }

  // STEP 1: Capture current wrapper height before any position changes
  const currentHeight = wrapper.offsetHeight;
  wrapper.style.height = `${currentHeight}px`;

  // Update button states immediately
  const allButtons = document.querySelectorAll('.distribution-btn');
  allButtons.forEach(button => {
    button.classList.remove('active');
  });

  const selectedButton = document.querySelector(`[data-distribution="${distributionName}"]`);
  if (selectedButton) {
    selectedButton.classList.add('active');
  }

  // Start the slide-out animation for current content
  currentContent.classList.remove('active');
  currentContent.classList.add('slide-out-left');

  // Start slide-in animation while slide-out is still in progress (overlapping for faster perceived transition)
  setTimeout(() => {
    // Remove slide-out animation and hide current content
    currentContent.classList.remove('slide-out-left');

    // Prepare new content for slide-in
    newContent.classList.add('slide-in-right');

    // Small delay to ensure the element is rendered
    requestAnimationFrame(() => {
      newContent.classList.add('active');

      // STEP 2: Measure new content height and transition to it
      requestAnimationFrame(() => {
        const newHeight = newContent.offsetHeight;
        wrapper.style.height = `${newHeight}px`;
      });
    });

    // After slide-in completes
    setTimeout(() => {
      // Clean up animations
      newContent.classList.remove('slide-in-right');

      // STEP 3: Reset to auto height for responsive behavior
      wrapper.style.height = 'auto';

      // Update current distribution tracker
      currentDistribution = distributionName;
      isAnimatingDistribution = false;
    }, 250); // Match the animation duration
  }, 125); // Start slide-in early while slide-out is still in progress (125ms overlap)
};

