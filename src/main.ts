import './style.css'
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

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

// 3D Visualization for distribution warping
function createDistributionVisualization(containerId: string, representation: string) {
  const container = document.getElementById(containerId);
  if (!container) return;

  const width = container.clientWidth;
  const height = container.clientHeight;

  // Scene setup
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf9fafb);

  // Camera
  const camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
  camera.position.set(1.05, 1.05, 1.05);

  // Renderer
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(width, height);
  container.appendChild(renderer.domElement);

  // Controls
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;

  // Add sphere wireframe to show the boundary
  const sphereGeometry = new THREE.SphereGeometry(1, 32, 32);
  const sphereMaterial = new THREE.MeshBasicMaterial({
    color: 0x9ca3af,
    wireframe: true,
    transparent: true,
    opacity: 0.1
  });
  const sphere = new THREE.Mesh(sphereGeometry, sphereMaterial);
  scene.add(sphere);

  // Add axes helper
  const axesHelper = new THREE.AxesHelper(1.2);
  scene.add(axesHelper);

  // Store points object for regeneration
  let points: THREE.Points | null = null;

  // Use representation-based colors
  const colorMap: Record<string, THREE.Color> = {
    'tangent': new THREE.Color(0x0E4DAB), // blue
    'matrix': new THREE.Color(0xF1750F),  // orange
    'quaternion': new THREE.Color(0xD426C2), // magenta
    'euler': new THREE.Color(0x26C706)    // green
  };
  const baseColor = colorMap[representation] || new THREE.Color(0x3b82f6);

  // Dimension mapping for each representation
  const dimensionMap: Record<string, number> = {
    'tangent': 3,
    'euler': 3,
    'quaternion': 4,
    'matrix': 9
  };
  const dim = dimensionMap[representation] || 3;

  // Function to generate points with given standard deviation
  function generatePoints(stdDev: number) {
    // Remove old points if they exist
    if (points) {
      scene.remove(points);
      points.geometry.dispose();
      (points.material as THREE.Material).dispose();
    }

    const numPoints = 3000;
    const pointsGeometry = new THREE.BufferGeometry();
    const positions = new Float32Array(numPoints * 3);
    const colors = new Float32Array(numPoints * 3);

    // Pre-allocate sample buffer for reuse
    const sampleBuffer = new Float32Array(dim);

    // Generate all random samples at once for better cache locality
    const totalSamples = numPoints * dim;
    const allSamples = new Float32Array(totalSamples);
    for (let i = 0; i < totalSamples; i++) {
      allSamples[i] = randomNormal() * stdDev;
    }

    // Reusable objects to avoid allocations
    const tempQuat = new THREE.Quaternion();
    const tempEuler = new THREE.Euler();
    const tempVec = new THREE.Vector3();
    const tempMat3 = new THREE.Matrix3();
    const tempMat4 = new THREE.Matrix4();
    const v1 = new THREE.Vector3();
    const v2 = new THREE.Vector3();
    const v3 = new THREE.Vector3();

    for (let i = 0; i < numPoints; i++) {
      // Extract samples for this point
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
        // Orthogonalize using Gram-Schmidt
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
          v1.z, v2.z, v3.z
        );

        tempMat4.setFromMatrix3(tempMat3);
        tempQuat.setFromRotationMatrix(tempMat4);
        axisAngle = quatToRotvec(tempQuat);
      } else {
        tempVec.set(0, 0, 0);
        axisAngle = tempVec;
      }

      // Clamp to unit sphere (axis-angle magnitudes > π wrap around)
      const magnitude = Math.sqrt(axisAngle.x * axisAngle.x + axisAngle.y * axisAngle.y + axisAngle.z * axisAngle.z);
      if (magnitude > Math.PI) {
        const scale = (2 * Math.PI - magnitude) / magnitude;
        axisAngle.x *= -scale;
        axisAngle.y *= -scale;
        axisAngle.z *= -scale;
      }

      // Normalize by π to map to unit sphere
      if (representation === 'quaternion' || representation === 'matrix') {
        const invPi = 1.0 / Math.PI;
        axisAngle.x *= invPi;
        axisAngle.y *= invPi;
        axisAngle.z *= invPi;
      }
      else if (representation === 'tangent') {
        const invPi = 1.0 / Math.sqrt(3);
        axisAngle.x *= invPi;
        axisAngle.y *= invPi;
        axisAngle.z *= invPi;
      }
      else if (representation === 'euler') {
        const euler_cnst = 0.5;
        axisAngle.x *= euler_cnst;
        axisAngle.y *= euler_cnst;
        axisAngle.z *= euler_cnst;
      }
      // Write directly to typed array
      const idx = i * 3;
      positions[idx] = axisAngle.x;
      positions[idx + 1] = axisAngle.y;
      positions[idx + 2] = axisAngle.z;

      // Calculate color - vectorized distance calculation
      const distance = Math.sqrt(axisAngle.x * axisAngle.x + axisAngle.y * axisAngle.y + axisAngle.z * axisAngle.z);
      const intensity = 0.5 + 0.5 * distance;
      colors[idx] = baseColor.r * intensity;
      colors[idx + 1] = baseColor.g * intensity;
      colors[idx + 2] = baseColor.b * intensity;
    }

    pointsGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    pointsGeometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const pointsMaterial = new THREE.PointsMaterial({
      size: 0.03,
      vertexColors: true,
      transparent: true,
      opacity: 0.8
    });

    points = new THREE.Points(pointsGeometry, pointsMaterial);
    scene.add(points);
  }

  // Generate initial points with std dev = 0.5
  generatePoints(0.5);

  // Set up slider
  const slider = document.getElementById(`slider-${representation}`) as HTMLInputElement;
  const stdLabel = document.getElementById(`std-${representation}`);

  if (slider && stdLabel) {
    slider.addEventListener('input', () => {
      const stdDev = parseFloat(slider.value);
      stdLabel.textContent = stdDev.toFixed(1);
      generatePoints(stdDev);
    });
  }

  // Animation loop
  function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  }
  animate();

  // Handle window resize
  function handleResize() {
    if (!container) return;
    const newWidth = container.clientWidth;
    const newHeight = container.clientHeight;
    camera.aspect = newWidth / newHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(newWidth, newHeight);
  }

  window.addEventListener('resize', handleResize);
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
    createDistributionVisualization(viz.id, viz.rep);
  });
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initializeVisualizations);
} else {
  initializeVisualizations();
}

console.log('SO(3) Primer website loaded');

