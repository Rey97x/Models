/*
 * Headless three.js renderer for GLB files (static, skinned, animated).
 * Usually driven by render.py; direct use:   node render.cjs job.json
 *
 * job = { glb, outDir, size:[w,h], zoom, center:[x,y,z], skeleton, grid, mode:"pbr"|"weights"|"flat"|"wire",
 *         follow, shots:[ {file, clip, time, view, pose} ] }
 *   view : "front" | "back" | "left" | "right" | "top" | "q" (3/4 front) | "qb" (3/4 back) | "az:el" degrees
 *          ("front" looks at the model from +Z, "left" from the character's left = +X)
 *   pose : { jointName: [ex,ey,ez] }  extra local euler rotation in degrees (XYZ order) applied on top of the clip/rest
 *   time : seconds into `clip`
 */
const { chromium } = require('playwright');
const path = require('path');
const fs = require('fs');

const job = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const NM = path.join(__dirname, 'node_modules', 'three');

const PAGE = `<!doctype html><html><head><meta charset="utf-8">
<script type="importmap">{"imports":{"three":"http://local/three/build/three.module.js","three/addons/":"http://local/three/examples/jsm/"}}</script>
</head><body style="margin:0;background:#fff"><script type="module">
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
const job = window.__job;
const [W, H] = job.size;
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setSize(W, H); renderer.setPixelRatio(1);
renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.0;
document.body.appendChild(renderer.domElement);
const scene = new THREE.Scene(); scene.background = new THREE.Color(0xd9dde2);
const pm = new THREE.PMREMGenerator(renderer);
scene.environment = pm.fromScene(new RoomEnvironment(), 0.04).texture;
const sun = new THREE.DirectionalLight(0xffffff, 1.6); sun.position.set(1.5, 3, 2.5); scene.add(sun);
const gltf = await new GLTFLoader().loadAsync('http://local/model.glb');
scene.add(gltf.scene);
let skinned = null; const meshes = [];
gltf.scene.traverse(o => { if (o.isMesh) { meshes.push(o); o.frustumCulled = false; if (o.isSkinnedMesh) skinned = o; } });
// bind-pose bounding box from raw geometry (independent of skinning state)
gltf.scene.updateMatrixWorld(true);
const box = new THREE.Box3();
for (const m of meshes) { m.geometry.computeBoundingBox(); const b = m.geometry.boundingBox.clone(); if (!m.isSkinnedMesh) b.applyMatrix4(m.matrixWorld); box.union(b); }
const size = box.getSize(new THREE.Vector3());
const center = job.center ? new THREE.Vector3(...job.center) : new THREE.Vector3(0, (box.min.y + box.max.y) / 2, 0);
const viewH = size.y * 1.25 / (job.zoom || 1);
const cam = new THREE.OrthographicCamera(-viewH * W / H / 2, viewH * W / H / 2, viewH / 2, -viewH / 2, -50, 50);
if (job.grid) {
  const g = new THREE.GridHelper(4, 40, 0x334499, 0x8a93a6); g.position.y = 0.0005; scene.add(g);  // 0.1-unit squares
  const ax = new THREE.AxesHelper(0.25); ax.position.y = 0.001; scene.add(ax);                    // X red, Y green, Z blue
}
// materials
const mode = job.mode || 'pbr';
if (mode !== 'pbr') for (const m of meshes) {
  if (mode === 'weights' && m.isSkinnedMesh) {
    const g = m.geometry, n = g.attributes.position.count, si = g.attributes.skinIndex, sw = g.attributes.skinWeight;
    const nb = m.skeleton.bones.length, col = new Float32Array(n * 3), c = new THREE.Color();
    const only = job.weightJoint ? m.skeleton.bones.findIndex(b => b.name === job.weightJoint) : -1;
    for (let i = 0; i < n; i++) for (let k = 0; k < 4; k++) {
      const j = si.getComponent(i, k), w = sw.getComponent(i, k);
      if (only >= 0) { if (j === only) { col[i*3] += w; col[i*3+1] += w * 0.25; col[i*3+2] += 0.15 * w; } }
      else { c.setHSL((j * 0.618034) % 1, 0.8, 0.5); col[i*3] += c.r * w; col[i*3+1] += c.g * w; col[i*3+2] += c.b * w; }
    }
    if (only >= 0) for (let i = 0; i < n; i++) { col[i*3+2] += 0.25; col[i*3] += 0.08; col[i*3+1] += 0.08; }
    g.setAttribute('color', new THREE.BufferAttribute(col, 3));
    m.material = new THREE.MeshBasicMaterial({ vertexColors: true });
  } else if (mode === 'wire') {
    m.material = new THREE.MeshBasicMaterial({ color: 0x223344, wireframe: true });
  } else {
    m.material = new THREE.MeshStandardMaterial({ color: 0xb9b2a6, roughness: 0.85, metalness: 0 });
  }
}
let helper = null, dots = null;
if (job.skeleton && skinned) {
  helper = new THREE.SkeletonHelper(skinned.skeleton.bones[0]);
  helper.material.depthTest = false; helper.material.vertexColors = false; helper.material.color.set(0x1133ff); helper.renderOrder = 10;
  const dg = new THREE.BufferGeometry();
  dg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(skinned.skeleton.bones.length * 3), 3));
  dots = new THREE.Points(dg, new THREE.PointsMaterial({ size: 5, sizeAttenuation: false, depthTest: false, color: 0xff2200 }));
  dots.renderOrder = 11; scene.add(helper, dots);
}
const mixer = new THREE.AnimationMixer(gltf.scene);
const clips = Object.fromEntries(gltf.animations.map(c => [c.name, c]));
const rest = new Map();
gltf.scene.traverse(o => rest.set(o, { p: o.position.clone(), q: o.quaternion.clone(), s: o.scale.clone() }));
const byName = {}; gltf.scene.traverse(o => { if (o.name) byName[o.name] = o; });
const VIEWS = { front: [0, 0], back: [180, 0], left: [90, 0], right: [-90, 0], top: [0, 89.9], q: [35, 15], qb: [145, 15], ql: [-35, 15] };
const v3 = new THREE.Vector3();
window.__info = { clips: gltf.animations.map(c => ({ name: c.name, duration: c.duration, tracks: c.tracks.length })),
  bones: skinned ? skinned.skeleton.bones.map(b => b.name) : [], bbox: [box.min.toArray(), box.max.toArray()] };
window.__shot = (s) => {
  mixer.stopAllAction();
  for (const [o, r] of rest) { o.position.copy(r.p); o.quaternion.copy(r.q); o.scale.copy(r.s); }
  if (s.clip) {
    const c = clips[s.clip]; if (!c) throw new Error('no clip ' + s.clip + ' (have: ' + Object.keys(clips).join(', ') + ')');
    const a = mixer.clipAction(c); a.reset(); a.setLoop(THREE.LoopRepeat, Infinity); a.play(); mixer.setTime(Math.min(s.time || 0, c.duration - 1e-5));
  }
  if (s.pose) for (const [n, e] of Object.entries(s.pose)) {
    const o = byName[n]; if (!o) throw new Error('no node ' + n);
    o.quaternion.multiply(new THREE.Quaternion().setFromEuler(new THREE.Euler(e[0] * Math.PI / 180, e[1] * Math.PI / 180, e[2] * Math.PI / 180, 'XYZ')));
  }
  scene.updateMatrixWorld(true);
  let [az, el] = VIEWS[s.view] || (s.view || '0:0').split(':').map(Number);
  az *= Math.PI / 180; el *= Math.PI / 180;
  const c = center.clone();
  if (job.follow && byName[job.follow]) { byName[job.follow].getWorldPosition(v3); c.x = v3.x; c.z = v3.z; }
  cam.position.set(c.x + 10 * Math.sin(az) * Math.cos(el), c.y + 10 * Math.sin(el), c.z + 10 * Math.cos(az) * Math.cos(el));
  cam.up.set(0, 1, 0); cam.lookAt(c); cam.updateMatrixWorld(true);
  if (dots) { const a = dots.geometry.attributes.position; skinned.skeleton.bones.forEach((b, i) => { b.getWorldPosition(v3); a.setXYZ(i, v3.x, v3.y, v3.z); }); a.needsUpdate = true; }
  renderer.render(scene, cam);
  return renderer.domElement.toDataURL('image/png');
};
window.__ready = true;
</script></body></html>`;

(async () => {
  const browser = await chromium.launch({ args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
  const page = await browser.newPage({ viewport: { width: job.size[0], height: job.size[1] } });
  const errors = [];
  page.on('pageerror', e => errors.push(String(e.message || e)));
  page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
  await page.route('**/*', async r => {
    const u = new URL(r.request().url());
    if (u.hostname !== 'local') return r.abort();
    if (u.pathname === '/index.html') return r.fulfill({ contentType: 'text/html', body: PAGE });
    if (u.pathname === '/model.glb') return r.fulfill({ path: path.resolve(job.glb), contentType: 'model/gltf-binary' });
    if (u.pathname.startsWith('/three/')) return r.fulfill({ path: path.join(NM, u.pathname.slice(7)), contentType: 'text/javascript' });
    return r.abort();
  });
  await page.addInitScript(j => { window.__job = j; }, job);
  await page.goto('http://local/index.html');
  try {
    await page.waitForFunction('window.__ready', null, { timeout: 180000 });
  } catch (e) { console.error('load failed', errors.join('\n')); await browser.close(); process.exit(2); }
  const info = await page.evaluate('window.__info');
  fs.mkdirSync(job.outDir, { recursive: true });
  fs.writeFileSync(path.join(job.outDir, 'info.json'), JSON.stringify(info, null, 1));
  for (const s of job.shots || []) {
    const url = await page.evaluate(s2 => window.__shot(s2), s);
    fs.writeFileSync(path.join(job.outDir, s.file), Buffer.from(url.split(',')[1], 'base64'));
  }
  if (errors.length) console.error('page errors:\n' + errors.join('\n'));
  await browser.close();
})();
