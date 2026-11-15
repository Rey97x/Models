/*
 * Headless smoke test of character3d/index.html.
 *   node tools/rig3d/test_viewer.cjs <glb to serve as nomade3d_web.glb> <out dir> [clipA,clipB,...]
 * Serves the page from character3d/, maps the CDN imports to tools/rig3d/node_modules, plays each clip for a moment
 * and writes desktop + phone screenshots.  Prints console errors and the mover position / yaw after each clip.
 */
const { chromium } = require('playwright');
const path = require('path'), fs = require('fs');
const [glb, outDir, clipArg] = process.argv.slice(2);
const ROOT = path.resolve(process.env.CHAR_DIR || path.join(__dirname, '..', '..', 'character3d'));   // CHAR_DIR=<character dir> to test another one
const NM = path.join(__dirname, 'node_modules', 'three');
(async () => {
  fs.mkdirSync(outDir, { recursive: true });
  const browser = await chromium.launch({ args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
  let failed = false;
  for (const [tag, vp] of [['desktop', { width: 1280, height: 820 }], ['phone', { width: 400, height: 860 }]]) {
    const page = await browser.newPage({ viewport: vp, deviceScaleFactor: 1 });
    page.on('pageerror', e => { failed = true; console.log(`[${tag}] PAGE ERROR`, e.message); });
    page.on('console', m => { if (m.type() === 'error' && !/Failed to load resource/.test(m.text())) { failed = true; console.log(`[${tag}] console error`, m.text()); } });
    await page.route('**/*', async r => {
      const u = new URL(r.request().url());
      if (u.hostname === 'cdn.jsdelivr.net') return r.fulfill({ path: path.join(NM, u.pathname.replace('/npm/three@0.160.0/', '')), contentType: 'text/javascript' });
      if (u.hostname === 'local') {
        if (glb && glb !== '-' && u.pathname === '/nomade3d_web.glb') return r.fulfill({ path: path.resolve(glb), contentType: 'model/gltf-binary' });
        // emulate a strict host: only same-origin fetches, scripts from the page + jsDelivr, no data:/blob: fetches
        if (u.pathname.endsWith('.html')) return r.fulfill({ path: path.join(ROOT, u.pathname), contentType: 'text/html',
          headers: { 'Content-Security-Policy': "default-src 'none'; script-src 'unsafe-inline' https://cdn.jsdelivr.net; style-src 'unsafe-inline'; img-src 'self' data: blob:; connect-src 'self'; font-src https://fonts.gstatic.com" } });
        return r.fulfill({ path: path.join(ROOT, u.pathname) });
      }
      return r.abort();
    });
    await page.goto('http://local/index.html');
    await page.waitForFunction(() => window.viewer && window.viewer.ready, null, { timeout: 240000 });
    const clips = clipArg ? clipArg.split(',') : await page.evaluate(() => window.viewer.recs.map(r => r.id));
    await page.waitForTimeout(1200);
    await page.screenshot({ path: path.join(outDir, `${tag}_start.png`) });
    if (tag === 'desktop') for (const c of clips) {
      await page.evaluate(id => window.viewer.play(id), c);
      await page.waitForTimeout(2500);
      const st = await page.evaluate(() => ({ x: +window.viewer.mover.position.x.toFixed(3), z: +window.viewer.mover.position.z.toFixed(3), yawDeg: +(window.viewer.mover.rotation.y * 180 / Math.PI).toFixed(1) }));
      console.log(`[${tag}] ${c}: mover`, JSON.stringify(st));
      await page.locator('#stage').screenshot({ path: path.join(outDir, `${tag}_${c}.png`) });
    }
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
    console.log(`[${tag}] horizontal overflow px:`, overflow);
    await page.close();
  }
  await browser.close();
  process.exit(failed ? 1 : 0);
})();
