/* glTF validator wrapper:  node tools/rig3d/validate.cjs file.glb   (exit code 1 on errors) */
const fs = require('fs');
const validator = require('gltf-validator');
const file = process.argv[2];
validator.validateBytes(new Uint8Array(fs.readFileSync(file)), { maxIssues: 200 }).then(r => {
  const i = r.issues;
  console.log(`errors ${i.numErrors}  warnings ${i.numWarnings}  infos ${i.numInfos}  hints ${i.numHints}`);
  const seen = {};
  for (const m of i.messages) { const k = m.code; seen[k] = (seen[k] || 0) + 1; if (seen[k] <= 3) console.log(` [${['ERROR','WARN','INFO','HINT'][m.severity]}] ${m.code} ${m.pointer || ''}: ${m.message}`); }
  for (const [k, n] of Object.entries(seen)) if (n > 3) console.log(`  ... ${k} x${n}`);
  console.log(JSON.stringify(r.info, null, 0).slice(0, 600));
  process.exit(i.numErrors ? 1 : 0);
}).catch(e => { console.error(e); process.exit(2); });
