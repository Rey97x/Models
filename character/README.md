# Nomade del deserto: rig 2D e animazioni

Rig a mesh deformabile (skinning lineare) costruito dalla foto `src/reference.jpg`.

| File | Contenuto |
|---|---|
| `index.html` | Visualizzatore three.js: 6 animazioni, vista scheletro, pesi e wireframe |
| `rig/rig.json` | Ossa (26), mesh (vertici, UV, triangoli), indici e pesi di skinning |
| `rig/texture.png` | Personaggio scontornato (RGBA) |
| `nomade_rig.glb` | Export glTF binario: SkinnedMesh + scheletro + 6 clip (Blender, Godot, Unity, Unreal) |

## Rig
- **Ossa**: root → hips → spine → chest → neck → head; braccia (upperarm/forearm/hand) L/R;
  gambe (thigh/shin/foot) L/R; nastri dei polsi (2 ossa per lato); fascia in vita (2 ossa); lembi della tunica L/R.
  `R` = destra del personaggio (sinistra dello schermo).
- **Pesi**: distanza geodesica calcolata dentro la silhouette, quindi gambe e braccia separate non si influenzano;
  massimo 4 influenze per vertice.
- **Gambe**: IK a due ossa con scorcio del ginocchio verso la camera; i piedi restano a terra.
- **Moto secondario**: i nastri seguono la gravità, fascia e tunica seguono il bacino.

## Animazioni (30 fps, in loop)
Respiro (4 s) · Saluto (2.4 s) · Guardia (1.6 s) · Attacco (1.4 s) · Marcia (1.1 s) · Salto (1.8 s)

## Rigenerare
```bash
pip install numpy pillow scipy
python3 tools/build_rig.py        # ricrea rig/rig.json e rig/texture.png
npx serve character               # apri http://localhost:3000
```
Nel browser, `await rigApp.exportGLB()` restituisce il `.glb` aggiornato.
