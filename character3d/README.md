# Nomade 3D: rig, skinning e animazioni

Modello sorgente: `src/desert_assassin_tripo.glb` (Tripo, statico, 63.492 vertici, 112.781 triangoli, texture PBR 4K).

| File | Contenuto |
|---|---|
| `nomade3d_rigged.glb` | Modello completo: scheletro a 29 ossa, pesi di skinning, 7 animazioni, texture 4K originali |
| `nomade3d_web.glb` | Stesso modello con texture 2K (5,8 MB) per web e mobile |
| `nomade3d_rigged.fbx` | Export FBX fatto con Blender (per Unity / Unreal) |
| `index.html`, `web/` | Visualizzatore three.js con griglia a terra e orme; `web/` contiene lo stesso modello leggero come glTF JSON + texture .jpg |
| `rig/skeleton.json`, `rig/weights.npz` | Posizioni delle articolazioni e pesi (dati intermedi) |

## Scheletro
`Root > Hips > Spine > Spine1 > Spine2 > Neck > Head`, braccia `Shoulder > Arm > ForeArm > Hand`,
gambe `UpLeg > Leg > Foot > ToeBase` (nomi in stile Mixamo), più ossa per i tessuti: due per ogni nastro dei polsi
(`Left/RightRibbon1-2`) e due per la fascia in vita (`Sash1-2`).
+Y in alto, il personaggio guarda +Z, la sua sinistra è +X. Tutte le ossa hanno rotazione di riposo identità.
Il modello mantiene la scala di Tripo: è alto 0,979 unità (1 unità ≈ 1,8 m di una persona reale).
Le dita non sono articolate (un osso per mano).

## Animazioni (30 fps)
| Clip | Durata | Tipo | Come usarla |
|---|---|---|---|
| `Idle` | 5,00 s | loop | In piedi: respiro, piccoli spostamenti di peso; i piedi non si muovono |
| `Walk` | 1,13 s | loop sul posto | Muovi il personaggio in avanti a **0,674 unità/s** (falcata 0,764): a quella velocità i piedi non scivolano |
| `Walk_RootMotion` | 1,13 s | loop con root motion | Lo stesso passo; l'osso `Root` avanza di 0,764 unità a ciclo lungo +Z |
| `Turn360_Left` / `Turn360_Right` | 3,33 s | singola, root motion | Giro completo in 8 passi; la rotazione 0 → ±360° è sull'osso `Root`; parte e finisce nella posa di `Idle` |
| `TurnLoop_Left` / `TurnLoop_Right` | 0,73 s | loop sul posto | Passi del giro senza rotazione: ruota il personaggio via codice a **122,7 °/s** (90° a ciclo) |

Ogni clip porta negli `extras` glTF `loop`, `rootMotion`, `speed` / `strideLength`, `turnRate` e `footsteps`
(istanti di appoggio dei piedi, utili per i suoni dei passi).

## Piedi ancorati
I piedi sono guidati da cinematica inversa su impronte fisse. `tools/rig3d/audit_feet.py` rilegge il GLB e misura di
quanto si sposta ogni punto della suola (tallone, pianta, punta) mentre è a terra, anche tra un fotogramma e l'altro:

| Clip | Spostamento massimo in appoggio |
|---|---|
| Idle | 0,00000 |
| Walk, Walk_RootMotion | 0,00015 unità (≈ 0,3 mm reali) |
| Turn360, TurnLoop | 0,00030 unità (≈ 0,5 mm reali) |

Un secondo controllo indipendente (`tools/rig3d/check_anim.py`) verifica anche penetrazione nel terreno,
ginocchia, continuità dei loop e coerenza del root motion: tutte le clip passano.

## Rigenerare
```bash
pip install numpy scipy pillow bpy          # bpy serve solo per il controllo Blender / FBX
(cd tools/rig3d && npm install)             # three.js e validatore glTF per i render di verifica
python3 tools/rig3d/skin_weights.py         # pesi  -> character3d/rig/weights.npz
python3 tools/rig3d/build.py                # GLB   -> character3d/nomade3d_rigged.glb
python3 tools/rig3d/make_web.py character3d/nomade3d_rigged.glb character3d/nomade3d_web.glb
python3 tools/rig3d/make_gltf_json.py character3d/nomade3d_web.glb character3d/web   # file per il visualizzatore
node tools/rig3d/validate.cjs character3d/nomade3d_rigged.glb
python3 tools/rig3d/audit_feet.py character3d/nomade3d_rigged.glb
python3 tools/rig3d/check_anim.py character3d/nomade3d_rigged.glb
python3 tools/rig3d/blender_check.py character3d/nomade3d_rigged.glb --fbx character3d/nomade3d_rigged.fbx
python3 tools/rig3d/render.py character3d/nomade3d_rigged.glb walk.png --clip Walk --frames 8 --views left,front --grid
```
Convenzioni e formati intermedi: `tools/rig3d/CONTRACT.md`.
