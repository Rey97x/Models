# Cavaliere 3D: rig, skinning e animazioni

Modello sorgente: `src/ornate_knight_tripo.glb` (Tripo, statico, 65.410 vertici, 112.378 triangoli, texture PBR 4K).
Riggato con la pipeline `tools/rig3d/pipeline.py` (vedi `CLAUDE.md` nella radice e `character3d/README.md`).

| File | Contenuto |
|---|---|
| `cavaliere3d_rigged.glb` | Modello completo: 27 ossa, pesi di skinning, 7 animazioni, texture 4K |
| `cavaliere3d_web.glb` | Stesso modello con texture 2K |
| `cavaliere3d_rigged.fbx` | Export FBX fatto con Blender |
| `index.html`, `web/` | Visualizzatore three.js |
| `rig/skeleton.json`, `rig/weights.npz` | Articolazioni e pesi |
| `checks/` | Tavole di verifica (scheletro, sezioni, pesi, pose estreme, idle, camminata, giro) |

Scheletro: le 23 ossa del corpo più due catene per i tessuti: `Cape1-2` (mantello sulla schiena) e `Sash1-2`
(pannello di stoffa che pende dalla cintura, tra le gambe). Il mantello pende per gravità, segue la rotazione del busto
e viene spinto indietro da ginocchia, polpacci e talloni; il pannello frontale viene spinto in avanti dalla coscia che avanza.
Dove il mantello era fuso con le ginocchiere e i fianchi la mesh è stata tagliata (78 vertici duplicati), così il
mantello scorre sulle gambe invece di stirarsi.
Le ali dell'elmo, gli spallacci e le piastre dell'armatura seguono le ossa del corpo (nessun osso dedicato).

Clip, velocità e uso sono gli stessi del primo personaggio: `Walk` a 0,674 unità/s, `TurnLoop` a 122,7 °/s.
Scivolamento massimo dei piedi in appoggio: 0,0003 unità.
