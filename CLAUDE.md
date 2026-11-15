# Models: rigging e animazione di personaggi 3D

Questo repository contiene personaggi 3D e la pipeline che li rigga e li anima. Quando l'utente manda un nuovo
modello 3D statico (di solito un `.glb` di Tripo, anche con estensione strana tipo `DOC-...-WA0000._`) e chiede di
riggarlo / animarlo, **riusa la pipeline `tools/rig3d/`**: non riscrivere niente da zero e non lanciare gruppi di agenti.
L'utente scrive in italiano, vuole risultati rapidi e i file da scaricare.

## Procedura per un nuovo personaggio (circa 1 minuto di calcolo + controllo visivo)
1. Prepara: `pip install numpy scipy pillow bpy` e `(cd tools/rig3d && npm install)` (il container è nuovo a ogni sessione).
2. Crea `<nome>/src/` e copiaci il `.glb` (controlla con `file` che sia glTF binario; un solo `.glb` nella cartella).
3. `python3 tools/rig3d/pipeline.py <nome> --title "Nome visibile"`
   - Senza `rig/skeleton.json` scrive una **prima stima** dello scheletro (template del primo personaggio scalato e
     riadattato alla mesh) più `checks/skeleton_overlay.png`, poi si ferma.
   - **Guarda l'overlay** (anche con zoom: `preview_skeleton.py ... --zoom 2.5 --center x,y,z`) e correggi a mano
     `rig/skeleton.json`: spalle dentro la testa dell'omero, gomiti e ginocchia sulla piega, polsi nel punto più
     stretto, anche sotto i vestiti, piedi (tallone a terra, punta, pianta). La stima è stata provata solo sul primo
     personaggio: su un altro va sempre verificata.
   - Se ha tessuti che pendono (nastri ai polsi, fascia in vita) aggiungi le catene facoltative
     `Left/RightRibbon1`, `Ribbon2`, `Ribbon_End` e `Sash1`, `Sash2`, `Sash_End`; senza, quei tessuti seguono
     braccio / bacino / coscia.
4. Rilancia lo stesso comando: pesi, GLB, versione web, FBX, visualizzatore e tavole in `checks/`. Tutti i controlli
   devono passare (validatore glTF, `audit_feet.py`, `check_anim.py`, import in Blender).
5. **Guarda le tavole** `checks/*.png` (pesi, pose estreme, idle, camminata, giro): punte nella mesh, tessuti
   attraversati dalle gambe, braccia dentro il corpo. Solo dopo consegna.
6. Consegna: invia `<nome>_rigged.glb` e `.fbx` con SendUserFile, pubblica `<nome>/index.html` come Artifact con i file
   di `<nome>/web/` (vedi sotto), commit e push sul ramo della sessione.

## Cosa produce
Scheletro in stile Mixamo (23 ossa + 6 facoltative per i tessuti), pesi con binding geodetico su voxel, 7 clip a
30 fps: `Idle`, `Walk` (sul posto, `extras.speed`), `Walk_RootMotion`, `Turn360_Left/Right` (giro completo con passi,
rotazione sull'osso Root), `TurnLoop_Left/Right` (sul posto, `extras.turnRate`). I piedi sono guidati da IK su impronte
fisse: lo scivolamento in appoggio deve restare sotto 0,0005 unità.
Dettagli tecnici e convenzioni: `tools/rig3d/CONTRACT.md`. Esempio completo: `character3d/` (README incluso).

## Cose imparate (non ripetere gli errori)
- Requisiti della mesh: guscio unico, posa ad A o T, guarda +Z, piedi su y = 0, alta circa 1 unità (le costanti delle
  animazioni sono in unità modello). Se è diversa, riscalala / spostala prima.
- Gli Artifact non servono file `.glb` e bloccano i `fetch` di URI `data:` e `blob:`. Il visualizzatore usa
  `web/<nome>_web.gltf.json` + texture `.jpg` e decodifica il buffer nella pagina: pubblica con
  `files: {"web/<nome>_web.gltf.json": ..., "web/<nome>_basecolor.jpg": ..., "web/<nome>_normal.jpg": ..., "web/<nome>_rm.jpg": ...}`.
- Controllo dello scivolamento: il punto della pianta appartiene all'osso `ToeBase` (non a `Foot`) e si misura solo tra
  i fotogrammi chiave in cui il punto è a terra; atterraggio e stacco non sono scivolamento.
- Tessuti: niente collisori a capsula per la fascia (il pendolo passa dal lato sbagliato della coscia e ci resta);
  `cloth_fix.sash_follow_thigh` la spinge in avanti con la coscia sinistra.
- Gruppi di agenti / workflow: lenti e soggetti al limite d'uso (una volta si sono fermati a metà). Lavora in linea
  con la pipeline; usa agenti solo se l'utente li chiede.
- Dopo ogni modifica a `tools/rig3d/` verifica che `character3d` si ricostruisca identico:
  `python3 tools/rig3d/build.py --out /tmp/x.glb && cmp /tmp/x.glb character3d/nomade3d_rigged.glb`.
- Il personaggio 2D in `character/` è un rig a ritagli separato (`tools/build_rig.py`), non c'entra con `rig3d`.
