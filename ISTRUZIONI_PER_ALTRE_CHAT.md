# Istruzioni per riggare e animare un personaggio 3D con la pipeline esistente

Da incollare in una nuova chat di Claude Code insieme al modello 3D da riggare.

---

Devi riggare e animare il modello 3D che ti allego usando una pipeline che esiste già. Non riscrivere niente da zero
e non lanciare gruppi di agenti: è tutto pronto e provato su due personaggi.

## Dove si trova
- Repository GitHub: `Rey97x/Models` (ramo predefinito `claude/rigging-animazione-personaggio-2qly8l`).
- Se il repository non è già in questa sessione, aggiungilo e clonalo. Senza il repository e senza la possibilità di
  eseguire comandi la pipeline non si può usare: in quel caso dimmelo subito invece di improvvisare.
- Leggi per primo `CLAUDE.md` nella radice: contiene la procedura completa e gli errori da non ripetere.
  I dettagli tecnici sono in `tools/rig3d/CONTRACT.md`. Esempi finiti: `character3d/` (nomade) e `cavaliere3d/` (cavaliere).

## Cosa deve avere il modello
File `.glb` statico (di solito esportato da Tripo, a volte con estensione strana tipo `DOC-...-WA0000._`): umanoide,
guscio unico, in posa ad A o a T, rivolto verso +Z, piedi a y = 0, alto circa 1 unità. Se è diverso va riscalato o
spostato prima.

## Procedura
1. Prepara l'ambiente: `pip install numpy scipy pillow bpy` e `(cd tools/rig3d && npm install)`.
2. Crea la cartella del personaggio e copiaci il modello: `<nome>/src/<modello>.glb` (un solo `.glb`).
3. Lancia `python3 tools/rig3d/pipeline.py <nome> --title "Nome visibile"`.
   La prima volta scrive una stima dello scheletro in `<nome>/rig/skeleton.json` e un'immagine di controllo, poi si ferma.
4. Controlla lo scheletro a occhio e correggilo a mano nel file:
   - `python3 tools/rig3d/preview_skeleton.py <nome>/rig/skeleton.json out.png --glb <nome>/src/<modello>.glb --views front,left,back --zoom 2.2 --center 0,0.7,0 --mode pbr`
   - `python3 tools/rig3d/slice_view.py <nome>/src/<modello>.glb <nome>/rig/skeleton.json out.png` (sezioni orizzontali:
     il modo più rapido per centrare le ossa e distinguere il corpo da mantelli e gonne)
   - Punti da guardare: spalle dentro la spalla, gomiti e ginocchia sulla piega, polsi nel punto più stretto,
     anche sotto i vestiti, caviglie sull'asse dello stinco, tallone a terra. Ginocchiere e armature spostano il centro.
5. Se il personaggio ha tessuti che pendono, aggiungi a mano le catene facoltative nello stesso file
   (tre punti ciascuna: `...1`, `...2`, `..._End`):
   - `LeftRibbon` / `RightRibbon`: strisce che pendono dai polsi
   - `Sash`: stoffa che pende dalla cintura (se è al centro fa da pannello frontale)
   - `Cape`: mantello sulla schiena (misura la linea centrale del mantello a x = 0)
6. Rilancia lo stesso comando del punto 3. Produce pesi, modello completo, versione leggera, FBX, visualizzatore e
   tavole di verifica in `<nome>/checks/`. Devono passare tutti i controlli: validatore glTF, `audit_feet.py`
   (scivolamento dei piedi), `check_anim.py`, import in Blender.
7. Guarda le tavole in `<nome>/checks/` prima di consegnare: pesi, pose estreme, idle, camminata, giro.
   Se c'è un mantello controlla anche la camminata con ingrandimento su ginocchia e orlo:
   `python3 tools/rig3d/render.py <nome>/<nome>_rigged.glb out.png --clip Walk --times 0,0.28,0.57,0.85 --views ql,left,240:10 --zoom 2.4 --center=-0.03,0.2,-0.03`
8. Consegna: inviami `<nome>_rigged.glb` e `<nome>_rigged.fbx` come file da scaricare, pubblica il visualizzatore
   `<nome>/index.html` con i file della cartella `<nome>/web/`, poi fai commit e push.

## Armi e combattimento (facoltativo)
Per dare un'arma al personaggio e aggiungere attacco e schivata, crea `<nome>/rig/character.json` e rilancia la pipeline:
`{"weightClass": "heavy", "weapons": [{"hand": "Right", "type": "sword"}]}` per un personaggio pesante con spada, oppure
`{"weightClass": "light", "weapons": [{"hand": "Right", "type": "dagger"}, {"hand": "Left", "type": "dagger"}]}`
per uno leggero con due pugnali. Si aggiungono le clip `Attack`, `Dodge_Back` e `Dash_Forward`: il personaggio pesante
attacca con un colpo unico e schiva corto e lento, quello leggero colpisce due volte e schiva lungo con un balzo.
Controlla le tavole `checks/attack.png`, `dodge_back.png` e `dash_forward.png`.

## Cosa ottengo
- Scheletro a 23 ossa con nomi in stile Mixamo, più le ossa dei tessuti se presenti.
- 7 animazioni a 30 fps: `Idle`, `Walk` (sul posto, 0,674 unità/s), `Walk_RootMotion`, `Turn360_Left` e `Turn360_Right`
  (giro completo con i passi, rotazione dentro la clip), `TurnLoop_Left` e `TurnLoop_Right` (passi del giro senza
  rotazione, da ruotare via codice a 122,7 °/s).
- Piedi ancorati a terra: lo scivolamento in appoggio deve restare sotto 0,0005 unità.

## Errori già incontrati, da non ripetere
- Il visualizzatore pubblicato non può caricare file `.glb` né URI `data:`: usa i file di `<nome>/web/`
  (`<nome>_web.gltf.json` più le tre texture `.jpg`).
- Mantelli: i bordi sono spesso fusi con fianchi, cosce e ginocchiere. Se nella camminata un pezzo si allunga tra
  ginocchio e mantello, è quello. La pipeline lo gestisce già (riconosce i pezzi d'armatura dal colore e taglia la mesh
  nei punti di fusione), ma va verificato a occhio.
- Fascia o pannello alla cintura: viene spinto in avanti dalla coscia; non usare collisori a capsula.
- Dopo qualsiasi modifica agli strumenti in `tools/rig3d/`, verifica che il primo personaggio si ricostruisca identico:
  `python3 tools/rig3d/build.py --out /tmp/x.glb && cmp /tmp/x.glb character3d/nomade3d_rigged.glb`.
- Se impari qualcosa di nuovo su un personaggio diverso, aggiungilo a `CLAUDE.md` e fai push.

## Limiti da dirmi con chiarezza
- Le dita non sono articolate (un osso per mano).
- Il movimento viene controllato su fotogrammi fissi e con misure numeriche, non in un motore di gioco.
- La stima automatica dello scheletro è stata provata solo su due personaggi: il controllo visivo è obbligatorio.
