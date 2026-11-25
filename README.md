# Scultera

An interactive 3D model of a sad leopard gecko, with step-by-step notes on how it was sculpted in Blender using code.

**Live:** https://linglab.org/scultera

## What's here

- `index.html`: the viewer and process write-up (three.js, vendored in `vendor/`)
- `models/`: Draco-compressed glTF stages (`base`, `grab`, `crease`, `final`) and the blockout primitives (`prims.json`)
- `renders/`: Cycles stills
- `sad_gecko.blend`: the full Blender 5.3 scene (lights, cameras, materials)
- `process/`: the pipeline
  - `build_sdf.py`: signed-distance blockout, eye sockets and lids, marching cubes
  - `sculpt.py`: brush engine (Draw, Crease, Grab, Smooth, Flatten), tubercles, vertex paint
  - `scene.py`: materials, eyes, tear, lights, cameras → `.blend`
  - `render.py` / `comp.py`: Cycles render and backdrop composite
  - `export_web.py`: iris bake and glTF export (then `gltf-transform draco`)

## Rebuild

```bash
pip install bpy==5.3.0 numpy scipy scikit-image
cd process
python3 build_sdf.py && python3 sculpt.py && python3 scene.py
python3 render.py -- Hero hero.png 1.0 96 && python3 comp.py hero.png hero_final.png
```

The scripts write to `/home/claude/gecko/`; change the paths at the top of each file for your machine.
