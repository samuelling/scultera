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

## Sam, take 2 (`sam.html#take2`)

A second bust of Sam built from one 12-second phone clip instead of nine photos. Scripts in `process/sam2/`, run in order:

1. `track.py`: MediaPipe face landmarks, blendshapes and hair/skin segmentation on every frame
2. `shape.py` → `poses.py` → `depthcheck.py`: averaged rest face, per-frame perspective pose, depth-scale test
3. `face_mesh.py` → `rmap.py`: symmetry plane, rest eyes/lips, Loop-subdivided face, radial face map
4. `fitskull.py`: skull ellipsoid fitted to the head outline in 30 frames
5. `build.py <voxel_cm> [stage]`: one distance field (face + skull + jaw, neck, bust, ears, light details), marching cubes
6. `finish.py`: eyes, mouth slit, teeth, hairline, buzz cut, T-shirt
7. `anim.py`: vertex ← landmark transfer matrix, PCA motion shapes, head/gaze/jaw curves
8. `paint.py`: video colour projected from 179 posed, expression-matched frames; de-lit and filled
9. `export_glb.py` (then `gltf-transform meshopt`), `bl_timelapse.py`, `timeline2.py`

The clip itself is not in the repo.
