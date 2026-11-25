import json, subprocess, os
L = json.load(open('log.json')); LOG = L['log']; nb = L['blk_frames']; snaps = L['snaps']
DUR = {'blockout': 0.6, 'mesh': 1.0, 'sculpt': 0.8, 'detail': 0.85, 'finish': 1.0}
frames = []
for i in range(nb):
    frames.append((f'render/tl_{i:03d}.png', DUR['blockout'], [i]))
prev = nb - 1
for j, (fn, topo, li) in enumerate(snaps):
    idxs = list(range(prev + 1, li + 1)) or [li]
    frames.append((f'render/tl_{nb + j:03d}.png', DUR.get(LOG[li]['phase'], 0.8), idxs)); prev = li
extra = [('render/lit_0.png', 1.2, dict(phase='light', tool='Material', target='clay', note='the finished sculpt, still in grey clay', params={})),
         ('render/lit_1.png', 1.6, dict(phase='light', tool='Material', target='wax + glaze', note='warm translucent wax, blue glazed dish', params={'wax_subsurface': 0.55})),
         ('render/lit_2.png', 2.2, dict(phase='light', tool='Light', target='flames + smoke', note='wicks, glowing flames, and smoke ribbons like the pencil lines', params={'flames': 4}))]
entries = []; lines = []; t = 0.0
for png, d, idxs in frames:
    for k, li in enumerate(idxs):
        e = dict(LOG[li]); e['t'] = round(t + d * k / max(1, len(idxs)), 2); entries.append(e)
    lines.append(f"file '{os.path.abspath(png)}'\nduration {d}"); t += d
for png, d, e in extra:
    if os.path.exists(png):
        e = dict(e); e['t'] = round(t, 2); entries.append(e)
        lines.append(f"file '{os.path.abspath(png)}'\nduration {d}"); t += d
lines.append(f"file '{os.path.abspath(extra[-1][0])}'")
open('concat.txt', 'w').write('\n'.join(lines) + '\n')
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', 'concat.txt', '-vf', 'fps=20,scale=480:600,format=yuv420p',
                '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '21', '-movflags', '+faststart', 'sculpt.mp4'], check=True)
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', 'render/finale.png', '-vf', 'scale=900:-1', '-q:v', '3', 'finale.jpg'], check=True)
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', 'render/tl_000.png', '-q:v', '4', 'poster.jpg'], check=True)
mesh = next(e for e in LOG if e['tool'] == 'Voxel remesh')['params']
json.dump(dict(entries=entries, duration=round(t, 2), candles=L['candles'], verts=mesh['verts']), open('timeline.json', 'w'), indent=1)
print('video', round(t, 1), 's, entries', len(entries))
