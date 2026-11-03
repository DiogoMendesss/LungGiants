"""Inspect original instance masks and NPZ annotations on the same Z,Y,X grid."""

from pathlib import Path
import base64
import json

import numpy as np
import pandas as pd


def inspect_nodule_labels(mask_zyx, annotation_dir, case_id, label_offset, html_path=None):
    """Display an axial slider and return its HTML document plus an overlap table.

    Supply the ORIGINAL instance mask in annotation (Z,Y,X) coordinates.
    No resampling, reorientation, or label reassignment is performed here.
    Bbox upper endpoints are checked against each annotation's crop shape.
    """
    from IPython.display import IFrame, display

    mask = np.asarray(mask_zyx)
    if mask.ndim != 3 or not np.isfinite(mask).all():
        raise ValueError("Expected a finite 3D instance mask in (Z,Y,X) order.")
    if np.any(mask < 0) or np.any(mask != np.floor(mask)):
        raise ValueError("Mask labels must be nonnegative integers; 0 is background.")
    labels, counts = np.unique(mask, return_counts=True)
    label_counts = {int(label): int(count) for label, count in zip(labels, counts)}
    paths = sorted(Path(annotation_dir).glob(f"{case_id}_nodule*.npz"))
    if not paths:
        raise FileNotFoundError(f"No annotations for {case_id} in {annotation_dir}")

    annotations, rows = [], []
    for path in paths:
        index = int(path.stem.rsplit("_nodule", 1)[1])
        expected = index + label_offset
        if expected <= 0:
            raise ValueError(f"{path.name}: expected label must be positive.")
        with np.load(path, allow_pickle=False) as info:
            if str(info["patient_id"].item()) != case_id:
                raise ValueError(f"{path.name}: patient_id disagrees with filename.")
            raw_bbox = np.asarray(info["bbox"])
            if raw_bbox.shape != (3, 2) or not np.isfinite(raw_bbox).all():
                raise ValueError(f"{path.name}: bbox must have shape (3,2).")
            if np.any(raw_bbox != np.floor(raw_bbox)):
                raise ValueError(f"{path.name}: bbox must contain integer indices.")
            bbox = raw_bbox.astype(int)
            crop = np.asarray(info["mask"]) > 0
            if crop.ndim != 3 or not crop.any():
                raise ValueError(f"{path.name}: annotation mask must be nonempty and 3D.")
            if "original_volume_shape" in info:
                if tuple(info["original_volume_shape"]) != mask.shape:
                    raise ValueError(f"{path.name}: original_volume_shape != {mask.shape}.")
            start = bbox[:, 0]
            stop = start + np.asarray(crop.shape)
            exclusive = bbox[:, 1] == stop
            inclusive = bbox[:, 1] == stop - 1
            if not np.all(exclusive | inclusive):
                raise ValueError(
                    f"{path.name}: bbox {bbox.tolist()} does not fit crop shape "
                    f"{crop.shape}. Check bbox axis order/convention."
                )
            if np.any(start < 0) or np.any(stop > mask.shape):
                raise ValueError(f"{path.name}: bbox outside mask; check the original grid.")
            region = mask[tuple(slice(int(a), int(b)) for a, b in zip(start, stop))]
            overlapping, overlap_counts = np.unique(region[crop], return_counts=True)
            overlaps = {int(k): int(v) for k, v in zip(overlapping, overlap_counts) if k > 0}
            intersection = overlaps.get(expected, 0)
            denominator = int(crop.sum()) + int(label_counts.get(expected, 0))
            rows.append({
                "annotation": path.name,
                "expected_label": expected,
                "overlapping_labels_voxels": overlaps,
                "annotation_voxels": int(crop.sum()),
                "expected_label_voxels": int(label_counts.get(expected, 0)),
                "dice_with_expected_label": 2 * intersection / denominator,
                "bbox_start_zyx": tuple(int(v) for v in start),
                "bbox_stop_zyx_exclusive": tuple(int(v) for v in stop),
            })
            annotations.append({"index": index, "expected": expected,
                                "start": start.tolist(), "stop": stop.tolist()})

    annotations.sort(key=lambda item: item["index"])
    table = pd.DataFrame(rows).sort_values("expected_label").reset_index(drop=True)
    foreground_ids = sorted(k for k in label_counts if k > 0)
    color_ids = sorted(set(foreground_ids) | {a["expected"] for a in annotations})
    palette = ["#56b4e9", "#e69f00", "#009e73", "#cc79a7", "#f0e442",
               "#d55e00", "#0072b2", "#ffffff"]
    colors = {label: palette[i % len(palette)] for i, label in enumerate(color_ids)}
    first = annotations[0]
    initial_z = int((first["start"][0] + first["stop"][0] - 1) // 2)
    # Encode only foreground voxels, keeping the standalone viewer small.
    slices = {}
    for z in range(mask.shape[0]):
        yy, xx = np.nonzero(mask[z])
        if len(xx):
            slices[z] = np.column_stack((xx, yy, mask[z, yy, xx])).astype(int).tolist()
    payload = json.dumps({"case": case_id, "shape": list(mask.shape), "colors": colors,
                          "labels": foreground_ids, "annotations": annotations,
                          "slices": slices, "initial": initial_z}).replace("<", "\\u003c")
    document = _VIEWER_HTML.replace("__PAYLOAD__", payload)
    if html_path is not None:
        html_path = Path(html_path)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_text(document, encoding="utf-8")
    print(f"{case_id}: mask shape (Z,Y,X)={mask.shape}; labels={foreground_ids}")
    print("Dice=1 confirms identical voxels for the proposed label on this grid.")
    display(table)
    encoded = base64.b64encode(document.encode("utf-8")).decode("ascii")
    display(IFrame(src=f"data:text/html;base64,{encoded}", width="100%", height=850))
    return document, table


_VIEWER_HTML = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><title>Nodule label check</title>
<style>
body {font:14px system-ui,sans-serif; margin:16px; color:#222; max-width:1000px}
h2 {margin:0 0 6px} p {margin:8px 0} input[type=range] {width:min(65vw,600px)}
canvas {width:min(100%,700px); height:auto; background:#111; display:block; margin-top:10px}
select,button {padding:5px; margin:4px} #legend span {display:inline-block;margin:4px 12px 4px 0}
#visible {min-height:20px} .swatch {display:inline-block;width:12px;height:12px;margin-right:4px}
</style>
<h2 id="title"></h2>
<p>Filled regions show actual mask labels. Dashed boxes come from NPZ annotations.<br>
Box colours use the proposed mapping; label text reports the actual mask value.</p>
<label>Jump to annotation <select id="jump"></select></label>
<label><input id="zoom" type="checkbox"> Zoom to nodules</label><br>
<button id="prev" aria-label="Previous slice">&#8592;</button>
<input id="slice" aria-label="Axial slice" type="range" min="0" step="1">
<button id="next" aria-label="Next slice">&#8594;</button> <output id="position"></output>
<div id="legend"></div><div id="visible"></div>
<canvas id="canvas" width="1000" height="1000" aria-label="Axial mask and annotation boxes"></canvas>
<p>Coordinates: original annotation grid, x left to right and y top to bottom; slice index z is zero-based.</p>
<script>
const d=__PAYLOAD__;
const slider=document.getElementById('slice'), jump=document.getElementById('jump');
const canvas=document.getElementById('canvas'), ctx=canvas.getContext('2d');
const zoom=document.getElementById('zoom');
document.title=d.case+' nodule labels'; document.getElementById('title').textContent=d.case;
slider.max=d.shape[0]-1; slider.value=d.initial;
for(const a of d.annotations) {
  const o=document.createElement('option'); o.value=a.index;
  o.textContent=`nodule${a.index} -> expected label ${a.expected}`; jump.appendChild(o);
}
for(const k of d.labels) {
  const s=document.createElement('span'), sw=document.createElement('i');
  sw.className='swatch'; sw.style.background=d.colors[k]; s.appendChild(sw);
  s.appendChild(document.createTextNode('Mask label '+k)); document.getElementById('legend').appendChild(s);
}
const pixels=document.createElement('canvas'); pixels.width=d.shape[2]; pixels.height=d.shape[1];
const pc=pixels.getContext('2d');
const bounds=[d.shape[2],d.shape[1],0,0];
for(const a of d.annotations) {
  bounds[0]=Math.min(bounds[0],a.start[2]); bounds[1]=Math.min(bounds[1],a.start[1]);
  bounds[2]=Math.max(bounds[2],a.stop[2]); bounds[3]=Math.max(bounds[3],a.stop[1]);
}
for(const points of Object.values(d.slices)) for(const [x,y] of points) {
  bounds[0]=Math.min(bounds[0],x); bounds[1]=Math.min(bounds[1],y);
  bounds[2]=Math.max(bounds[2],x+1); bounds[3]=Math.max(bounds[3],y+1);
}
function draw() {
  const z=Number(slider.value), points=d.slices[z]||[];
  document.getElementById('position').textContent=`z = ${z} / ${d.shape[0]-1}`;
  pc.fillStyle='#111'; pc.fillRect(0,0,pixels.width,pixels.height);
  const centres={};
  for(const [x,y,k] of points) {
    pc.fillStyle=d.colors[k]; pc.fillRect(x,y,1,1);
    const c=centres[k]||(centres[k]=[0,0,0]); c[0]+=x+0.5; c[1]+=y+0.5; c[2]++;
  }
  const x0=zoom.checked?Math.max(0,bounds[0]-25):0;
  const y0=zoom.checked?Math.max(0,bounds[1]-25):0;
  const w=(zoom.checked?Math.min(pixels.width,bounds[2]+25):pixels.width)-x0;
  const h=(zoom.checked?Math.min(pixels.height,bounds[3]+25):pixels.height)-y0;
  canvas.height=Math.round(canvas.width*h/w); const scale=canvas.width/w;
  ctx.imageSmoothingEnabled=false; ctx.drawImage(pixels,x0,y0,w,h,0,0,canvas.width,canvas.height);
  function text(s,x,y,color) {
    ctx.font='17px system-ui'; ctx.fillStyle='#111';
    const tw=ctx.measureText(s).width; x=Math.max(2,Math.min(x,canvas.width-tw-6));
    y=Math.max(20,Math.min(y,canvas.height-3)); ctx.fillRect(x-2,y-17,tw+5,21);
    ctx.fillStyle=color; ctx.fillText(s,x,y);
  }
  const visible=d.annotations.filter(a=>a.start[0]<=z&&z<a.stop[0]);
  for(const a of visible) {
    const x=(a.start[2]-x0)*scale, y=(a.start[1]-y0)*scale;
    ctx.strokeStyle=d.colors[a.expected]; ctx.lineWidth=2; ctx.setLineDash([7,4]);
    ctx.strokeRect(x,y,(a.stop[2]-a.start[2])*scale,(a.stop[1]-a.start[1])*scale);
    ctx.setLineDash([]); text(`nodule${a.index} -> ${a.expected}`,x,y-6,d.colors[a.expected]);
  }
  for(const [k,c] of Object.entries(centres))
    text('label '+k,(c[0]/c[2]-x0)*scale,(c[1]/c[2]-y0)*scale+22,'white');
  document.getElementById('visible').textContent='Annotations on this slice: '+
    (visible.map(a=>'nodule'+a.index).join(', ')||'none');
}
slider.addEventListener('input',draw); zoom.addEventListener('change',draw);
jump.addEventListener('change',()=>{const a=d.annotations.find(a=>a.index===Number(jump.value));
  slider.value=Math.floor((a.start[0]+a.stop[0]-1)/2); draw();});
for(const [id,step] of [['prev',-1],['next',1]]) document.getElementById(id).onclick=()=>{
  slider.value=Math.max(0,Math.min(Number(slider.max),Number(slider.value)+step)); draw();};
draw();
</script></html>'''
