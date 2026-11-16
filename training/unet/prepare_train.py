"""Prepare ephemeral GPU-job input cache, preserving original annotation entries."""
import glob
import hashlib
import json
from pathlib import Path
import sys
import cv2
import numpy as np
from pycocotools import mask as mu


def main():
    assert Path("/kaggle/input").is_dir()
    protocol=json.loads(Path("protocol.json").read_text());wave=json.loads(Path("wave.json").read_text())
    lib=glob.glob("/kaggle/input/**/filseg/__init__.py",recursive=True);assert len(lib)==1
    sys.path.insert(0,str(Path(lib[0]).parents[1]))
    from filseg.coco_io import load_coco,annotation_sets
    from filseg.guard import TEST_FILENAMES,assert_clean
    paths=glob.glob("/kaggle/input/**/MAGFiLO_1.0_Annotations_kaggle2026_train.json",recursive=True);assert len(paths)==1
    annotation=Path(paths[0]);groups=annotation_sets(load_coco(annotation))
    assert set(groups)==set(protocol["training_files"])|set(protocol["validation_files"])
    assert_clean(protocol["training_files"]+protocol["validation_files"],"Relabel training/validation")
    corrections=[];digest=None
    if wave["arm"]=="corrected":
        raw=Path("corrections.json").read_bytes();digest=hashlib.sha256(raw).hexdigest()
        payload=json.loads(raw);assert payload["status"]=="frozen" and payload["fold"]==protocol["fold"]
        corrections=payload["corrections"];assert corrections
    by_file={r["file"]:r for r in corrections};assert len(by_file)==len(corrections)
    assert set(by_file)<=set(protocol["training_files"])
    assert not set(by_file)&set(protocol["validation_files"])
    cache=Path("/tmp/relabel-pilot-cache");cache.mkdir(exist_ok=True)
    max_annotations=0;core_pixels={}
    for number,f in enumerate(sorted(groups)):
        image=cv2.imread(str(annotation.parents[1]/"train/train_images"/(f+".jpeg")),cv2.IMREAD_GRAYSCALE)
        assert image is not None and image.shape==(2048,2048)
        masks=[]
        for entry in groups[f]:
            m=np.zeros((2048,2048),np.uint8)
            for a in entry.annotations:m |= mu.decode(mu.merge(mu.frPyObjects(a["segmentation"],2048,2048)))
            masks.append(m)
        assert masks;max_annotations=max(max_annotations,len(masks))
        np.save(cache/(f+"_image.npy"),image);np.save(cache/(f+"_mask.npy"),np.stack(masks))
        if f in by_file:
            record=by_file[f];core=np.zeros((2048,2048),np.uint8)
            for line in record["native_polylines"]:
                points=np.asarray(line,dtype=float);assert points.ndim==2 and points.shape[1]==2 and len(points)>=2
                assert np.isfinite(points).all() and ((points>=0)&(points<2048)).all()
                cv2.polylines(core,[np.rint(points).astype(np.int32)],False,1,thickness=protocol["core_line_width_native_pixels"])
            assert 0<int(core.sum())<2048*2048*.01,"Correction core area exceeds conservative pilot bound"
            np.save(cache/(f+"_core.npy"),core);core_pixels[f]=int(core.sum())
        if (number+1)%100==0:print(json.dumps(dict(cached=number+1,total=len(groups))),flush=True)
    manifest=dict(root=str(cache),max_annotations=max_annotations,test_files=sorted(TEST_FILENAMES),
        test_root=str(annotation.parents[1]/"test/test_images"),n_corrected_images=len(by_file),
        corrections_sha256=digest,core_pixels=core_pixels)
    Path("cache_manifest.json").write_text(json.dumps(manifest,indent=2))


if __name__=="__main__":main()
