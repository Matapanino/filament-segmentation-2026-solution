"""Synthetic contracts; no local model execution or competition-image decoding."""
import hashlib
import numpy as np
import pytest
from filseg.reproduction.arithmetic import mf_mean, candidate_blend
from filseg.reproduction.audit import array_sha, compare_arrays
from filseg.reproduction.preprocess import input_view, boxes, crop_resize, restore_tiles
from filseg.reproduction.csv_io import write_rows, read_rows


def test_mf_native_integer_sum_and_order():
    rng=np.random.default_rng(814)
    a,b,c=[rng.integers(0,511,(32,32),dtype=np.uint16) for _ in range(3)]
    full,local=[rng.integers(0,256,(32,32),dtype=np.uint8) for _ in range(2)]
    mf=mf_mean(a,b,c)
    assert np.array_equal(mf,(a+b+c).astype(np.float32)/np.float32(1530))
    expected=sum(np.float32(w)*x for w,x in [(0.25,mf),(0.375,full.astype(np.float32)/255),(0.375,local.astype(np.float32)/255)])
    assert array_sha(candidate_blend(full,local,a,b,c))==array_sha(expected)
    with pytest.raises(ValueError):mf_mean(a.astype(np.uint8),b,c)


def test_preprocessing_geometry_and_reconstruction():
    raw=np.arange(64*64,dtype=np.uint8).reshape(64,64)
    x=input_view(raw,dict(channels='repeat',transform='raw'))
    assert np.array_equal(x[:,:,0],raw.astype(np.float32)/255)
    assert np.array_equal(x[:,:,0],x[:,:,2])
    bs=boxes(64,32)
    result=restore_tiles([np.full((16,16),.25,np.float32) for _ in bs],bs,64)
    assert np.array_equal(result,np.full((64,64),.25,np.float32))
    assert crop_resize(x,(0,0,64,64),32).shape==(32,32,3)


def test_content_hash_detects_dtype_shape_and_one_pixel():
    a=np.zeros((2,4),np.uint16);b=a.copy();b[1,2]=1
    assert array_sha(a)!=array_sha(a.reshape(4,2))
    assert array_sha(a)!=array_sha(a.astype(np.uint8))
    r=compare_arrays(a,b)
    assert not r['equal'] and r['changed_pixels']==1 and r['max_abs_difference']==1


def test_csv_kept_order_and_crlf(tmp_path):
    p=tmp_path/'s.csv';write_rows(p,['a','b'],{'a':['x','y','z'],'b':[]},{'a':[True,False,True],'b':[]})
    assert p.read_bytes()==b'filament_id,segmentation_rle\r\na_1,x\r\na_2,z\r\n'
    assert read_rows(p)==[('a_1','x'),('a_2','z')]


def test_frozen_shipping_modules_have_exact_source_hashes():
    import json
    from pathlib import Path
    import filseg.reproduction as package
    root=Path(package.__file__).parent
    manifest=json.loads((root/'source_manifest.json').read_text())
    for name,row in manifest.items():
        if row.get('byte_identical'):
            assert hashlib.sha256((root/name).read_bytes()).hexdigest()==row['sha256']


def test_ship_r3_synthetic_nonoverlap_and_small_object_rejection():
    from filseg.reproduction.instancing import iter_instances,SHIP_R3
    p=np.zeros((2048,2048),np.float32)
    p[200:220,200:300]=.75;p[800:820,800:900]=.8;p[1200:1202,1200:1202]=1
    masks=list(iter_instances(p,SHIP_R3))
    assert len(masks)==2
    occupied=np.zeros_like(p,dtype=bool)
    for m in masks:
        m=m.astype(bool);assert m.sum()>=256 and not (occupied&m).any();occupied|=m


def test_instance_diagnostics_are_order_independent_and_count_missing_masks():
    from filseg.reproduction.pipeline import compare_instances
    from pycocotools import mask as mu
    masks=[]
    for offset in (10,100):
        a=np.zeros((2048,2048),np.uint8);a[offset:offset+10,offset:offset+20]=1
        masks.append(mu.encode(np.asfortranarray(a))['counts'].decode())
    equal=compare_instances(masks[::-1],masks)
    assert equal['exact_instances']==2 and equal['foreground_IoU']==1
    assert equal['assigned_IoU_with_unmatched_zero']==1
    missing=compare_instances(masks[:1],masks)
    assert missing['exact_reference_fraction']==.5 and missing['unmatched_reference']==1
    assert missing['assigned_IoU_with_unmatched_zero']==.5
    assert compare_instances([],[])['foreground_IoU']==1


def test_artifact_integrity_rejects_changed_bytes(tmp_path):
    from filseg.reproduction.audit import identity,verify
    p=tmp_path/'artifact';p.write_bytes(b'frozen');pin=identity(p);verify(p,pin)
    p.write_bytes(b'change')
    with pytest.raises(ValueError):verify(p,pin)


def test_feature_and_saved_lr_stages_on_synthetic_mask():
    from filseg.reproduction.features import features_for_mask
    from filseg.reproduction.quality import apply_quality
    raw=np.full((2048,2048),128,np.uint8);mask=np.zeros(raw.shape,bool);mask[200:220,200:300]=True
    full=np.full(raw.shape,191,np.uint8);local=np.full(raw.shape,127,np.uint8)
    probs={'full':full.astype(np.float32)/255,'local':local.astype(np.float32)/255}
    probs['blend']=(probs['full']+probs['local'])*.5
    row=features_for_mask(mask,probs,raw)
    assert len(row)==39 and row['log_area']==np.log1p(2000)
    assert row['bbox_width']==100/2048 and row['image_mean']==pytest.approx(128/255)
    assert row['disagreement_mean']==pytest.approx(64/255)
    model=dict(feature_order=list(row),scaler_mean=[0.]*39,scaler_scale=[1.]*39,coef=[[0.]*39],intercept=[0.],threshold=.35)
    rows,x,score,keep=apply_quality([mask],full,local,raw,model)
    assert x.shape==(1,39) and score.tolist()==[.5] and keep.tolist()==[True]
    assert apply_quality([],full,local,raw,model)[1].shape==(0,39)
