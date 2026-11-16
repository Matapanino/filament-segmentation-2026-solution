"""Content hashes and inspectable comparisons, independent of NPZ timestamps."""
import hashlib,json
from pathlib import Path
import numpy as np


def file_hash(path,algorithm='sha256'):
    h=hashlib.new(algorithm)
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def array_sha(value):
    a=np.ascontiguousarray(value)
    h=hashlib.sha256(json.dumps({'dtype':a.dtype.str,'shape':a.shape},sort_keys=True).encode())
    h.update(a.tobytes());return h.hexdigest()


def compare_arrays(actual,reference):
    a,b=np.asarray(actual),np.asarray(reference)
    row=dict(actual_sha256=array_sha(a),reference_sha256=array_sha(b),equal=bool(a.shape==b.shape and a.dtype==b.dtype and np.array_equal(a,b)))
    if a.shape==b.shape:
        delta=np.abs(a.astype(np.float64)-b.astype(np.float64));row.update(changed_pixels=int(np.count_nonzero(a!=b)),max_abs_difference=float(delta.max()) if delta.size else 0.,mean_abs_difference=float(delta.mean()) if delta.size else 0.)
    else:row.update(actual_shape=list(a.shape),reference_shape=list(b.shape))
    return row


def identity(path):
    p=Path(path);return dict(bytes=p.stat().st_size,sha256=file_hash(p),md5=file_hash(p,'md5'))


def verify(path,expected):
    actual=identity(path)
    for k in ('bytes','sha256','md5'):
        if k in expected and actual[k]!=expected[k]:raise ValueError('Artifact integrity failure: '+str(path)+' '+k)
    return actual
