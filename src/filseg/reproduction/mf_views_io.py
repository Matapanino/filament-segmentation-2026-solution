"""Streaming per-view archive writer (no torch): filseg's save_npz member layout (members '<key>.npy', fixed timestamps,
mode 0600), keys written in sorted order, deflate level 1 (the archives hold (8, 2048, 2048) uint16 arrays per image)."""
import io, zipfile
import numpy as np


class ViewsWriter:
    def __init__(self, path):
        self.z = zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED, allowZip64=True); self.last = None; self.keys = []

    def write(self, key, array):
        assert self.last is None or key > self.last, ('KEYS_NOT_SORTED', self.last, key); self.last = key; self.keys.append(key)
        buffer = io.BytesIO(); np.lib.format.write_array(buffer, np.asanyarray(array), allow_pickle=False)
        info = zipfile.ZipInfo(key + '.npy', date_time=(1980, 1, 1, 0, 0, 0)); info.compress_type = zipfile.ZIP_DEFLATED; info.external_attr = 0o600 << 16
        self.z.writestr(info, buffer.getvalue(), compresslevel=1)

    def close(self): self.z.close()
