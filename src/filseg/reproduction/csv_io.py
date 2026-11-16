"""Byte-compatible competition CSV formatting."""
import csv

def write_rows(path, files, per_file_rles, keep_per_file):
    """Write the competition CSV: one row per kept mask, ids '<file>_<k>' numbered 1.. over kept masks."""
    counts = {}
    with open(path, 'w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['filament_id', 'segmentation_rle'])
        for f in files:
            assert len(per_file_rles[f]) == len(keep_per_file[f])
            k = 0
            for rle, keep in zip(per_file_rles[f], keep_per_file[f]):
                if keep:
                    k += 1
                    writer.writerow([f'{f}_{k}', rle])
            counts[f] = k
    return counts

def read_rows(path):
    with open(path, newline='') as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ['filament_id', 'segmentation_rle']
        return [(r['filament_id'], r['segmentation_rle']) for r in reader]
