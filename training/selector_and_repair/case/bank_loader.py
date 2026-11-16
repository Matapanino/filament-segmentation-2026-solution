"""Pinned original bank and remaining-fold map loaders, transplanted unchanged."""
import glob,hashlib,json,tarfile
from pathlib import Path
import numpy as np,pandas as pd
from load_inputs import read,sha,unique,checked

def load_bank(inv):
 path=checked(unique('bank.bin','filament-matchedquality-input-1109'),inv['archive']);bank=Path('source');bank.mkdir(exist_ok=True)
 with tarfile.open(path) as tar:
  for name,row in inv['members'].items():
   assert not Path(name).is_absolute() and '..' not in Path(name).parts
   b=tar.extractfile(name).read();assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256'];dest=bank/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(b)
 tables=[];labels=[];records={};features=None;offset=0
 for fold in range(5):
  d=bank/f'bank/fold{fold}_s42/V00_R3';manifest=read(d/'manifest.json');assert len(manifest['base_features'])==39 and (features is None or features==manifest['base_features']);features=manifest['base_features'];t=pd.read_csv(d/'features.csv').set_index('candidate_id');t.index+=offset;l=pd.read_csv(d/'labels.csv');l.candidate_id+=offset;r=read(d/'iou_records.json')
  for f,v in r.items():assert f not in records;v['candidate_ids']=[i+offset for i in v['candidate_ids']];records[f]=v
  tables.append(t);labels.append(l);offset+=len(t)
 table=pd.concat(tables);labs=pd.concat(labels,ignore_index=True);files=sorted(records);assert len(files)==707 and len(table)==5148 and len(labs)==8402
 assert np.array_equal(table.index,np.arange(len(table))) and set(labs.candidate_id)<=set(table.index)
 return bank,table,labs,records,features,files

def native_original(cfg,pins,root,fold,view):
 declared=cfg['original_map_inventory'][f'fold{fold}_s42'][view];transport=pins['original_transport'][str(fold)][view];scope=cfg['original_kernel_sources'][str(fold)]
 if transport['mode']=='kernel':
  pattern=f'original_{"full_raw_1024" if view=="full" else "local_raw_clahe_detail"}_f{fold}_s42/oof_prob.npz';paths=[Path(p) for p in glob.glob('/kaggle/input/**/'+pattern,recursive=True) if scope in p];assert len(paths)==1;(p,)=paths;p=checked(p,declared['map']);mrow=declared['manifest'];direct=[Path(p) for p in glob.glob('/kaggle/input/**/'+mrow['member'],recursive=True) if scope in p]
  if direct:m=read(checked(direct[0],mrow));assert len(direct)==1
  else:
   archive=unique('diagnostics.tar.gz',scope)
   with tarfile.open(archive) as tar:payload=tar.extractfile(mrow['member']).read()
   assert len(payload)==mrow['bytes'] and hashlib.sha256(payload).hexdigest()==mrow['sha256'] and hashlib.md5(payload).hexdigest()==mrow['md5'];m=json.loads(payload)
 else:
  assert transport['mode']=='opaque';p=checked(root/transport['map_filename'],declared['map']);m=read(checked(root/transport['manifest_filename'],declared['manifest']))
 return dict(declared['map'],local=str(p),denominator=255),m
