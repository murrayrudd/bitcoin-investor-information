"""Record immutable source/model hashes before generating validation data."""
import argparse,hashlib,json,datetime
from pathlib import Path

def freeze(root):
    root=Path(root)
    paths=sorted(Path('pilot').glob('*.py'))+[Path('pilot/_excess.cpp'),Path('EXPANSION_PROTOCOL.md')]+sorted((root/'models').glob('*'))
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    if len(list((root/'models').glob('bank_*_O*.npz')))!=8:raise RuntimeError('All eight models must be fitted before freeze')
    target=root/'source_and_model_freeze.json'
    if target.exists():
        old=json.loads(target.read_text())
        if old['sha256']!=hashes:raise RuntimeError('Frozen source or model has changed')
        return old
    if list((root/'evaluation').glob('truth_*.npz')):raise RuntimeError('Validation already started before source freeze')
    record=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),sha256=hashes)
    target.write_text(json.dumps(record,indent=2)+'\n');return record

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();print(json.dumps(freeze(a.root),indent=2))
