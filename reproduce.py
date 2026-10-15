#!/usr/bin/env python3
"""Reproduction entry point. The archived scientific source is never edited.

Run ``python reproduce.py --help``. Results are written under work/, which is
excluded from Git. Release assets are verified before extraction.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / 'release-assets.json'
ENV = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
           MKL_NUM_THREADS='1', MPLBACKEND='Agg')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def source_check():
    expected = read(ROOT / 'provenance/source-sha256.json')
    for name, digest in expected.items():
        p = ROOT / name
        if not p.is_file() or sha256(p) != digest:
            raise RuntimeError(f'Scientific source differs from the release: {name}')
    print(f'PASS: {len(expected)} scientific source and test files match their release hashes.', flush=True)


def asset_path(item, supplied=None):
    candidates = ([Path(supplied)] if supplied else []) + [ROOT / 'assets', ROOT.parent / 'release-assets']
    for folder in candidates:
        p = folder / item['name']
        if p.is_file():
            if p.stat().st_size != item['size_bytes'] or sha256(p) != item['sha256']:
                raise RuntimeError(f'Asset size or SHA-256 mismatch: {p}')
            return p
    raise FileNotFoundError(
        f"Missing {item['name']}. Download it from this repository's v1.0.0 Release "
        "into assets/, or use --assets-dir PATH. The ZIP stays compressed there.")


def selected_assets(roles, supplied=None):
    return [(a, asset_path(a, supplied)) for a in read(MANIFEST)['assets'] if a['role'] in roles]


def unpack(path, destination, item, selection=None):
    """Reject unsafe paths, duplicates, links, and unexpected archive dimensions."""
    destination = destination.resolve()
    with zipfile.ZipFile(path) as z:
        members = z.infolist()
        if len(members) != item['members'] or sum(m.file_size for m in members) != item['uncompressed_bytes']:
            raise RuntimeError('Archive dimensions disagree with the release manifest')
        seen = set()
        for m in members:
            n = PurePosixPath(m.filename)
            if (n.is_absolute() or '..' in n.parts or '\\' in m.filename
                    or ':' in m.filename or m.filename in seen
                    or (m.external_attr >> 16) & 0o170000 == 0o120000):
                raise RuntimeError(f'Unsafe archive member: {m.filename}')
            seen.add(m.filename)
            if selection is not None and not selection(m.filename):
                continue
            target = destination.joinpath(*n.parts)
            if not target.resolve().is_relative_to(destination):
                raise RuntimeError('Archive path escapes the work directory')
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(m) as src, target.open('wb') as dst:
                shutil.copyfileobj(src, dst)


def workspace(mode, roles, args, selection=None):
    source_check()
    pairs = selected_assets(roles, args.assets_dir)
    location = Path(args.workdir).resolve() if args.workdir else ROOT / 'work' / mode
    identity = {'mode': mode, 'manifest_sha256': sha256(MANIFEST),
                'source_manifest_sha256': sha256(ROOT / 'provenance/source-sha256.json')}
    marker = location / '.replication-workspace.json'
    if location.exists():
        if not marker.is_file() or read(marker) != identity:
            raise RuntimeError(f'Work directory belongs to another run or is incomplete: {location}. Use a new --workdir.')
        print(f'Resuming work directory: {location}', flush=True)
        return location
    location.mkdir(parents=True)
    for p in (ROOT / 'engine').rglob('*'):
        if p.is_file():
            dest = location / p.relative_to(ROOT / 'engine')
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, dest)
    for item, path in pairs:
        print(f"Extracting verified {item['name']} into {location}", flush=True)
        unpack(path, location, item, selection)
    marker.write_text(json.dumps(identity, indent=2) + '\n')
    return location


def command(args, cwd):
    print('RUN:', ' '.join(str(a) for a in args), flush=True)
    subprocess.run([str(a) for a in args], cwd=cwd, env=ENV, check=True)


def compare_tables(work):
    # The pinned stack normally reproduces exact text. A small numerical
    # tolerance allows final-bit arithmetic differences across supported hosts.
    import numpy as np
    import pandas as pd
    folder = work / 'results/confirmation_v06/summary'
    rows = 0
    for original in sorted((ROOT / 'reference').glob('*.csv')):
        a, b = pd.read_csv(original), pd.read_csv(folder / original.name)
        if list(a.columns) != list(b.columns) or a.shape != b.shape:
            raise AssertionError(f'Table structure changed: {original.name}')
        for c in a.columns:
            if pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c]):
                np.testing.assert_allclose(a[c], b[c], rtol=1e-10, atol=1e-12,
                                           equal_nan=True, err_msg=f'{original.name}: {c}')
            else:
                pd.testing.assert_series_equal(a[c], b[c], check_dtype=False)
        rows += len(a)
    expected, actual = read(ROOT / 'reference/integrity.json'), read(folder / 'integrity.json')
    for k in expected:
        if k == 'summary_utc':
            continue
        if isinstance(expected[k], float):
            np.testing.assert_allclose(actual[k], expected[k], rtol=1e-10, atol=1e-12, err_msg=k)
        elif actual[k] != expected[k]:
            raise AssertionError(f'Integrity field changed: {k}')
    primary = pd.read_csv(folder / 'primary_intervals.csv')
    assert len(primary) == 131072, 'Wrong primary interval count'
    print(f'PASS: {rows} reference-table rows, integrity fields, and 131,072 interval records.', flush=True)


def figures(summary, output):
    command([sys.executable, ROOT / 'publication/figures.py', '--summary', summary,
             '--output', output], ROOT)


def quick(args):
    work = workspace('quick', {'main'}, args)
    command([sys.executable, 'confirmation_summary.py'], work)
    compare_tables(work)
    figures(work / 'results/confirmation_v06/summary', work / 'figures')
    print(f'Quick reproduction complete: {work}', flush=True)


def verify(args):
    work = workspace('verify', {'main', 'calibration'}, args)
    command([sys.executable, '-c', 'import confirmation_runner as r; r.verify(); '
             'from pilot.continuation_evaluate import verify; verify()'], work)
    command([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'], work)
    command([sys.executable, 'confirmation_checks.py'], work)
    command([sys.executable, '-m', 'scripts.continuation_summary'], work)
    command([sys.executable, 'confirmation_summary.py'], work)
    compare_tables(work)
    print('Verification complete: source/model freezes, original tests, selected exact replays, '
          'preliminary summaries, and main summaries.', flush=True)


def refit(args):
    work = workspace('refit', {'main', 'calibration'}, args)
    code = r'''
from pathlib import Path
import json, shutil
import numpy as np
from pilot.repair_fit import train_bank
from pilot.continuation_calibration import run
base=Path('results'); new=base/'reproduction_v04'; new.mkdir(exist_ok=True)
for b in (0,1):
    dest=new/f'bank_{b}'
    if not dest.exists():shutil.copytree(base/'repaired_pilot'/f'bank_{b}',dest)
    train_bank(new,base/'expanded_pilot',b)
new05=base/'reproduction_v05'; new05.mkdir(exist_ok=True)
if not (new05/'calibration').exists():
    shutil.copytree(base/'continuation_v05/calibration',new05/'calibration')
run(new05,previous=new,workers=1)
for rebuilt,original in [(new,base/'repaired_pilot'),(new05,base/'continuation_v05')]:
    for p in sorted((rebuilt/'models').glob('*.npz')):
        with np.load(p,allow_pickle=False) as a,np.load(original/'models'/p.name,allow_pickle=False) as b:
            assert set(a.files)==set(b.files),p.name
            for key in a.files:
                if key=='metadata':
                    ma,mb=json.loads(str(a[key])),json.loads(str(b[key]))
                    for important in ['panel','mean_degree','response_degree','version']:
                        assert ma[important]==mb[important],(p.name,important)
                    np.testing.assert_allclose(ma['response_error_pp'],mb['response_error_pp'],rtol=1e-9,atol=1e-11)
                else:
                    np.testing.assert_allclose(a[key],b[key],rtol=1e-8,atol=1e-10,err_msg=f'{p.name}: {key}')
print('PASS: eight rebuilt pre-adjustment models and eight final calibrated models match retained numerical objects.')
'''
    command([sys.executable, '-c', code], work)
    print('Refit complete. Training and calibration market draws were retained, not regenerated.', flush=True)


def full(args):
    prefix = 'results/confirmation_v06/'
    keep = lambda n: n.startswith(prefix + 'models/') or n in {
        prefix + 'specification.json', prefix + 'source_and_model_freeze.json'}
    work = workspace('full', {'main'}, args, selection=keep)
    command([sys.executable, 'confirmation_runner.py', 'all', '--workers', args.workers], work)
    command([sys.executable, 'confirmation_summary.py'], work)
    compare_tables(work)
    figures(work / 'results/confirmation_v06/summary', work / 'figures')
    print('Full Monte Carlo reproduction complete. The original fitted models were held fixed.', flush=True)


def fetch(args):
    repository = args.repository
    if not repository:
        try:
            remote = subprocess.check_output(['git', 'remote', 'get-url', 'origin'], cwd=ROOT, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            remote = ''
        match = re.search(r'github\.com[:/]([\w.-]+/[\w.-]+?)(?:\.git)?$', remote)
        repository = match.group(1) if match else None
    if not repository or not re.fullmatch(r'[\w.-]+/[\w.-]+', repository):
        raise ValueError('Supply --repository OWNER/REPO or run from a GitHub clone.')
    dest = Path(args.assets_dir).resolve() if args.assets_dir else ROOT / 'assets'
    dest.mkdir(parents=True, exist_ok=True)
    for item in read(MANIFEST)['assets']:
        if not args.all_assets and item['role'] != 'main':
            continue
        target = dest / item['name']
        if target.exists():
            asset_path(item, dest)
            print(f'Already verified: {target.name}', flush=True)
            continue
        url = f"https://github.com/{repository}/releases/download/{read(MANIFEST)['release_tag']}/{item['name']}"
        temporary = target.with_suffix(target.suffix + '.download')
        print(f"Downloading {item['name']} ({item['size_bytes']/1048576:.1f} MiB)", flush=True)
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'Bitcoin-investor-information-replication/1.0.0'})
            with urllib.request.urlopen(request, timeout=60) as response, temporary.open('wb') as out:
                shutil.copyfileobj(response, out)
            if temporary.stat().st_size != item['size_bytes'] or sha256(temporary) != item['sha256']:
                raise RuntimeError('Downloaded file does not match the release manifest')
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    print(f'Assets available in {dest}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('route', choices=['check','quick','verify','refit','full','figures','fetch'])
    parser.add_argument('--assets-dir', help='Folder containing the two Release ZIP attachments')
    parser.add_argument('--workdir', help='Separate directory for this route; existing unrelated directories are refused')
    parser.add_argument('--workers', type=int, default=4, help='Processes for the full run (default: 4)')
    parser.add_argument('--source-only', action='store_true', help='For check: do not require Release assets')
    parser.add_argument('--repository', help='For fetch: GitHub OWNER/REPO')
    parser.add_argument('--all-assets', action='store_true', help='For fetch: include calibration and development data')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    if args.route == 'check':
        source_check()
        if not args.source_only:
            for item, path in selected_assets({'main','calibration'}, args.assets_dir):
                print(f"PASS: {path.name}, {item['members']} retained records; SHA-256 verified.")
    elif args.route == 'figures':
        figures(ROOT / 'reference', ROOT / 'work/figures')
    else:
        globals()[args.route](args)


if __name__ == '__main__':
    try:
        main()
    except (OSError, RuntimeError, ValueError, AssertionError, subprocess.CalledProcessError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(1)
