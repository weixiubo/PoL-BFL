import gzip
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import urllib.error

import pytest

SCRIPT = Path(__file__).resolve().parents[1]/'scripts/artifacts.py'
spec = importlib.util.spec_from_file_location('polbfl_artifacts', SCRIPT)
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)


def checksum(data):
    return hashlib.sha256(data).hexdigest()


def fixture_archive(tmp_path, payloads=None, rows_extra=(), malicious_member=None):
    payloads = payloads or {'materials/中文路径/原始记录.txt': '实验记录\n'.encode(), 'materials/config.json': b'{"seed":1337}\n'}
    rows = [{'path':p,'type':'file','size':len(b),'sha256':checksum(b),'mode':0o644} for p,b in payloads.items()]
    dirs = {str(q) for p in payloads for q in Path(p).parents if str(q)!='.'}
    rows += [{'path':p,'type':'directory','mode':0o755} for p in sorted(dirs)]
    rows += list(rows_extra)
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw,mode='w:gz') as t:
        for data in {checksum(b):b for b in payloads.values()}.values():
            entry=tarfile.TarInfo('blobs/'+checksum(data));entry.size=len(data)
            t.addfile(entry,io.BytesIO(data))
        if malicious_member:
            entry=tarfile.TarInfo(malicious_member);entry.size=1;t.addfile(entry,io.BytesIO(b'x'))
    data=raw.getvalue();mid=len(data)//2
    parts={f'materials.part{i:03d}':b for i,b in enumerate([data[:mid],data[mid:]],1)}
    embedded=tmp_path/'catalog';embedded.mkdir()
    mf=gzip.compress(('\n'.join(json.dumps(r) for r in rows)+'\n').encode())
    (embedded/'manifest.jsonl.gz').write_bytes(mf)
    catalog={'schema_version':1,'archive_id':'fixture-v1',
        'manifest':{'name':'manifest.jsonl.gz','size':len(mf),'sha256':checksum(mf)},
        'sources':{'github':'https://github.example/{name}','unused':'https://unused.example/{name}'},
        'parts':[{'name':n,'size':len(b),'sha256':checksum(b)} for n,b in parts.items()]}
    file=embedded/'catalog.json';file.write_text(json.dumps(catalog))
    return file,catalog,rows,parts


def downloader(monkeypatch, parts, fail=None):
    calls=[]
    def open_url(request,timeout=None):
        url=request.full_url;calls.append(url)
        if fail and fail in url:raise urllib.error.URLError('fixture unavailable')
        return io.BytesIO(parts[url.rsplit('/',1)[-1]])
    monkeypatch.setattr(h.urllib.request,'urlopen',open_url)
    return calls


@pytest.mark.parametrize('source,domain',[('github','github.example')])
def test_public_provider_chinese_paths_and_repeat(tmp_path,monkeypatch,source,domain):
    path,c,rows,parts=fixture_archive(tmp_path)
    calls=downloader(monkeypatch,parts)
    c,rows=h.load_catalog(path)
    root=h.root_path(tmp_path/'中文恢复目录',True)
    assert h.restore(root,c,rows,source)['passed']
    assert (root/'materials/中文路径/原始记录.txt').read_text()=='实验记录\n'
    assert len(calls)==2 and all(domain in u for u in calls)
    before=len(calls)
    assert h.restore(root,c,rows,source,True)['passed']
    assert h.verify(root,c,rows)['passed']
    assert len(calls)==before


def test_network_failure_does_not_use_another_provider(tmp_path,monkeypatch):
    _,c,rows,parts=fixture_archive(tmp_path)
    calls=downloader(monkeypatch,parts,'github.example')
    with pytest.raises(h.ArchiveError,match='Download failed'):
        h.restore(h.root_path(tmp_path/'github-only',True),c,rows,'github')
    assert all('github' in u for u in calls)


def test_offline_restore_from_verified_cache(tmp_path,monkeypatch):
    _,c,rows,parts=fixture_archive(tmp_path)
    cache=tmp_path/'preloaded';cache.mkdir()
    for n,b in parts.items():(cache/n).write_bytes(b)
    monkeypatch.setattr(h.urllib.request,'urlopen',lambda *a,**k:pytest.fail('network used'))
    assert h.restore(h.root_path(tmp_path/'offline',True),c,rows,offline=True,cache_dir=cache)['passed']


@pytest.mark.parametrize('failure',['missing','corrupt','download-corrupt'])
def test_bad_part_stops_before_materializing(tmp_path,monkeypatch,failure):
    _,c,rows,parts=fixture_archive(tmp_path)
    root=h.root_path(tmp_path/'out',True);cache=root/'.artifact-cache';cache.mkdir()
    if failure=='missing':
        (cache/next(iter(parts))).write_bytes(next(iter(parts.values())))
    elif failure=='corrupt':
        for n,b in parts.items():(cache/n).write_bytes(b+b'x')
    else:
        parts={n:b'corrupt' for n in parts};downloader(monkeypatch,parts)
    with pytest.raises(h.ArchiveError):
        h.restore(root,c,rows,offline=failure!='download-corrupt')
    assert not (root/'materials').exists()


@pytest.mark.parametrize('kind',['content','mode','control','symlink'])
def test_changed_user_state_is_not_overwritten(tmp_path,monkeypatch,kind):
    _,c,rows,parts=fixture_archive(tmp_path);downloader(monkeypatch,parts)
    root=h.root_path(tmp_path/'out',True);h.restore(root,c,rows)
    target=root/'materials/config.json'
    if kind=='content':target.write_bytes(b'user edits')
    elif kind=='mode':target.chmod(0o600)
    elif kind=='control':(root/'resources.env').write_text('user configuration')
    else:
        target.unlink();target.symlink_to(tmp_path/'outside')
    with pytest.raises(h.ArchiveError):h.restore(root,c,rows)
    with pytest.raises(h.ArchiveError):h.verify(root,c,rows)
    if kind=='content':assert target.read_bytes()==b'user edits'
    if kind=='control':assert (root/'resources.env').read_text()=='user configuration'


@pytest.mark.parametrize('invalid_name',['../escape','/absolute','a/../../escape','a\\b'])
def test_unsafe_manifest_path_rejected(tmp_path,invalid_name):
    path,c,rows,parts=fixture_archive(tmp_path,{invalid_name:b'no'})
    with pytest.raises(h.ArchiveError):h.load_catalog(path)


def test_escaping_symlink_rejected_before_restore(tmp_path):
    path,*_=fixture_archive(tmp_path,rows_extra=[{'path':'link','type':'symlink','mode':0o777,'target':'../outside'}])
    with pytest.raises(h.ArchiveError):h.load_catalog(path)


def test_relative_symlink_and_target_verified(tmp_path,monkeypatch):
    path,c,rows,parts=fixture_archive(tmp_path,rows_extra=[{'path':'materials/link','type':'symlink','mode':0o777,'target':'config.json'}])
    c,rows=h.load_catalog(path);downloader(monkeypatch,parts)
    root=h.root_path(tmp_path/'out',True)
    assert h.restore(root,c,rows)['passed']
    assert os.readlink(root/'materials/link')=='config.json'


def test_archive_path_traversal_rejected(tmp_path,monkeypatch):
    _,c,rows,parts=fixture_archive(tmp_path,malicious_member='../escape');downloader(monkeypatch,parts)
    root=h.root_path(tmp_path/'out',True)
    with pytest.raises(h.ArchiveError,match='Unsafe archive member'):h.restore(root,c,rows)
    assert not (tmp_path/'escape').exists()
    assert not (root/'materials').exists()


def test_manifest_corruption_rejected(tmp_path):
    path,*_=fixture_archive(tmp_path);(path.parent/'manifest.jsonl.gz').write_bytes(b'changed')
    with pytest.raises(h.ArchiveError):h.load_catalog(path)


def test_history_metadata_is_optional(tmp_path,monkeypatch):
    _,c,rows,parts=fixture_archive(tmp_path);del c['sources']['unused'];downloader(monkeypatch,parts)
    assert h.restore(h.root_path(tmp_path/'out',True),c,rows)['passed']


def test_bare_history_restores_exact_refs_and_protects_changes(tmp_path,monkeypatch):
    original=tmp_path/'git-original';original.mkdir()
    def git(*args):
        return subprocess.check_output(['git','-c','user.name=Archive Test','-c','user.email=archive@example.invalid','-C',str(original),*args],text=True).strip()
    git('init','-b','research')
    (original/'code.txt').write_text('original research\n');git('add','code.txt');git('commit','-m','Original')
    git('tag','paper-version')
    bundle=tmp_path/'history.bundle';git('bundle','create',str(bundle),'--all')
    refs=dict(l.split() for l in git('for-each-ref','--format=%(refname) %(objectname)').splitlines())
    _,c,rows,parts=fixture_archive(tmp_path,{'history/research-history.bundle':bundle.read_bytes()})
    c['history']={'bundle':'history/research-history.bundle','repository':'history/research.git','refs':refs,'head':'refs/heads/research'}
    downloader(monkeypatch,parts);root=h.root_path(tmp_path/'out',True)
    assert h.restore(root,c,rows)['passed']
    assert h.restore(root,c,rows,offline=True)['passed']
    h.git(root/'history/research.git','update-ref','refs/heads/changed',next(iter(refs.values())))
    with pytest.raises(h.ArchiveError,match='references have changed'):h.restore(root,c,rows)
    assert 'refs/heads/changed' in h.git(root/'history/research.git','show-ref')
