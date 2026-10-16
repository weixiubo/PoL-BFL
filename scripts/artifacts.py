#!/usr/bin/env python3
"""Restore and verify the public PoL-BFL research archive (standard library only)."""
from __future__ import annotations

import argparse
import contextlib
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import stat
import subprocess
import tarfile
import tempfile
import urllib.error
import urllib.parse
import urllib.request


class ArchiveError(RuntimeError):
    pass


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def relative(name):
    if not isinstance(name, str) or not name or '\\' in name:
        raise ArchiveError(f'Invalid relative path: {name!r}')
    p = PurePosixPath(name)
    if p.is_absolute() or any(x in {'..', '.'} for x in p.parts) or '/'.join(p.parts) != name:
        raise ArchiveError(f'Unsafe relative path: {name!r}')
    return p


def safe(root, name):
    path = root.joinpath(*relative(name).parts)
    for parent in [root, *list(path.parents)[:-1][::-1], path]:
        if parent != root and root not in parent.parents:
            continue
        if parent.is_symlink():
            raise ArchiveError(f'Symlink in managed path: {parent}')
    return path


def root_path(value, create=False):
    root = Path(value).expanduser().absolute()
    for p in [*root.parents, root]:
        if p.is_symlink():
            raise ArchiveError(f'Restore root contains a symlink: {p}')
    root = root.resolve()
    if create:
        root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir():
        raise ArchiveError(f'Restore directory does not exist: {root}')
    return root


def check_file(path, info):
    if path.is_symlink() or not path.is_file():
        raise ArchiveError(f'Missing or nonregular file: {path}')
    if path.stat().st_size != info['size'] or sha256(path) != info['sha256']:
        raise ArchiveError(f'Size or SHA-256 mismatch; refusing to overwrite: {path}')


def load_catalog(path):
    path = Path(path).absolute()
    catalog = json.loads(path.read_text(encoding='utf-8'))
    if catalog.get('schema_version') != 1:
        raise ArchiveError('Unsupported catalog version')
    for key in ['archive_id', 'manifest', 'sources', 'parts']:
        if key not in catalog:
            raise ArchiveError(f'Catalog is missing {key}')
    names = set()
    for part in catalog['parts']:
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', part['name']) or part['name'] in names:
            raise ArchiveError('Invalid or duplicate archive part name')
        if not re.fullmatch(r'[0-9a-f]{64}', part['sha256']) or not 0 < part['size'] <= 1024 ** 3:
            raise ArchiveError('Invalid archive part checksum or size')
        names.add(part['name'])
    manifest_path = safe(path.parent, catalog['manifest']['name'])
    check_file(manifest_path, catalog['manifest'])
    with gzip.open(manifest_path, 'rt', encoding='utf-8') as f:
        rows = [json.loads(line) for line in f if line.strip()]
    seen = set()
    for row in rows:
        relative(row['path'])
        if row['path'] in seen or row['type'] not in {'file', 'directory', 'symlink'}:
            raise ArchiveError('Invalid or duplicate manifest entry')
        seen.add(row['path'])
        if row['type'] == 'file':
            if not re.fullmatch(r'[0-9a-f]{64}', row['sha256']) or row['size'] < 0:
                raise ArchiveError('Invalid file checksum or size')
        if not isinstance(row['mode'], int) or row['mode'] & ~0o777:
            raise ArchiveError('Special file permission bits are not supported')
        if row['type'] == 'symlink':
            target = PurePosixPath(row['target'])
            if target.is_absolute() or '\\' in row['target']:
                raise ArchiveError('Absolute or unsafe symlink target')
            virtual = Path('/__polbfl_archive_root__')
            resolved = (virtual/row['path']).parent.joinpath(row['target']).resolve()
            if resolved != virtual and virtual not in resolved.parents:
                raise ArchiveError('Symlink target escapes archive root')
    link_paths = {r['path'] for r in rows if r['type'] == 'symlink'}
    for row in rows:
        if any(p.as_posix() in link_paths for p in PurePosixPath(row['path']).parents):
            raise ArchiveError('Manifest places a file beneath a symlink')
    return catalog, rows


def urls(catalog, source, name):
    providers = ['github'] if source == 'auto' else [source]
    result = []
    for provider in providers:
        base = catalog['sources'].get(provider)
        if not base:
            continue
        encoded = urllib.parse.quote(name, safe='')
        url = base.replace('{name}', encoded) if '{name}' in base else base + encoded
        if urllib.parse.urlparse(url).scheme != 'https':
            raise ArchiveError('Archive downloads require HTTPS')
        result.append((provider, url))
    if not result:
        raise ArchiveError(f'No configured download location for {source}')
    return result


def obtain_part(catalog, part, cache, source, offline):
    path = safe(cache, part['name'])
    if path.exists() or path.is_symlink():
        check_file(path, part)
        return path
    if offline:
        raise ArchiveError(f'Offline restore is missing part: {part["name"]}')
    errors = []
    for provider, url in urls(catalog, source, part['name']):
        fd, tmpname = tempfile.mkstemp(prefix='.download-', dir=cache)
        tmp = Path(tmpname)
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'PoL-BFL-artifacts/1'})
            with os.fdopen(fd, 'wb') as f:
                with urllib.request.urlopen(request, timeout=120) as response:
                    size = 0
                    while block := response.read(4 * 1024 * 1024):
                        size += len(block)
                        if size > part['size']:
                            raise ArchiveError(f'Oversized download from {provider}: {part["name"]}')
                        f.write(block)
            check_file(tmp, part)
            # Exclusive publication preserves an unexpected existing user file.
            os.link(tmp, path)
            print(f'downloaded {part["name"]} from {provider}', flush=True)
            return path
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            errors.append(f'{provider}: {type(exc).__name__}: {exc}')
        finally:
            tmp.unlink(missing_ok=True)
    raise ArchiveError('Download failed: ' + '; '.join(errors))


class JoinedParts(io.RawIOBase):
    def __init__(self, paths):
        self.paths = iter(paths)
        self.current = None

    def readable(self):
        return True

    def readinto(self, buffer):
        while True:
            if self.current is None:
                try:
                    self.current = next(self.paths).open('rb')
                except StopIteration:
                    return 0
            count = self.current.readinto(buffer)
            if count:
                return count
            self.current.close()
            self.current = None

    def close(self):
        if self.current:
            self.current.close()
        super().close()


def extract_blobs(paths, blobdir, rows):
    expected = {}
    for row in rows:
        if row['type'] == 'file':
            if row['sha256'] in expected and expected[row['sha256']]['size'] != row['size']:
                raise ArchiveError('Inconsistent content-addressed file size')
            expected[row['sha256']] = row
    seen = set()
    with JoinedParts(paths) as raw, io.BufferedReader(raw) as stream:
        with tarfile.open(fileobj=stream, mode='r|gz') as tar:
            for entry in tar:
                if not re.fullmatch(r'blobs/[0-9a-f]{64}', entry.name) or not entry.isfile():
                    raise ArchiveError(f'Unsafe archive member: {entry.name}')
                key = entry.name.split('/')[1]
                if key in seen or key not in expected or entry.size != expected[key]['size']:
                    raise ArchiveError(f'Unexpected archive member: {entry.name}')
                seen.add(key)
                target = safe(blobdir, key)
                if target.exists() or target.is_symlink():
                    check_file(target, expected[key])
                    continue
                fd, tmpname = tempfile.mkstemp(prefix='.blob-', dir=blobdir)
                tmp = Path(tmpname)
                try:
                    with os.fdopen(fd, 'wb') as output, tar.extractfile(entry) as source:
                        shutil.copyfileobj(source, output, 4 * 1024 * 1024)
                    check_file(tmp, expected[key])
                    os.link(tmp, target)
                finally:
                    tmp.unlink(missing_ok=True)
    if seen != set(expected):
        raise ArchiveError('Archive does not contain every required blob')


def inspect_entry(root, row, missing_allowed=False):
    path = root.joinpath(*relative(row['path']).parts)
    safe(root, str(PurePosixPath(row['path']).parent)) if '/' in row['path'] else None
    if not path.exists() and not path.is_symlink():
        if missing_allowed:
            return False
        raise ArchiveError(f'Missing restored path: {path}')
    if row['type'] == 'symlink':
        if not path.is_symlink() or os.readlink(path) != row['target']:
            raise ArchiveError(f'Changed symlink: {path}')
        try:
            path.resolve().relative_to(root)
        except ValueError:
            raise ArchiveError(f'Symlink escapes restore root: {path}') from None
    else:
        safe(root, row['path'])
        if row['type'] == 'directory':
            if not path.is_dir():
                raise ArchiveError(f'Not a directory: {path}')
        else:
            check_file(path, row)
        if stat.S_IMODE(path.stat().st_mode) != row['mode']:
            raise ArchiveError(f'Changed permissions: {path}')
    return True


def git(root, *args):
    env = dict(os.environ, GIT_NO_LAZY_FETCH='1', GIT_OPTIONAL_LOCKS='0',
               GIT_TERMINAL_PROMPT='0', GIT_CONFIG_NOSYSTEM='1')
    result = subprocess.run(['git', '-c', 'gc.auto=0', '-c', 'maintenance.auto=false',
        '-c', 'core.hooksPath=/dev/null', '-C', str(root), *args], env=env,
        capture_output=True, text=True)
    if result.returncode:
        raise ArchiveError(f'Git {args[0]} failed: {result.stderr.strip()}')
    return result.stdout.strip()


def check_history(root, history):
    path = safe(root, history['repository'])
    if not path.is_dir() or (path/'objects/info/alternates').exists():
        raise ArchiveError('Historical repository is missing or depends on external objects')
    if git(path, 'rev-parse', '--is-bare-repository') != 'true':
        raise ArchiveError('Historical repository is not bare')
    refs = dict(line.split() for line in git(path, 'for-each-ref', '--format=%(refname) %(objectname)').splitlines())
    if refs != history['refs']:
        raise ArchiveError('Historical Git references have changed')
    if 'head' in history and git(path, 'symbolic-ref', 'HEAD') != history['head']:
        raise ArchiveError('Historical Git HEAD has changed')
    git(path, 'fsck', '--full', '--no-reflogs')


def restore_history(root, history):
    target = safe(root, history['repository'])
    if target.exists():
        check_history(root, history)
        return
    bundle = safe(root, history['bundle'])
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix='.history-', dir=target.parent))
    try:
        git(tmp, 'init', '--bare', '.')
        git(tmp, 'fetch', '--no-tags', str(bundle), '+refs/*:refs/*')
        if 'head' in history:
            git(tmp, 'symbolic-ref', 'HEAD', history['head'])
        tmp.rename(target)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)
    check_history(root, history)


def controlled_file(path, content):
    raw = content.encode('utf-8')
    if path.exists() or path.is_symlink():
        if path.is_symlink() or path.read_bytes() != raw:
            raise ArchiveError(f'User-modified control file: {path}')
    else:
        with path.open('xb') as f:
            f.write(raw)


def resource_files(root, catalog):
    data = root/'datasets'
    legacy = root/'zk-build/parameter_update'
    resources = {'archive_id': catalog['archive_id'], 'data_root': str(data),
        'legacy_zk_build': str(legacy), 'formal_production_zk_build': None,
        'note': 'The archived legacy ZK circuit is not the formal production circuit. See docs/ARTIFACTS.md.'}
    return {'resources.json': json.dumps(resources, ensure_ascii=False, indent=2)+'\n',
        'resources.env': 'export POLBFL_DATA_ROOT='+shlex.quote(str(data))+'\n'+
        'export POLBFL_LEGACY_ZK_BUILD='+shlex.quote(str(legacy))+'\n',
        '.artifact-state.json': json.dumps({'schema_version': 1, 'archive_id': catalog['archive_id'],
            'manifest_sha256': catalog['manifest']['sha256']}, sort_keys=True)+'\n'}


def verify(root, catalog, rows):
    for row in rows:
        inspect_entry(root, row)
    if catalog.get('history'):
        check_history(root, catalog['history'])
    for name, text in resource_files(root, catalog).items():
        path = safe(root, name)
        if not path.is_file() or path.read_text(encoding='utf-8') != text:
            raise ArchiveError(f'Missing or changed resource configuration: {path}')
    return {'archive_id': catalog['archive_id'], 'entries_verified': len(rows),
            'files_verified': sum(r['type']=='file' for r in rows), 'passed': True}


def restore(root, catalog, rows, source='auto', offline=False, cache_dir=None):
    # Validate all existing managed paths before downloading or creating content.
    existing = {r['path']: inspect_entry(root, r, missing_allowed=True) for r in rows}
    for name, text in resource_files(root, catalog).items():
        path = safe(root, name)
        if path.exists() or path.is_symlink():
            controlled_file(path, text)
    if catalog.get('history') and safe(root, catalog['history']['repository']).exists():
        check_history(root, catalog['history'])
    cache = root_path(cache_dir, create=True) if cache_dir else safe(root, '.artifact-cache')
    cache.mkdir(exist_ok=True)
    blobs = safe(cache, 'blobs')
    blobs.mkdir(exist_ok=True)
    if not all(existing.values()):
        parts = [obtain_part(catalog, part, cache, source, offline) for part in catalog['parts']]
        extract_blobs(parts, blobs, rows)
    for row in sorted(rows, key=lambda r: (r['type'] != 'directory', len(PurePosixPath(r['path']).parts))):
        if existing[row['path']]:
            continue
        target = safe(root, row['path'])
        target.parent.mkdir(parents=True, exist_ok=True)
        if row['type'] == 'directory':
            target.mkdir(exist_ok=True)
        elif row['type'] == 'file':
            with target.open('xb') as output, safe(blobs, row['sha256']).open('rb') as input_file:
                shutil.copyfileobj(input_file, output, 4 * 1024 * 1024)
        else:
            resolved = (target.parent/row['target']).resolve()
            if root != resolved and root not in resolved.parents:
                raise ArchiveError('Symlink target escapes restore root')
            target.symlink_to(row['target'])
        if row['type'] != 'symlink':
            target.chmod(row['mode'])
    if catalog.get('history'):
        restore_history(root, catalog['history'])
    for name, text in resource_files(root, catalog).items():
        controlled_file(safe(root, name), text)
    return verify(root, catalog, rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['restore', 'verify'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source', choices=['auto', 'github'], default='auto')
    parser.add_argument('--offline', action='store_true', help='Never use the network; verify is always offline')
    parser.add_argument('--cache-dir', type=Path, help='Reuse an explicitly selected, checksum-verified download cache')
    parser.add_argument('--catalog', type=Path, default=Path(__file__).resolve().parents[1]/'artifacts/catalog.json')
    args = parser.parse_args(argv)
    try:
        catalog, rows = load_catalog(args.catalog)
        root = root_path(args.root, create=args.command=='restore')
        result = restore(root,catalog,rows,args.source,args.offline,args.cache_dir) if args.command=='restore' else verify(root,catalog,rows)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ArchiveError, OSError, ValueError, KeyError, tarfile.TarError, EOFError) as exc:
        print(f'Archive check failed: {exc}', file=__import__('sys').stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
