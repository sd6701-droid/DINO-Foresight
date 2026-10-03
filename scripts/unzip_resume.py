#!/usr/bin/env python3
"""Resumable zip extraction.

Extracts every member of ZIP into DEST, skipping members that already exist on
disk with the expected uncompressed size. A member that was only partially
written when a previous run was killed has the wrong size and is re-extracted.
Writes go through a temporary ".part" file that is renamed into place, so a
file with the right name always has complete contents.

    python unzip_resume.py archive.zip /dest/dir [--exclude README license.txt]
"""
import argparse
import os
import shutil
import sys
import time
import zipfile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('zip')
    ap.add_argument('dest')
    ap.add_argument('--exclude', nargs='*', default=[], help='member names to skip')
    ap.add_argument('--report-every', type=int, default=5000, help='progress line every N members')
    args = ap.parse_args()

    exclude = set(args.exclude)
    os.makedirs(args.dest, exist_ok=True)
    t0 = time.time()
    done = skipped = 0
    with zipfile.ZipFile(args.zip) as zf:
        members = zf.infolist()
        total = len(members)
        print(f"[unzip_resume] {args.zip}: {total} members -> {args.dest}", flush=True)
        for i, info in enumerate(members, 1):
            name = info.filename
            if name in exclude or os.path.basename(name) in exclude:
                continue
            target = os.path.join(args.dest, name)
            if info.is_dir():
                os.makedirs(target, exist_ok=True)
                continue
            try:
                if os.path.getsize(target) == info.file_size:
                    skipped += 1
                    continue
            except OSError:
                pass
            os.makedirs(os.path.dirname(target), exist_ok=True)
            tmp = target + '.part'
            with zf.open(info) as src, open(tmp, 'wb') as dst:
                shutil.copyfileobj(src, dst, 1024 * 1024)
            os.replace(tmp, target)
            done += 1
            if i % args.report_every == 0:
                print(f"[unzip_resume] {i}/{total} members, extracted {done}, "
                      f"skipped {skipped}, {time.time() - t0:.0f}s", flush=True)
    print(f"[unzip_resume] finished: extracted {done}, skipped {skipped} (already present), "
          f"{time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
