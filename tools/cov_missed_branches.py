#!/usr/bin/env python3
"""Dump specific uncovered branch/region locations per file."""
import json
import sys
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else 'target/cov-baseline.json'
filter_name = sys.argv[2] if len(sys.argv) > 2 else None
d = json.load(open(path))['data'][0]

for f in d['files']:
    name = f['filename'].replace('\\', '/')
    short = name.split('/src/', 1)[1] if '/src/' in name else name.split('src/', 1)[-1]
    if filter_name and filter_name not in short:
        continue
    miss_br = f['summary']['branches']['count'] - f['summary']['branches']['covered']
    miss_reg = f['summary']['regions']['count'] - f['summary']['regions']['covered']
    if miss_br == 0 and miss_reg == 0:
        continue
    print(f"\n== {short} (missed branches={miss_br}, regions={miss_reg}) ==")

    # Branches: [start_line, start_col, end_line, end_col, true_count, false_count, expansion_id, file_id, kind]
    # kind 4 = default, we want branches where true_count==0 or false_count==0
    branch_locs = []
    for b in f.get('branches', []):
        if len(b) < 6:
            continue
        sl, sc, el, ec, tc, fc = b[0], b[1], b[2], b[3], b[4], b[5]
        if tc == 0 or fc == 0:
            side = "true" if tc == 0 else "false"
            branch_locs.append((sl, sc, side, tc, fc))
    by_line = defaultdict(list)
    for sl, sc, side, tc, fc in branch_locs:
        by_line[sl].append((side, tc, fc))
    for ln in sorted(by_line.keys()):
        sides = by_line[ln]
        parts = [f"{s}(t={tc},f={fc})" for s, tc, fc in sides]
        print(f"  L{ln}: {', '.join(parts)}")
