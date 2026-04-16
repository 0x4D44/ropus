#!/usr/bin/env python3
"""Summarize llvm-cov JSON for branch coverage analysis."""
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else 'target/cov-baseline.json'
d = json.load(open(path))['data'][0]
files = d['files']
rows = []
for f in files:
    s = f['summary']
    name = f['filename'].replace('\\', '/')
    if '/src/' in name:
        name = name.split('/src/', 1)[1]
    elif 'src\\' in f['filename']:
        name = f['filename'].split('src\\', 1)[1].replace('\\', '/')
    rows.append((
        name,
        s['branches']['count'],
        s['branches']['covered'],
        s['branches']['count'] - s['branches']['covered'],
        s['branches']['percent'],
        s['lines']['count'] - s['lines']['covered'],
        s['regions']['count'] - s['regions']['covered'],
    ))
rows.sort(key=lambda r: -r[3])
print(f"{'file':<32} {'total':>6} {'cov':>6} {'miss':>6} {'pct':>7} {'lmiss':>6} {'rmiss':>6}")
for r in rows:
    print(f"{r[0]:<32} {r[1]:>6} {r[2]:>6} {r[3]:>6} {r[4]:>6.2f}% {r[5]:>6} {r[6]:>6}")
print(f"\nTotal missed branches: {sum(r[3] for r in rows)}")
print(f"Total missed lines: {sum(r[5] for r in rows)}")
print(f"Total missed regions: {sum(r[6] for r in rows)}")

tot = d['totals']
print(f"\nLines:     {tot['lines']['percent']:.2f}% ({tot['lines']['covered']}/{tot['lines']['count']})")
print(f"Regions:   {tot['regions']['percent']:.2f}% ({tot['regions']['covered']}/{tot['regions']['count']})")
print(f"Functions: {tot['functions']['percent']:.2f}% ({tot['functions']['covered']}/{tot['functions']['count']})")
print(f"Branches:  {tot['branches']['percent']:.2f}% ({tot['branches']['covered']}/{tot['branches']['count']})")
