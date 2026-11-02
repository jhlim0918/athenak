#!/usr/bin/env python3
"""Compare AthenaK .bin payloads after the parameter header: cmpbin.py DIR_A DIR_B"""
import os, re, sys
def payload(p):
    b = open(p, 'rb').read()
    m = re.search(rb'header offset=(\d+)', b)
    start = b.index(b'\n', m.end()) + 1 + int(m.group(1))
    return b[start:]
a, c = sys.argv[1], sys.argv[2]
n = same = 0
for f in sorted(os.listdir(a)):
    if not f.endswith('.bin'): continue
    n += 1
    if payload(os.path.join(a, f)) == payload(os.path.join(c, f)): same += 1
    else: print('DIFF', f)
print(f'{same}/{n} payloads bit-identical')
