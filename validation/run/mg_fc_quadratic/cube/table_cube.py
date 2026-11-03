import json, sys, math
def load(fn):
    for line in open(fn):
        if 'MG_INTERFACE_WAVE' in line:
            return json.loads(line.split('MG_INTERFACE_WAVE')[1])
keys=('interface_acceleration_linf','interface_acceleration_l2','acceleration_linf','acceleration_relative_l2','phi_linf','phi_relative_l2')
tag, mb, Ns = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3].split(',')]
print("| N | blocks | " + " | ".join(keys) + " |"); print("|---|---|" + "---|"*len(keys))
prev=None
for N in Ns:
    d=load(f"log_{tag}_mb{mb}_{N}")
    cells=[]
    for k in keys:
        s=f"{d[k]:.3e}"
        if prev and d[k]>0 and prev[k]>0: s+=f" ({math.log2(prev[k]/d[k]):.2f})"
        cells.append(s)
    print(f"| {N} | {d['blocks']} | " + " | ".join(cells) + " |"); prev=d
