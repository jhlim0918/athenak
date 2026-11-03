import json, sys, math, re
def load(fn):
    for line in open(fn):
        if 'MG_INTERFACE_WAVE' in line:
            return json.loads(line.split('MG_INTERFACE_WAVE')[1])
    return None
def table(tag, mb, Ns, keys=('interface_acceleration_linf','interface_acceleration_l2','acceleration_linf','acceleration_relative_l2','phi_linf','phi_relative_l2'), pre='log_'):
    rows=[]
    prev=None
    print(f"| N | " + " | ".join(keys) + " |")
    print("|---|" + "---|"*len(keys))
    for N in Ns:
        d=load(f"{pre}{tag}_mb{mb}_{N}")
        if d is None: print(f"| {N} | missing |"); continue
        cells=[]
        for k in keys:
            s=f"{d[k]:.3e}"
            if prev is not None and d[k] > 0 and prev[k] > 0:
                s+=f" ({math.log2(prev[k]/d[k]):.2f})"
            cells.append(s)
        print(f"| {N} | " + " | ".join(cells) + " |")
        prev=d
if __name__=='__main__':
    tag, mb = sys.argv[1], sys.argv[2]
    Ns=[int(x) for x in sys.argv[3].split(',')]
    pre = sys.argv[4] if len(sys.argv)>4 else 'log_'
    table(tag, mb, Ns, pre=pre)
