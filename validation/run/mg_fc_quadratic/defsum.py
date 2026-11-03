# usage: defsum.py LOG...  -> per log: mean contraction over V-cycles 2..11, min/max ratio there,
# floor (last defect) and the first cycle within 2x of the floor
import sys, math
for fn in sys.argv[1:]:
    d=[float(l.split()[-1]) for l in open(fn) if 'MG iteration' in l]
    r=[d[i]/d[i-1] for i in range(2,12)]
    g=math.exp(sum(math.log(x) for x in r)/len(r))
    fl=d[-1]; n=next(i for i,x in enumerate(d) if x < 2*fl)
    print(f"{fn}: mean {g:.3f} (min {min(r):.2f}, max {max(r):.2f}) floor {fl:.2e} reached at cycle {n}")
