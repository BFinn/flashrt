import csv, re, sys
d = sys.argv[1]
def load(f):
    r = {}
    for row in csv.DictReader(open(f)):
        name = row["Name"][:70]
        t, n = r.get(name, (0.0, 0))
        r[name] = (t + float(row["Total Time (ns)"]) / 1e6, n + int(row["Instances"]))
    return r
sp = load(d + "/spec2_cuda_gpu_kern_sum.csv"); pl = load(d + "/plain_cuda_gpu_kern_sum.csv")
txt = open(d + "/spec2.txt").read()
rounds = int(sys.argv[2])
tot_sp = sum(v[0] for v in sp.values()); tot_pl = sum(v[0] for v in pl.values())
print("rounds", rounds)
print("total GPU ms: spec2 %.1f (%.2f/round), plain %.1f (%.2f/token)" % (tot_sp, tot_sp / rounds, tot_pl, tot_pl / 128))
names = sorted(set(sp) | set(pl), key=lambda n: -(sp.get(n, (0, 0))[0]))
print("%-70s %13s %12s %6s" % ("kernel", "spec ms/rnd", "plain ms/tok", "ratio"))
for n in names[:34]:
    a = sp.get(n, (0, 0))[0] / rounds; b = pl.get(n, (0, 0))[0] / 128
    print("%-70s %13.3f %12.3f %6.2f" % (n, a, b, a / b if b else 0))
