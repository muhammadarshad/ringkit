"""Production tests for ringkit.ml (autograd + optim + nn). Run: python3 -m ringkit.tests.test_ml"""
from ringkit.core import native as rn
from ringkit.ml import autograd as ag
from ringkit.ml import optim as opt
from ringkit.ml import nn

fails = []
def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond: fails.append(name)
def sg(x): return rn._signed(x)
V = ag.Var

print("== autograd (dual-ring: ARC value / ENERGY grad) ==")
def dsin(a): x=V(a); x.sin().backward(); return x.grad
def dcos(a): x=V(a); x.cos().backward(); return x.grad
def dsq(a):  x=V(a); x.mul(x).backward(); return x.grad
def dmul(a,b): x=V(a); w=V(b); x.mul(w).backward(); return x.grad, w.grad
check("d SIN == signed(COS) over 256", all(dsin(a) == sg(rn.COS(a)) for a in range(256)))
check("d COS == -signed(SIN) over 256", all(dcos(a) == -sg(rn.SIN(a)) for a in range(256)))
check("d(a^2) == 2*signed(a)", all(dsq(a) == 2*sg(a) for a in range(256)))
check("d(a*b) == (signed b, signed a)", all(dmul(a,b) == (sg(b), sg(a)) for a in range(0,256,9) for b in range(0,256,9)))
def chain(a,b):
    x=V(a); w=V(b); x.mul(w).sin().backward(); return x.grad
def manual(a,b):
    p=rn.qsm(a,b)&0xFF; return rn.mul(sg(rn.COS(p)), sg(b))
check("chain d SIN(a*b) == manual", all(chain(a,b)==manual(a,b) for a in range(0,256,17) for b in range(0,256,17)))
x=V(7); x.add(x).backward(); check("reuse y=a+a -> grad 2", x.grad == 2)

print("== optim (full-stack scalar training) ==")
check("sgd_step sign", opt.sgd_step(10, 5) == 9 and opt.sgd_step(10, -5) == 11 and opt.sgd_step(10, 0) == 10)
target = rn.SIN(100)
def loss(xv): d = sg(rn.SIN(xv)) - sg(target); return d*d
conv = 0
for x0 in range(256):
    x = x0
    for _ in range(300):
        if loss(x) == 0: conv += 1; break
        xv = V(x); d = xv.sin().sub(V(target)); L = d.mul(d); L.backward()
        g = xv.grad
        if g == 0: x = (x+1) & 0xFF; continue
        x = opt.sgd_step(x, g)
check("SIN scalar descent converges 256/256", conv == 256)

print("== nn (RingModule / Neuron) ==")
m = nn.Neuron([10, 20], 5)
check("parameters count", len(m.parameters()) == 3)
check("forward = SIN(dot+b)", m.forward([3,4]).val == rn.SIN((rn.qsm(10,3)+rn.qsm(20,4)+5) & 0xFF))
out = m.forward([3,4]); out.backward()
check("grads flow to all params", all(isinstance(p.grad, int) for p in m.parameters()))
before = [p.val for p in m.parameters()]; m.step(lr=1)
check("step updates params", [p.val for p in m.parameters()] != before or all(p.grad == 0 for p in m.parameters()))
check("zero_grad", (m.zero_grad(), all(p.grad == 0 for p in m.parameters()))[1])

print("== generalization: structure vs random-label control (are we learning, or memorizing noise?) ==")
# Honest test (charter D1/D6): a model that only memorizes fits ANY labels on train but fails
# held-out. Real learning recovers the true rule and generalizes. We check BOTH must hold:
#   structured -> perfect held-out ;  random labels -> chance held-out (can't fake generalization).
import random as _rnd
from ringkit.linalg.solve import solve as _solve
_rnd.seed(11)
_d = 8
def _dot(a, b):
    s = 0
    for x, y in zip(a, b):
        s = (s + rn.qsm(x, y)) & 0xFF
    return s
def _vec(): return [_rnd.randint(0, 255) for _ in range(_d)]
while True:
    _X = [_vec() for _ in range(_d)]
    try:
        _solve(_X, [0 for _ in range(_d)]); break      # invertible design only
    except Exception:
        continue
_w = _vec()
_y_struct = [_dot(r, _w) for r in _X]
_y_rand = [_rnd.randint(0, 255) for _ in range(_d)]
_wh_s = _solve(_X, _y_struct)
_wh_r = _solve(_X, _y_rand)
check("both fit TRAIN exactly (memorizing is easy)",
      all(_dot(_X[i], _wh_s) == _y_struct[i] for i in range(_d)) and
      all(_dot(_X[i], _wh_r) == _y_rand[i] for i in range(_d)))
_held = [_vec() for _ in range(2000)]
_truth = [_dot(x, _w) for x in _held]
_acc_s = sum(_dot(x, _wh_s) == t for x, t in zip(_held, _truth)) / len(_held)
_acc_r = sum(_dot(x, _wh_r) == t for x, t in zip(_held, _truth)) / len(_held)
check("structured recovers TRUE rule -> held-out == 1.0", _wh_s == _w and _acc_s == 1.0)
check("random labels -> held-out at chance (<0.05, can't fake generalization)", _acc_r < 0.05)

print()
print("RESULT:", "ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
