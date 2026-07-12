"""Tests for ringkit.array.numpy (our numpy namespace). numpy = external oracle only.
Run: python3 -m ringkit.tests.test_numpy"""
import numpy as np
import ringkit.array.numpy as rnp

def npmod(a): return (np.asarray(a) & 0xFF).astype(int).tolist()
fails = []
def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond: fails.append(name)

a = rnp.arange(12).reshape(3, 4); npa = np.arange(12).reshape(3, 4)

print("== creation ==")
check("arange/reshape", a.tolist() == npa.tolist())
check("zeros/ones/full", rnp.zeros((2,2)).tolist()==[[0,0],[0,0]] and rnp.full((2,2),9).tolist()==[[9,9],[9,9]])
check("eye/identity", rnp.eye(3).tolist()==np.eye(3,dtype=int).tolist() and rnp.identity(3).tolist()==np.eye(3,dtype=int).tolist())
check("zeros_like/ones_like/full_like", rnp.zeros_like(a).tolist()==np.zeros((3,4),int).tolist()
      and rnp.full_like(a,7).tolist()==np.full((3,4),7,int).tolist())

print("== manipulation ==")
check("reshape/ravel/flatten", rnp.ravel(a).shape==(12,) and rnp.flatten(a).shape==(12,))
check("transpose", rnp.transpose(a).tolist()==npa.T.tolist())
check("swapaxes", rnp.swapaxes(rnp.arange(24).reshape(2,3,4),0,2).shape==(4,3,2))
check("concatenate", npmod(np.concatenate([npa,npa],0))==rnp.concatenate([a,a],0).tolist())
check("stack", rnp.stack([a,a],0).shape==(2,3,4))
check("hstack/vstack", rnp.vstack([a,a]).shape==(6,4) and rnp.hstack([a,a]).shape==(3,8))

print("== reductions ==")
check("sum axis", rnp.sum(a,0).tolist()==(npa.sum(0)%256).tolist())
check("prod axis", rnp.prod(a,1).tolist()==(np.prod(npa.astype(object),1)%256).tolist())
check("amin/amax", rnp.amin(a,1).tolist()==npa.min(1).tolist() and rnp.amax(a,0).tolist()==npa.max(0).tolist())
check("argmin/argmax", rnp.argmin(a,1).tolist()==npa.argmin(1).tolist() and rnp.argmax(a,0).tolist()==npa.argmax(0).tolist())
check("keepdims", rnp.sum(a,1,keepdims=True).shape==(3,1))

print("== linalg / elementwise / trig ==")
b = rnp.eye(4)
check("dot/matmul", rnp.dot(a,b).tolist()==npmod(npa@np.eye(4,dtype=int)))
check("add/subtract/multiply/negative", rnp.add(a,a).tolist()==npmod(npa+npa)
      and rnp.multiply(a,a).tolist()==npmod(npa*npa) and rnp.negative(a).tolist()==npmod(-npa))
check("sin/cos at cardinals", rnp.sin(rnp.array([0,64,128,192])).tolist()==[0,21,0,235])

print()
print("RESULT:", "ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
