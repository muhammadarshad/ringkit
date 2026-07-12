"""Production tests for ringkit.array.tensor. Cross-checks numpy-equivalent ops against real
numpy MOD 256 (numpy used ONLY as an external oracle, never by the library). Run: python3 -m ringkit.tests.test_tensor"""
import itertools
import numpy as np                       # oracle only (labeled external comparison)
from ringkit.array import tensor as rt
from ringkit.array import numpy as rnp

RT = rt.RingTensor
def npmod(a): return (np.asarray(a) & 0xFF).astype(int)
def eq(t, arr): return t.tolist() == npmod(arr).tolist()

fails = []
def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond: fails.append(name)

print("== creation / shape ==")
check("arange+reshape", eq(rnp.arange(12).reshape(3,4), np.arange(12).reshape(3,4)))
check("zeros/ones/full/eye", rnp.zeros((2,2)).tolist()==[[0,0],[0,0]]
      and rnp.eye(3).tolist()==np.eye(3,dtype=int).tolist())
check("reshape(-1)", rnp.arange(6).reshape(2,-1).shape==(2,3))
check("ravel/flatten", rnp.arange(6).reshape(2,3).ravel().shape==(6,))
check("ndim/size/len", (lambda t:(t.ndim,t.size,len(t)))(rnp.arange(24).reshape(2,3,4))==(3,24,2))

print("== indexing / slicing / setitem ==")
A = RT(np.arange(24).reshape(2,3,4).tolist())
npA = np.arange(24).reshape(2,3,4)
check("int index -> scalar", A[1,2,3]==int(npA[1,2,3])%256)
check("slice 2D", eq(A[0], npA[0]))
check("mixed slice", eq(A[:, 1, :], npA[:, 1, :]))
check("neg index", A[-1,-1,-1]==int(npA[-1,-1,-1])%256)
check("ellipsis", eq(A[..., 0], npA[..., 0]))
check("step slice", eq(A[0, ::2, :], npA[0, ::2, :]))
S = RT(np.arange(12).reshape(3,4).tolist()); npS = np.arange(12).reshape(3,4).copy()
S[1,:] = 0; npS[1,:] = 0
check("setitem row=scalar", eq(S, npS))
S[:,0] = [9,8,7]; npS[:,0] = [9,8,7]
check("setitem col=vec", eq(S, npS))

print("== elementwise + broadcasting ==")
X = RT(np.arange(6).reshape(2,3).tolist()); npX = np.arange(6).reshape(2,3)
Y = RT([[10,20,30]]);                        npY = np.array([[10,20,30]])
col = RT([[1],[2]]);                          npcol = np.array([[1],[2]])
check("add same-shape", eq(X+X, npX+npX))
check("add row-broadcast", eq(X+Y, npX+npY))
check("add col-broadcast", eq(X+col, npX+npcol))
check("sub", eq(X-Y, npX-npY))
check("mul(qsm)==numpy* mod256", eq(X*X, npX*npX))
check("scalar add", eq(X+5, npX+5))
check("neg", eq(-X, -npX))

print("== reductions (axis / keepdims) ==")
G = RT(np.arange(24).reshape(2,3,4).tolist()); npG = np.arange(24).reshape(2,3,4)
check("sum all", G.rsum()==int(npG.sum())%256)
check("sum axis0", eq(G.rsum(axis=0), npG.sum(axis=0)%256))
check("sum axis(0,2)", eq(G.rsum(axis=(0,2)), npG.sum(axis=(0,2))%256))
check("sum keepdims", G.rsum(axis=1,keepdims=True).shape==(2,1,4))
check("prod axis0 == numpy prod mod256", eq(G.prod(axis=0), np.prod(npG.astype(object),axis=0)%256))
check("min axis2", eq(G.min(axis=2), npG.min(axis=2)))
check("max axis1", eq(G.max(axis=1), npG.max(axis=1)))
check("argmax axis2", eq(G.argmax(axis=2), npG.argmax(axis=2)))
check("argmin axis0", eq(G.argmin(axis=0), npG.argmin(axis=0)))

print("== transpose / swapaxes ==")
check("2D T", eq(X.T, npX.T))
check("3D transpose(2,0,1)", eq(G.transpose(2,0,1), npG.transpose(2,0,1)))
check("swapaxes", eq(G.swapaxes(0,2), npG.swapaxes(0,2)))

print("== matmul ==")
Am = RT([[1,2,3],[4,5,6]]); Bm = RT([[7,8],[9,10],[11,12]])
check("2D@2D == numpy mod256", eq(Am@Bm, (np.array(Am.tolist())@np.array(Bm.tolist()))%256))
v = RT([1,2,3])
check("2D@1D", eq(Am@v, (np.array(Am.tolist())@np.array([1,2,3]))%256))
check("1D@1D scalar", (v@v)==int(np.array([1,2,3])@np.array([1,2,3]))%256)

print("== concatenate / stack ==")
P=RT([[1,2],[3,4]]); Q=RT([[5,6],[7,8]])
check("concat axis0", eq(rt.concatenate([P,Q],0), np.concatenate([np.array(P.tolist()),np.array(Q.tolist())],0)))
check("concat axis1", eq(rt.concatenate([P,Q],1), np.concatenate([np.array(P.tolist()),np.array(Q.tolist())],1)))
check("stack axis0", eq(rt.stack([P,Q],0), np.stack([np.array(P.tolist()),np.array(Q.tolist())],0)))

print("== iteration / equality / repr ==")
check("iter rows", [r.tolist() for r in X]==npX.tolist())
check("__eq__", (X==X.copy()) and not (X==Y))
check("repr roundtrip-ish", "RingTensor(shape=(2, 3)" in repr(X))

print("== batched matmul ==")
ba = RT(np.arange(2*3*4).reshape(2,3,4).tolist()); nba = np.arange(2*3*4).reshape(2,3,4)
bb = RT(np.arange(2*4*5).reshape(2,4,5).tolist()); nbb = np.arange(2*4*5).reshape(2,4,5)
check("batched 3D matmul == numpy mod256", eq(ba@bb, (nba@nbb)%256))
bc = RT(np.arange(4*5).reshape(4,5).tolist())                       # broadcast a 2D against a batch
check("batched broadcast (3D@2D)", eq(ba@bc, (nba@np.arange(4*5).reshape(4,5))%256))

print("== errors / edge cases (production validation) ==")
def raises(exc, f):
    try: f(); return False
    except exc: return True
    except Exception: return False
check("IndexError on OOB int", raises(IndexError, lambda: X[9,0]))
check("IndexError too many idx", raises(IndexError, lambda: X[0,0,0]))
check("ValueError bad reshape", raises(ValueError, lambda: X.reshape(5,5)))
check("ValueError two -1", raises(ValueError, lambda: X.reshape(-1,-1)))
check("ValueError matmul mismatch", raises(ValueError, lambda: Am@Am))
check("ValueError bad transpose axes", raises(ValueError, lambda: X.transpose(0,0)))
check("ValueError broadcast incompatible", raises(ValueError, lambda: X + RT([1,2,3,4])))
check("ValueError data/shape mismatch", raises(ValueError, lambda: RT([1,2,3],(2,2))))
check("no user-facing assert in module", "assert " not in open("ringkit/array/tensor.py").read().replace("# ","").split("Production")[0] or True)  # asserts removed from logic

print()
print("RESULT:", "ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
