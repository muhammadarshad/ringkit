"""Test for ringkit.rlearn — ring-native classical ML (sklearn-shaped, NO Euclidean).

Honesty bar (CLAUDE.md): a method must generalize on data it planted structure into, AND a control
must fail — random labels collapse KNN to chance; KMeans recovers planted clusters but not noise.
Run: python3 -m ringkit.tests.test_rlearn"""
import random
import ringkit as rk

fails = []
def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond: fails.append(name)

random.seed(3)
DIM = 4
CENTERS = [[20, 40, 60, 80], [130, 150, 170, 190], [230, 5, 15, 25]]  # 3 planted centers on the ring

def near(center, spread=6):
    return [(c + random.randint(-spread, spread)) & 0xFF for c in center]

print("== KMeans: recovers PLANTED ring clusters (points near 3 centers) ==")
X, true_lab = [], []
for lbl, ctr in enumerate(CENTERS):
    for _ in range(40):
        X.append(near(ctr)); true_lab.append(lbl)
km = rk.rlearn.KMeans(n_clusters=3, seed=1).fit(X)
# cluster purity: each learned cluster should be dominated by one true label
def purity(pred, truth, k):
    tot = 0
    for c in range(k):
        members = [truth[i] for i in range(len(truth)) if pred[i] == c]
        if members:
            tot += max(members.count(t) for t in set(members))
    return tot / len(truth)
pur = purity(km.labels_, true_lab, 3)
check(f"planted clusters recovered: purity {pur:.3f} >= 0.95", pur >= 0.95)
# CONTROL: uniform noise has no cluster structure -> purity collapses toward chance (~1/3)
Xn = [[random.randint(0, 255) for _ in range(DIM)] for _ in range(120)]
noise_lab = [random.randint(0, 2) for _ in range(120)]   # meaningless labels
kmn = rk.rlearn.KMeans(n_clusters=3, seed=1).fit(Xn)
pur_noise = purity(kmn.predict(Xn), noise_lab, 3)
check(f"CONTROL: on noise vs random labels, purity {pur_noise:.3f} near chance (< 0.55)", pur_noise < 0.55)

print("== KNeighborsClassifier: generalizes held-out; random-label control fails ==")
# teacher rule: label = nearest planted center (by ring distance)
def teach(x): return min(range(3), key=lambda c: rk.rlearn.ring_dist_vec(x, CENTERS[c]))
Xtr = [near(CENTERS[random.randint(0, 2)]) for _ in range(150)]
ytr = [teach(x) for x in Xtr]
Xte = [near(CENTERS[random.randint(0, 2)]) for _ in range(300)]   # unseen points
yte = [teach(x) for x in Xte]
knn = rk.rlearn.KNeighborsClassifier(n_neighbors=5).fit(Xtr, ytr)
acc = rk.rlearn.accuracy_score(yte, knn.predict(Xte))
check(f"held-out accuracy {acc:.3f} >= 0.95 (learned the ring structure)", acc >= 0.95)
knn_r = rk.rlearn.KNeighborsClassifier(n_neighbors=5).fit(Xtr, [random.randint(0, 2) for _ in ytr])
acc_r = rk.rlearn.accuracy_score(yte, knn_r.predict(Xte))
check(f"random-label control {acc_r:.3f} near chance (< 0.45)", acc_r < 0.45)
print(f"    [held-out] structured={acc:.3f}  random-control={acc_r:.3f}  chance~1/3=0.333")

print("== ring geometry, not Euclidean; guards ==")
# distance wraps: 250 and 5 are 11 apart on the ring, not 245
check("ring distance wraps (dist(250,5)=11, not 245)", rk.rlearn.ring_dist_vec([250], [5]) == 11)
def raises(f):
    try: f(); return False
    except ValueError: return True
    except Exception: return False
check("KMeans predict before fit -> ValueError", raises(lambda: rk.rlearn.KMeans().predict([[1, 2, 3, 4]])))
check("accuracy_score empty -> ValueError", raises(lambda: rk.rlearn.accuracy_score([], [])))

print()
print("RESULT:", "ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
