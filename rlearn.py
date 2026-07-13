"""
ringkit.rlearn — ring-native classical ML (the sklearn-shaped layer), on Z256.

NOT scikit-learn, and NOT Euclidean. Every distance is the RING distance (shortest arc on Z256, via
stats.ring_dist), every centroid is the CIRCULAR mean (angles wrap — a linear mean of 255 and 1 is
the antipode 128, which is wrong; stats.circular_mean is exact). There is no `sqrt(x^2+y^2)`, no
Euclidean norm, anywhere — that is the MPRC anti-pattern (Prime Directive / C9). The API is
sklearn-shaped (fit / predict / accuracy_score) so an engineer uses it familiarly; the geometry is
the ring's.

The ring geometry (distance, centroid) is multiplier-free and composed from the verified ring
primitives. accuracy_score returns an ordinary float — it is a reporting metric, not a ring value.
"""
from ringkit.stats import stats as rs
import random as _random


def ring_dist_vec(a, b):
    """Ring L1 metric: sum of per-coordinate shortest-arc distances between two ring vectors. No
    Euclidean square-root — distance on the ring is the arc, summed over coordinates."""
    s = 0
    for x, y in zip(a, b):
        s += rs.ring_dist(x, y)
    return s


class KMeans:
    """Ring k-means. Assign each point to the nearest centroid by RING distance; update each centroid
    as the CIRCULAR mean of its members, per coordinate. Data-free, ring-native (no Euclidean)."""

    def __init__(self, n_clusters=3, max_iter=25, seed=0):
        self.k = int(n_clusters)
        self.max_iter = int(max_iter)
        self.seed = int(seed)
        self.cluster_centers_ = None
        self.labels_ = None

    def fit(self, X):
        if len(X) < self.k:
            raise ValueError(f"KMeans: need >= n_clusters points, got {len(X)} < {self.k}")
        rng = _random.Random(self.seed)
        centers = [list(X[i]) for i in rng.sample(range(len(X)), self.k)]
        labels = [0 for _ in X]
        dim = len(X[0])
        for _ in range(self.max_iter):
            changed = False
            for i, x in enumerate(X):
                best = min(range(self.k), key=lambda c: ring_dist_vec(x, centers[c]))
                if best != labels[i]:
                    changed = True
                labels[i] = best
            new_centers = []
            for c in range(self.k):
                members = [X[i] for i in range(len(X)) if labels[i] == c]
                if not members:
                    new_centers.append(centers[c])
                    continue
                new_centers.append([rs.circular_mean([m[d] for m in members]) for d in range(dim)])
            centers = new_centers
            if not changed:
                break
        self.cluster_centers_ = centers
        self.labels_ = labels
        return self

    def predict(self, X):
        if self.cluster_centers_ is None:
            raise ValueError("KMeans: call fit() before predict()")
        return [min(range(self.k), key=lambda c: ring_dist_vec(x, self.cluster_centers_[c])) for x in X]


class KNeighborsClassifier:
    """Ring k-NN classifier: label a point by majority vote of its k nearest neighbours under the
    RING distance. The classic non-parametric classifier, ring-native."""

    def __init__(self, n_neighbors=3):
        self.k = int(n_neighbors)
        self._X = None
        self._y = None

    def fit(self, X, y):
        if len(X) != len(y):
            raise ValueError(f"fit: X/y length mismatch {len(X)} vs {len(y)}")
        self._X = [list(r) for r in X]
        self._y = list(y)
        return self

    def predict(self, X):
        if self._X is None:
            raise ValueError("KNeighborsClassifier: call fit() before predict()")
        out = []
        for x in X:
            order = sorted(range(len(self._X)), key=lambda i: ring_dist_vec(x, self._X[i]))
            votes = {}
            for i in order[:self.k]:
                votes[self._y[i]] = votes.get(self._y[i], 0) + 1
            out.append(max(votes, key=lambda lbl: votes[lbl]))
        return out


def accuracy_score(y_true, y_pred):
    """Fraction correct (a reporting metric, ordinary float — not a ring value)."""
    n = len(y_true)
    if n == 0:
        raise ValueError("accuracy_score: empty")
    hits = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    return hits / n
