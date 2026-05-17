#!/usr/bin/env python3

import numpy as np
import skfuzzy as fuzz

from collections import Counter, defaultdict
from sklearn.metrics import accuracy_score, f1_score

import utils


FEATURES = [
    "SocialMedia_WhileEating",
    "Your overeating level",
    "Employment_Status",
    "Low_Energy",
]


def train_fuzzy_cmeans(X, n_clusters=10):

    data = X[FEATURES].values.T

    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        data, c=n_clusters, m=2.0, error=0.005, maxiter=1000, init=None
    )

    print(f"[INFO] FPC: {fpc:.4f}")

    return cntr, u


def generate_pure_rules(X, y, cntr, u, purity_threshold=0.75):

    cluster_labels = np.argmax(u, axis=0)

    rules = []

    for k in range(cntr.shape[0]):
        idx = np.where(cluster_labels == k)[0]

        if len(idx) < 5:
            continue

        y_cluster = y.iloc[idx]
        counts = Counter(y_cluster)

        total = sum(counts.values())
        most_class, most_count = counts.most_common(1)[0]

        purity = most_count / total

        if purity < purity_threshold:
            continue

        probs = np.array(list(counts.values())) / total
        entropy = -np.sum(probs * np.log(probs + 1e-9))

        rules.append(
            {
                "cluster": k,
                "center": cntr[k],
                "output": most_class,
                "purity": purity,
                "entropy": entropy,
                "support": len(idx),
            }
        )

    rules.sort(key=lambda r: (r["purity"], r["support"]), reverse=True)

    return rules


def predict(X, rules):

    if len(rules) == 0:
        raise ValueError(
            "No rules generated. Lower purity_threshold or reduce n_clusters."
        )

    X_vals = X[FEATURES].values
    preds = []

    for x in X_vals:
        best_score = -1
        best_rule = None

        for r in rules:
            dist = np.linalg.norm(x - r["center"])
            dist = max(dist, 1e-6)

            score = r["purity"] * np.exp(-dist)

            if score > best_score:
                best_score = score
                best_rule = r

        if best_rule is None:
            best_rule = min(rules, key=lambda r: np.linalg.norm(x - r["center"]))

        preds.append(best_rule["output"])

    return np.array(preds)


def evaluate(y_true, y_pred):

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")

    print("\n=== RESULTS ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-score: {f1:.4f}")


def main():

    DATA_PATH = "data/raw/Mental Health Classification.csv"
    TARGET_COL = "Depression_Type"

    X, X_test, y, y_test = utils.load_and_split_stripped_data(DATA_PATH, TARGET_COL)

    print("[INFO] Training fuzzy c-means...")
    cntr, u = train_fuzzy_cmeans(X, n_clusters=50)

    print("[INFO] Generating PURE rules...")
    rules = generate_pure_rules(X, y, cntr, u, purity_threshold=0.7)

    print(f"[INFO] rules kept: {len(rules)}")

    print("\n=== TOP RULES ===")
    for r in rules[:10]:
        print(
            f"Cluster {r['cluster']} -> {r['output']} "
            f"(purity={r['purity']:.2f}, support={r['support']})"
        )

    print("\n[INFO] Predicting...")
    y_pred = predict(X_test, rules)

    evaluate(y_test, y_pred)


if __name__ == "__main__":
    main()
