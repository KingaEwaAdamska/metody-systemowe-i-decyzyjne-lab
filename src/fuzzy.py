#!/usr/bin/env python3

import utils
from fuzzy_system_definition import depression_sim as fuzzy_sim

from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def predict_fuzzy(X, y):
    print(
        f"SocialMedia_WhileEating, Your overeating level, Employment_Status, Low_Energy, Result, True_value"
    )
    i = 0
    results = []
    for idx, row in X.iterrows():
        fuzzy_sim.reset()

        fuzzy_sim.input["social_media_while_eating"] = row["SocialMedia_WhileEating"]
        fuzzy_sim.input["overeating_level"] = row["Your overeating level"]
        # fuzzy_sim.input["employment_status"] = row["Employment_Status"]
        fuzzy_sim.input["low_energy"] = row["Low_Energy"]

        try:
            fuzzy_sim.compute()
            result = fuzzy_sim.output["depression_type"]
        except KeyError:
            result = 4.0

        print(
            f"{row['SocialMedia_WhileEating']}, {row['Your overeating level']}, {row['Employment_Status']}, {row['Low_Energy']}, {round(result)}, {y[idx]}"
        )
        if round(result) == y[idx]:
            i += 1
        results.append(round(result))

    print(f"matches: {i}")
    return results


def main():
    DATA_PATH = "data/raw/Mental Health Classification.csv"
    TARGET_COL = "Depression_Type"

    X, y = utils.load_data(DATA_PATH, TARGET_COL)

    y_pred = predict_fuzzy(X, y)

    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average="weighted")
    report = classification_report(y, y_pred)

    print("\n=== FUZZY C-MEANS RESULTS ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:\n", report)


if __name__ == "__main__":
    main()
