import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

from utils import load_and_split_data, train_model, evaluate_model, save_model


def main():
    DATA_PATH = "data/raw/Mental Health Classification.csv"
    TARGET_COL = "Depression_Type"

    X_train, X_test, y_train, y_test = load_and_split_data(DATA_PATH, TARGET_COL)

    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42),
    }

    results = []

    for name, model in models.items():
        print(f"Training {name}...")
        trained_model = train_model(model, X_train, y_train)

        metrics = evaluate_model(trained_model, X_test, y_test)

        results.append(
            {
                "Model": name,
                "Accuracy": metrics["accuracy"],
                "F1 Score": metrics["f1_score"],
                "Classification Report": metrics["classification_report"],
            }
        )

        save_model(trained_model, f"models/{name.replace(' ', '_').lower()}.pkl")

    results_df = pd.DataFrame(results).sort_values(by="F1 Score", ascending=False)
    print("\nModel Performance:")
    print(results_df[["Model", "Accuracy", "F1 Score"]])


if __name__ == "__main__":
    main()
