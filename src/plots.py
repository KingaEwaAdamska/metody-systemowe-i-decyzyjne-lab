from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

from utils import evaluate_model, train_model


def test_random_forest_estimators(X_train, y_train, X_test, y_test):
    estimator_counts = [10, 25, 50, 100, 200, 500]
    results = []

    plt.figure(figsize=(10, 6))

    for n in estimator_counts:
        model = RandomForestClassifier(random_state=42, n_estimators=n)
        trained_model = train_model(model, X_train, y_train)
        metrics = evaluate_model(trained_model, X_test, y_test)

        results.append(
            {
                "n_estimators": n,
                "accuracy": metrics["accuracy"],
                "f1_score": metrics["f1_score"],
            }
        )

    results_df = pd.DataFrame(results)
    plt.plot(
        results_df["n_estimators"], results_df["accuracy"], marker="o", label="Accuracy"
    )
    plt.plot(
        results_df["n_estimators"], results_df["f1_score"], marker="o", label="F1 Score"
    )
    plt.title("Random Forest Performance vs Number of Estimators")
    plt.xlabel("Number of Estimators")
    plt.ylabel("Score")
    plt.legend()
    plt.savefig(f"plots/{datetime.now().strftime('%Y%m%d_%H%M%S')}_rf_estimators.png")


def test_random_forest_depth(X_train, y_train, X_test, y_test):
    max_depths = [5, 10, 20, 50, 100]
    results = []

    plt.figure(figsize=(10, 6))

    for depth in max_depths:
        model = RandomForestClassifier(
            random_state=42, n_estimators=100, max_depth=depth
        )
        trained_model = train_model(model, X_train, y_train)
        metrics = evaluate_model(trained_model, X_test, y_test)

        results.append(
            {
                "max_depth": depth if depth is not None else "None",
                "accuracy": metrics["accuracy"],
                "f1_score": metrics["f1_score"],
            }
        )

    results_df = pd.DataFrame(results)
    plt.plot(
        results_df["max_depth"], results_df["accuracy"], marker="o", label="Accuracy"
    )
    plt.plot(
        results_df["max_depth"], results_df["f1_score"], marker="o", label="F1 Score"
    )
    plt.title("Random Forest Performance vs Max Depth")
    plt.xlabel("Max Depth")
    plt.ylabel("Score")
    plt.legend()
    plt.savefig(f"plots/{datetime.now().strftime('%Y%m%d_%H%M%S')}_rf_max_depth.png")


def test_random_forest_depth_and_estimators(X_train, y_train, X_test, y_test):
    max_depths = [5, 10, 20, 50, 100]
    estimator_counts = [10, 25, 50, 100, 200, 500]
    results = []

    # 3d plot
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    for depth in max_depths:
        for n in estimator_counts:
            model = RandomForestClassifier(
                random_state=42, n_estimators=n, max_depth=depth
            )
            trained_model = train_model(model, X_train, y_train)
            metrics = evaluate_model(trained_model, X_test, y_test)

            results.append(
                {
                    "max_depth": depth if depth is not None else "None",
                    "n_estimators": n,
                    "accuracy": metrics["accuracy"],
                    "f1_score": metrics["f1_score"],
                }
            )
    results_df = pd.DataFrame(results)
    ax.scatter(
        results_df["max_depth"],
        results_df["n_estimators"],
        results_df["accuracy"],
        label="Accuracy",
    )
    ax.scatter(
        results_df["max_depth"],
        results_df["n_estimators"],
        results_df["f1_score"],
        label="F1 Score",
    )
    ax.set_title("Random Forest Performance vs Max Depth and Number of Estimators")
    ax.set_xlabel("Max Depth")
    ax.set_ylabel("Number of Estimators")
    ax.set_zlabel("Score")
    ax.legend()
    plt.savefig(
        f"plots/{datetime.now().strftime('%Y%m%d_%H%M%S')}_rf_depth_estimators.png"
    )


def test_decision_tree_depth(X_train, y_train, X_test, y_test):
    max_depths = [5, 10, 20, 50, 100]
    results = []

    plt.figure(figsize=(10, 6))

    for depth in max_depths:
        model = DecisionTreeClassifier(random_state=42, max_depth=depth)
        trained_model = train_model(model, X_train, y_train)
        metrics = evaluate_model(trained_model, X_test, y_test)

        results.append(
            {
                "max_depth": depth if depth is not None else "None",
                "accuracy": metrics["accuracy"],
                "f1_score": metrics["f1_score"],
            }
        )

    results_df = pd.DataFrame(results)
    plt.plot(
        results_df["max_depth"], results_df["accuracy"], marker="o", label="Accuracy"
    )
    plt.plot(
        results_df["max_depth"], results_df["f1_score"], marker="o", label="F1 Score"
    )
    plt.title("Decision Tree Performance vs Max Depth")
    plt.xlabel("Max Depth")
    plt.ylabel("Score")
    plt.legend()
    plt.savefig(
        f"plots/{datetime.now().strftime('%Y%m%d_%H%M%S')}_decision_tree_depth.png"
    )


def test_logistic_regression(X_train, y_train, X_test, y_test):
    C_values = [0.01, 0.1, 1, 10, 100]
    results = []

    plt.figure(figsize=(10, 6))

    for C in C_values:
        model = LogisticRegression(random_state=42, max_iter=1000, C=C)
        trained_model = train_model(model, X_train, y_train)
        metrics = evaluate_model(trained_model, X_test, y_test)

        results.append(
            {"C": C, "accuracy": metrics["accuracy"], "f1_score": metrics["f1_score"]}
        )

    results_df = pd.DataFrame(results)
    plt.plot(results_df["C"], results_df["accuracy"], marker="o", label="Accuracy")
    plt.plot(results_df["C"], results_df["f1_score"], marker="o", label="F1 Score")
    plt.xscale("log")
    plt.title("Logistic Regression Performance vs C")
    plt.xlabel("C (Inverse of Regularization Strength)")
    plt.ylabel("Score")
    plt.legend()
    plt.savefig(
        f"plots/{datetime.now().strftime('%Y%m%d_%H%M%S')}_logistic_regression_C.png"
    )


def test_hist_gradient_boosting(X_train, y_train, X_test, y_test):
    learning_rates = [0.01, 0.1, 0.2, 0.3, 0.5, 1, 1.5, 10]
    results = []

    plt.figure(figsize=(10, 6))

    for lr in learning_rates:
        model = HistGradientBoostingClassifier(random_state=42, learning_rate=lr)
        trained_model = train_model(model, X_train, y_train)
        metrics = evaluate_model(trained_model, X_test, y_test)

        results.append(
            {
                "learning_rate": lr,
                "accuracy": metrics["accuracy"],
                "f1_score": metrics["f1_score"],
            }
        )

    results_df = pd.DataFrame(results)
    plt.plot(
        results_df["learning_rate"],
        results_df["accuracy"],
        marker="o",
        label="Accuracy",
    )
    plt.plot(
        results_df["learning_rate"],
        results_df["f1_score"],
        marker="o",
        label="F1 Score",
    )
    plt.xscale("log")
    plt.title("HistGradientBoosting Performance vs Learning Rate")
    plt.xlabel("Learning Rate")
    plt.ylabel("Score")
    plt.legend()
    plt.savefig(
        f"plots/{datetime.now().strftime('%Y%m%d_%H%M%S')}_hist_gradient_boosting_lr.png"
    )
