from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIRECTORY = Path("technical_report_figures")
OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)


model_results = {
    "Email": {
        "Naive Bayes": {
            "Accuracy": 99.46,
            "Precision": 100.00,
            "Recall": 98.92,
            "F1 Score": 99.46,
        },
        "Random Forest": {
            "Accuracy": 98.92,
            "Precision": 98.92,
            "Recall": 98.92,
            "F1 Score": 98.92,
        },
        "Logistic Regression": {
            "Accuracy": 98.92,
            "Precision": 100.00,
            "Recall": 97.84,
            "F1 Score": 98.91,
        },
        "XGBoost": {
            "Accuracy": 97.84,
            "Precision": 98.90,
            "Recall": 96.76,
            "F1 Score": 97.81,
        },
    },

    "SMS": {
        "Naive Bayes": {
            "Accuracy": 97.55,
            "Precision": 95.35,
            "Recall": 100.00,
            "F1 Score": 97.62,
        },
        "Logistic Regression": {
            "Accuracy": 97.55,
            "Precision": 96.43,
            "Recall": 98.78,
            "F1 Score": 97.59,
        },
        "Random Forest": {
            "Accuracy": 96.32,
            "Precision": 96.34,
            "Recall": 96.34,
            "F1 Score": 96.34,
        },
        "XGBoost": {
            "Accuracy": 93.87,
            "Precision": 97.37,
            "Recall": 90.24,
            "F1 Score": 93.67,
        },
    },

    "URL": {
        "Logistic Regression": {
            "Accuracy": 90.11,
            "Precision": 91.47,
            "Recall": 88.48,
            "F1 Score": 89.95,
        },
        "Random Forest": {
            "Accuracy": 89.20,
            "Precision": 90.25,
            "Recall": 87.89,
            "F1 Score": 89.05,
        },
        "XGBoost": {
            "Accuracy": 89.11,
            "Precision": 89.26,
            "Recall": 88.92,
            "F1 Score": 89.09,
        },
    },
}


metric_names = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
]


def create_model_graph(
    scanner_name,
    scanner_results,
):
    model_names = list(
        scanner_results.keys()
    )

    positions = np.arange(
        len(model_names)
    )

    bar_width = 0.19

    colours = [
        "#1769C7",
        "#16A34A",
        "#F59E0B",
        "#DC2626",
    ]

    figure, axis = plt.subplots(
        figsize=(11, 6.5)
    )

    for metric_index, metric_name in enumerate(
        metric_names
    ):
        values = [
            scanner_results[model_name][metric_name]
            for model_name in model_names
        ]

        bar_positions = (
            positions
            + (
                metric_index
                - 1.5
            )
            * bar_width
        )

        bars = axis.bar(
            bar_positions,
            values,
            width=bar_width,
            label=metric_name,
            color=colours[metric_index],
        )

        axis.bar_label(
            bars,
            labels=[
                f"{value:.2f}"
                for value in values
            ],
            padding=3,
            fontsize=7,
            rotation=90,
        )

    axis.set_title(
        f"{scanner_name} Model Performance",
        fontsize=16,
        fontweight="bold",
        pad=16,
    )

    axis.set_xlabel(
        "Machine Learning Model",
        fontsize=11,
    )

    axis.set_ylabel(
        "Performance (%)",
        fontsize=11,
    )

    axis.set_xticks(
        positions
    )

    axis.set_xticklabels(
        model_names,
        fontsize=9,
    )

    minimum_value = min(
        result[metric_name]
        for result in scanner_results.values()
        for metric_name in metric_names
    )

    axis.set_ylim(
        max(0, minimum_value - 5),
        103,
    )

    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.35,
    )

    axis.legend(
        loc="lower left",
        ncol=2,
        frameon=True,
    )

    figure.tight_layout()

    output_filename = (
        f"{scanner_name.lower()}_model_performance.png"
    )

    output_path = (
        OUTPUT_DIRECTORY
        / output_filename
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    print(
        f"Created: {output_path}"
    )


def create_accuracy_comparison_graph():
    scanner_names = list(
        model_results.keys()
    )

    selected_models = [
        "Naive Bayes",
        "Naive Bayes",
        "Logistic Regression",
    ]

    accuracy_values = [
        model_results["Email"]["Naive Bayes"]["Accuracy"],
        model_results["SMS"]["Naive Bayes"]["Accuracy"],
        model_results["URL"]["Logistic Regression"]["Accuracy"],
    ]

    figure, axis = plt.subplots(
        figsize=(8.5, 5.5)
    )

    bars = axis.bar(
        scanner_names,
        accuracy_values,
        color=[
            "#1769C7",
            "#16A34A",
            "#F59E0B",
        ],
        width=0.58,
    )

    axis.bar_label(
        bars,
        labels=[
            f"{value:.2f}%"
            for value in accuracy_values
        ],
        padding=5,
        fontsize=10,
        fontweight="bold",
    )

    axis.set_title(
        "Performance of the Selected Models",
        fontsize=16,
        fontweight="bold",
        pad=16,
    )

    axis.set_xlabel(
        "Scanner",
        fontsize=11,
    )

    axis.set_ylabel(
        "Validation Accuracy (%)",
        fontsize=11,
    )

    axis.set_ylim(
        80,
        102,
    )

    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.35,
    )

    for position, model_name in enumerate(
        selected_models
    ):
        axis.text(
            position,
            81,
            model_name,
            ha="center",
            va="bottom",
            fontsize=9,
        )

    figure.tight_layout()

    output_path = (
        OUTPUT_DIRECTORY
        / "selected_models_accuracy.png"
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    print(
        f"Created: {output_path}"
    )


def main():
    for scanner_name, scanner_results in (
        model_results.items()
    ):
        create_model_graph(
            scanner_name,
            scanner_results,
        )

    create_accuracy_comparison_graph()

    print(
        "\nAll model performance graphs "
        "were generated successfully."
    )


if __name__ == "__main__":
    main()