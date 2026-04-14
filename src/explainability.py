import shap
import matplotlib.pyplot as plt
import os

def explain_model(model, X_test, feature_names: list):
    os.makedirs("outputs", exist_ok=True)

    print("Calculating SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    print("Done.")

    return explainer, shap_values

def plot_global_importance(shap_values, X_test, feature_names: list):
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=feature_names,
        show=False
    )
    plt.title("What drives Apple stock movement predictions?")
    plt.tight_layout()
    plt.savefig("outputs/shap_global.png", dpi=150)
    plt.close()
    print("Saved: outputs/shap_global.png")

def plot_single_day(explainer, shap_values, X_test,
                    feature_names: list, index: int):
    print(f"\nExplaining prediction for day #{index}...")

    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values[index],
            base_values=explainer.expected_value,
            data=X_test[index],
            feature_names=feature_names
        ),
        show=False
    )
    plt.tight_layout()
    plt.savefig(f"outputs/shap_day_{index}.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: outputs/shap_day_{index}.png")