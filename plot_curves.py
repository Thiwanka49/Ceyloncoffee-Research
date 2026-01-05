import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_combined(csv_path, out_png, title):
    df = pd.read_csv(csv_path)
    epochs = df["epoch"]

    fig = plt.figure(figsize=(12, 4))

    # Accuracy subplot
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(epochs, df["train_acc"], label="Training Accuracy")
    ax1.plot(epochs, df["val_acc"], label="Validation Accuracy")
    ax1.set_title("Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Loss subplot
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(epochs, df["train_loss"], label="Training Loss")
    ax2.plot(epochs, df["val_loss"], label="Validation Loss")
    ax2.set_title("Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print("✅ Saved:", out_png)

if __name__ == "__main__":
    os.makedirs("plots", exist_ok=True)

    if os.path.exists("runs/type_history.csv"):
        plot_combined(
            "runs/type_history.csv",
            "plots/type_combined.png",
            "Bean Type Model (Arabica vs Robusta)"
        )
    else:
        print("⚠️ Missing runs/type_history.csv")

    if os.path.exists("runs/defect_history.csv"):
        plot_combined(
            "runs/defect_history.csv",
            "plots/defect_combined.png",
            "Defect Model (Broken / Good / Severe Defect)"
        )
    else:
        print("⚠️ Missing runs/defect_history.csv (train defect model or create this log)")
