import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot():
    df = pd.read_csv("results.csv")

    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(10, 6))

    plot = sns.lineplot(data=df, x="Tuples", y="Time (s)", marker="o", linewidth=2.5)

    plot.set_xscale("log")

    plt.title("Execution Time vs. Number of Tuples", fontsize=16, pad=15)
    plt.xlabel("Size (Tuples) - Log Scale", fontsize=12)
    plt.ylabel("Time (Seconds)", fontsize=12)

    plt.grid(True, which="both", ls="-", alpha=0.5)

    plt.tight_layout()
    plt.savefig("results.png", dpi=300)
    print("Plot saved as results.png")
    plt.show()


if __name__ == "__main__":
    plot()
