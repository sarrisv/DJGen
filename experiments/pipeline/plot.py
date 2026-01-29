import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot():
    df = pd.read_csv("results.csv")
    df_melted = df.melt(
        id_vars=["Relations & Attributes"],
        value_vars=["Data Generation (s)", " Plan Generation (s)", " Analysis (s)"],
        var_name="Stage",
        value_name="Time (s)",
    )
    df_melted.loc[:, "Stage"] = df_melted["Stage"].str.strip()

    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(10, 6))

    sns.histplot(
        data=df_melted,
        x="Relations & Attributes",
        weights="Time (s)",
        hue="Stage",
        hue_order=["Analysis (s)", "Plan Generation (s)", "Data Generation (s)"],
        multiple="stack",
        discrete=True,
    )

    plt.title("Performance Breakdown by Size", fontsize=16, pad=15)
    plt.xlabel("Relations & Attributes", fontsize=12)
    plt.ylabel("Time (Seconds)", fontsize=12)

    plt.grid(True, which="both", ls="-", alpha=0.5)

    plt.tight_layout()
    plt.savefig("results.png", dpi=300)
    print("Plot saved as results.png")
    plt.show()


if __name__ == "__main__":
    plot()
