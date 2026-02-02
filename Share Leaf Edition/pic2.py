# plot_skills.py
# Run in VS Code:  python plot_skills.py
# Output: skills_on_the_rise_replica.png (in the same folder as this script)

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def main(show: bool = False) -> Path:
    # ----------------------------
    # 1) Data (replace with your numbers)
    # ----------------------------
    data = [
        ("Creative thinking", 73.2, 4.0, "skill"),
        ("Analytical thinking", 71.6, 4.5, "skill"),
        ("Technological literacy", 67.7, 5.0, "skill"),
        ("Curiosity and lifelong learning", 66.8, 4.0, "attitude"),
        ("Resilience, flexibility and agility", 66.8, 5.0, "attitude"),
        ("Systems thinking", 59.9, 5.5, "skill"),
        ("AI and big data", 59.5, 8.0, "skill"),
        ("Motivation and self-awareness", 58.9, 4.0, "attitude"),
        ("Talent management", 56.4, 5.0, "skill"),
        ("Service orientation and customer service", 54.8, 6.0, "skill"),
        ("Leadership and social influence", 53.1, 6.0, "attitude"),
        ("Empathy and active listening", 52.3, 6.0, "attitude"),
        ("Dependability and attention to detail", 52.0, 7.0, "attitude"),
        ("Resource management and operations", 51.4, 7.0, "skill"),
        ("Networks and cybersecurity", 50.3, 5.0, "skill"),
        ("Quality control", 49.5, 6.5, "skill"),
        ("Design and user experience", 48.4, 6.0, "skill"),
        ("Teaching and mentoring", 47.8, 6.0, "attitude"),
        ("Environmental stewardship", 43.2, 5.0, "attitude"),
        ("Programming", 38.8, 6.0, "skill"),
        ("Marketing and media", 38.4, 6.0, "skill"),
        ("Multi-lingualism", 38.0, 6.0, "skill"),
        ("Reading, writing and mathematics", 26.4, 6.0, "skill"),
        ("Global citizenship", 23.8, 7.0, "attitude"),
        ("Sensory-processing abilities", 22.6, 9.0, "skill"),
        ("Manual dexterity, endurance and precision", 14.9, 11.0, "skill"),
    ]

    df = pd.DataFrame(data, columns=["skill", "inc", "dec", "kind"])
    df["net"] = df["inc"] - df["dec"]
    df = df.sort_values("net", ascending=True).reset_index(drop=True)

    # ----------------------------
    # 2) Plot styling
    # ----------------------------
    INC_COLOR = "#9CCBEA"  # light blue
    DEC_COLOR = "#9C86B3"  # muted purple
    EDGE = "white"

    fig, ax = plt.subplots(figsize=(10.5, 11.5), dpi=150)

    y = range(len(df))

    # Declining shown to the left (negative)
    ax.barh(
        y,
        -df["dec"],
        color=DEC_COLOR,
        edgecolor=EDGE,
        height=0.65,
        label="Declining importance",
    )

    # Increasing shown to the right (positive)
    ax.barh(
        y,
        df["inc"],
        color=INC_COLOR,
        edgecolor=EDGE,
        height=0.65,
        label="Increasing importance",
    )

    # Marker + % label on the right end (WEF-like)
    for i, row in enumerate(df.itertuples(index=False)):
        marker = "D" if row.kind == "skill" else "o"
        ax.scatter(
            row.inc,
            i,
            marker=marker,
            s=55,
            facecolors="none",
            edgecolors="#2C2C2C",
            linewidths=1.2,
            zorder=5,
        )
        ax.text(
            row.inc + 1.2,
            i,
            f"{row.inc:.1f}%",
            va="center",
            ha="left",
            fontsize=9,
            color="#2C2C2C",
        )

    # Axis / grid
    ax.axvline(0, color="#666666", linewidth=1)
    ax.set_xlim(-25, 100)  # adjust if your dec values exceed 25
    ax.set_xticks([-25, 0, 25, 50, 75, 100])
    ax.grid(axis="x", linestyle="-", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)

    # Labels
    ax.set_yticks(list(y))
    ax.set_yticklabels(df["skill"], fontsize=10)
    ax.set_xlabel("Share of companies surveyed (%)", fontsize=11)
    ax.set_title("Skills on the rise", fontsize=14, pad=10)

    # Legend (bars + marker meaning)
    legend_items = [
        Line2D([0], [0], color=INC_COLOR, lw=10, label="Increasing importance"),
        Line2D([0], [0], color=DEC_COLOR, lw=10, label="Declining importance"),
        Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            markeredgecolor="#2C2C2C",
            markerfacecolor="none",
            markersize=7,
            label="Skills, knowledge and abilities",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markeredgecolor="#2C2C2C",
            markerfacecolor="none",
            markersize=7,
            label="Attitudes",
        ),
    ]
    ax.legend(handles=legend_items, loc="lower right", frameon=False, fontsize=9)

    # Save to the same folder as this script (works reliably in VS Code)
    out_path = Path(__file__).resolve().parent / "skills_on_the_rise_replica.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[OK] Saved figure to: {out_path}")

    # Optional show (set show=True if you want the window/inline plot)
    if show:
        plt.show()
    else:
        plt.close(fig)

    return out_path


if __name__ == "__main__":
    # show=False: recommended for running in VS Code terminal (always saves image, doesn't block)
    # show=True: if you want it to pop up / display inline in the interactive window
    main(show=False)