import matplotlib.pyplot as plt
import numpy as np

from hopfield_network import HopfieldNetwork
from letters import letters

plt.rcParams["font.size"] = 18


if __name__ == "__main__":
    letters_key = list(letters.keys())
    experiments = [
        ["CDJ", "CDJM", "CDJMX", "CDJMXS"],
        ["CEF", "CEFG", "CEFGH", "CEFGHS"],
    ]
    colors = ["blue", "orange", "green", "red"]

    avg_rate_exp = []
    damage_percentages = np.arange(0, 101, 5)
    for experiment in experiments:
        plt.figure(figsize=(15, 5))
        plt.xlabel("Percentage of Damaged Weights")
        plt.ylabel("Success Rate")
        plt.xticks(np.linspace(0, 100, 11))
        plt.ylim(-0.05, 1.1)
        plt.tight_layout()
        plt.grid(alpha=0.25)
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["left"].set_visible(False)
        plt.gca().spines["bottom"].set_visible(False)

        i = 0
        success_rate = []

        for key, color in zip(experiment, colors):
            patterns = [np.array(letters[letter]).flatten() for letter in key]
            patterns = np.array(patterns)

            hops = HopfieldNetwork(patterns)
            hops.hebb_rule()

            success_rate.append(hops.run_experiments(damage_percentages))
            plt.plot(damage_percentages, success_rate[-1], label=key, color=color)

        avg_rate_exp.append(np.mean(success_rate, axis=0))

        plt.legend()
        plt.show()

    plt.figure(figsize=(15, 5))
    plt.xlabel("Percentage of Damaged Weights")
    plt.ylabel("Avg Success Rate")
    plt.xticks(np.linspace(0, 100, 11))
    plt.ylim(-0.05, 1.1)
    plt.tight_layout()
    plt.grid(alpha=0.25)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["left"].set_visible(False)
    plt.gca().spines["bottom"].set_visible(False)
    plt.plot(damage_percentages, avg_rate_exp[0], label="Non Corr", color="g")
    plt.plot(damage_percentages, avg_rate_exp[1], label="Corr", color="r")
    plt.legend()
    plt.show()
