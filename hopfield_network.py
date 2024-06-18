import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


class HopfieldNetwork:
    def __init__(self, patterns: np.ndarray) -> None:
        self.patterns = patterns
        self.n_neurons = patterns.shape[1]
        self.W = np.zeros((self.n_neurons, self.n_neurons))

    def hebb_rule(self):
        self.W = np.sum([np.outer(p, p) for p in self.patterns], axis=0)
        np.fill_diagonal(self.W, 0)
        return self.W

    def _update_neuron(self, state, index):
        return np.sign(self.W[index] @ state)

    def recall_pattern(self, input_pattern, max_iter=50, record=False):
        state = input_pattern.copy()
        j = 0

        while j < max_iter:
            if record:
                plt.figure(figsize=(2, 2))
                plt.imshow(state.reshape((5, 5)), cmap="gray")
                plt.axis("off")
                plt.show()

            old_state = state.copy()
            for i in range(self.n_neurons):
                state[i] = self._update_neuron(state, i)

            if np.array_equal(state, old_state):
                break

            j += 1

        return state, np.array_equal(state, input_pattern)

    def _damage_weights(self, damage_percent):
        n = self.W.shape[0]

        num_damage = int((n * (n - 1) / 2) * damage_percent / 100)

        indices_upper = np.triu_indices(n, k=1)

        upper_indices_to_damage = np.random.choice(
            len(indices_upper[0]), num_damage, replace=False
        )

        W_damaged = np.copy(self.W)

        i_indices = indices_upper[0][upper_indices_to_damage]
        j_indices = indices_upper[1][upper_indices_to_damage]
        W_damaged[i_indices, j_indices] = 0
        W_damaged[j_indices, i_indices] = 0

        return W_damaged

    def run_experiments(self, damage_percentages, num_trials=100):
        success_rates = []
        clean_W = self.W.copy()

        for damage_percent in tqdm(damage_percentages):
            total_success = 0

            for _ in range(num_trials):
                self.W = self._damage_weights(damage_percent)
                successful_recall_count = 0

                for pattern in self.patterns:
                    recalled_pattern, converged = self.recall_pattern(pattern)
                    if converged and np.array_equal(recalled_pattern, pattern):
                        successful_recall_count += 1
                self.W = clean_W

                success_rate = successful_recall_count / len(self.patterns)
                total_success += success_rate

            average_success_rate = total_success / num_trials
            success_rates.append(average_success_rate)

        return success_rates
