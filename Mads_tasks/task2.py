import numpy as np 
from Mads_tasks.task1 import plot_hist
from scipy.stats import chi2
import matplotlib.pyplot as plt

def breast_cancer_state_dists(transition_matrix, sample_size, months):
    states_at_t = []
    for _ in range(sample_size):
        state = 0
        for i in range(1, months + 1):
            if state == 4:
                break
            state = np.random.choice(5, p=transition_matrix[state])
        states_at_t.append(state)
    states_at_t = np.array(states_at_t)
    proportions = np.bincount(states_at_t, minlength=5) / sample_size
    
    return proportions

def chi_square_test(observed, expected):
    chi2_stat = np.sum((observed - expected) ** 2 / expected)
    df = 4
    p_value = 1 - chi2.cdf(chi2_stat, df)
    return p_value

def analytical_distribution(months, p0, transition_matrix):
    return p0 @ np.linalg.matrix_power(transition_matrix, months)

def plot_analytical_vs_simulated(analytical_dist, simulated_dist, months=60):
    states = np.arange(len(analytical_dist))
    plt.bar(states - 0.2, analytical_dist, width=0.4, label='Analytical', alpha=0.7)
    plt.bar(states + 0.2, simulated_dist, width=0.4, label='Simulated', alpha=0.7)
    plt.xlabel('States')
    plt.ylabel('Proportion')
    plt.title(f'Analytical vs Simulated State Distribution (Months: {months})')
    plt.xticks(states)
    plt.legend()
    plt.savefig("figures/task2_comparison.png")
    plt.show()
    


if __name__ == "__main__":
    transition_matrix = np.array([[0.9915, 0.005, 0.0025, 0, 0.001],
                                  [0, 0.986, 0.005, 0.004, 0.005],
                                  [0, 0, 0.992, 0.003, 0.005],
                                  [0, 0, 0, 0.991, 0.009],
                                  [0, 0, 0, 0, 1]])
    sample_size = 1000
    proportions = breast_cancer_state_dists(transition_matrix, sample_size, 120)
    analytical_dist = analytical_distribution(120, np.array([1, 0, 0, 0, 0]), transition_matrix)
    p_value = chi_square_test(proportions, analytical_dist)
    print(f"p_value: {p_value}")

    plot_analytical_vs_simulated(analytical_dist, proportions, months=120)
