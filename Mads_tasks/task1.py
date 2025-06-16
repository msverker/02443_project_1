import numpy as np
import matplotlib.pyplot as plt

def breast_cancer_model(transition_matrix, sample_size):
    months_survived = []
    for _ in range(sample_size):
        state = 0 
        months = 0
        path = [state]
        while state != 4:
            state = np.random.choice(5, p=transition_matrix[state])
            path.append(state)
            months = len(path) - 1
        months_survived.append(months)
    return np.array(months_survived)

def plot_hist(months):
    plt.hist(months, bins=30, alpha=0.7, color='blue', density=True)
    plt.title("Months Survived with Breast Cancer")
    plt.xlabel("Months")
    plt.ylabel("Density")
    plt.savefig("figures/task1_hist.png")

if __name__ == "__main__":
    transition_matrix = np.array([[0.9915, 0.005, 0.0025, 0, 0.001],
                                  [0, 0.986, 0.005, 0.004, 0.005],
                                  [0, 0, 0.992, 0.003, 0.005],
                                  [0, 0, 0, 0.991, 0.009],
                                  [0, 0, 0, 0, 1]])
    sample_size = 1000
    months_survived = breast_cancer_model(transition_matrix, sample_size)

    plot_hist(months_survived)