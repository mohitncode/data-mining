import sys
import numpy as np
import matplotlib.pyplot as plt

filename = sys.argv[1]
num_dimensions = 2
file = open(filename)

# Extract data and labels from dataset
data, labels = [], []
for l in file:
    cols = l.strip().split('\t')
    vals, label = list(map(float, cols[:-1])), cols[-1]
    data.append(vals)
    labels.append(label)
data = np.array(data)
cov_matrix = np.cov(data.T)
eig_vals, eig_vecs = np.linalg.eig(cov_matrix)
eig_pairs = [(np.abs(ev), eig_vecs[:, i]) for i, ev in enumerate(eig_vals)]

matrix_w = eig_vecs[:, :num_dimensions]
result = np.dot(data, matrix_w)

# Visualize the top two dimensions
diseases = list(set(labels))
colors = [plt.cm.jet(float(i) / len(diseases)) for i, u in enumerate(diseases)]
for i, u in enumerate(diseases):
    xi = [result[j][0] for j in range(len(result)) if labels[j] == u]
    yi = [result[j][1] for j in range(len(result)) if labels[j] == u]
    plt.scatter(xi, yi, c=colors[i], label=str(u))
plt.title("PCA scatter plot for " + filename)
plt.legend()
plt.show()
