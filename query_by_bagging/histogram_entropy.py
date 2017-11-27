import numpy as np

X = np.random.uniform(0, 100, 1000)
hist, bin_edges = np.histogram(X, 10) # What is the appropriate bin choice? Constant-width? Shared data FD?
w = bin_edges[1] - bin_edges[0]
total_points = 1.0 * sum(hist)
p = hist / total_points
ent = -sum(p * np.log(p / w))
print(ent)
print(np.log(100))