import numpy as np

def histogram_entropy(X, l, r, nbins=10):
    hist, bin_edges = np.histogram(X, nbins, (l, r))  # What is the appropriate bin choice? Constant-width? Shared data FD?
    w = (r - l) / nbins
    total_points = 1.0 * sum(hist)
    p = hist / total_points
    p = np.array([prob for prob in p if prob > 0])
    ent = -sum(p * np.log(p / w))
    return ent

def multiple_histogram_entropy(X_samples, nbins=10):
    l = min([min(x) for x in X_samples])
    r = max([max(x) for x in X_samples])
    return [histogram_entropy(x, l, r, nbins=nbins) for x in X_samples]

X = np.random.uniform(0, 100, 1000)
print(histogram_entropy(X, 0, 100, nbins=100))
X2 = np.random.uniform(300, 400, 1000)
print(multiple_histogram_entropy([X, X2], 10))
print(np.log(100))