import numpy as np

RANDOM = np.random.default_rng(seed=0xC330)

def learnkNN(x: np.ndarray, k: int=20, tol=1.e-10) -> np.ndarray:
    
    d = x.shape[-1]

    # c = RANDOM.normal(0, 1, size=(k, d))
    c = RANDOM.choice(x, size=k, replace=False) # generally converges quicker and yields fewer trivial centroids
    cprev = np.zeros_like(c)
    cnew = np.zeros((k, d))
    count = np.zeros((k, ))
    
    iters = 0

    while(np.sum(np.square(c - cprev)) > tol):

        for xi in x:
            j = np.argmin(np.sum(np.square(xi - c), axis=1), axis=0)
            cnew[j] += xi
            count[j] += 1

        cprev = c.copy()
        
        count[count == 0] = 1
        c = (cnew.T / count).T

        cnew = np.zeros((k, d))
        count = np.zeros((k, ))
        iters += 1
        print(np.sum(np.square(c - cprev)), iters)

    return c

def predictLabel(c: np.ndarray, x: np.ndarray, topk: int=1):
    if (len(x.shape) == 1):
        x = x[None]
    
    n = x.shape[0]
    topk = min(x.shape[1], topk)

    labels = np.zeros((n, topk), dtype=np.int8)
    for i in range(n):
        xi = x[i]
        dists = np.sum(np.square(xi - c), axis=1)
        labels[i] = np.argsort(dists)[:topk]

    return labels

def _dataGen(n: int=1000, d: int=3, nmix: int=5, mixmeans: np.ndarray=None, mixsigmas: np.ndarray=None):

    if mixmeans is None or mixsigmas is None:
        mixmeans = np.zeros((nmix,d))
        mixsigmas = np.zeros((nmix,d,d))

        for j in range(nmix):
            mixmeans[j] = RANDOM.normal(0, 5, size=d)
            
            # PSD cov mtx - choose pos eigvals
            evals = np.abs(RANDOM.normal(0, 1, size=d))
            cov = np.diag(evals)
            mixsigmas[j] = cov

    else:
        nmix = mixmeans.shape[0]
        d = mixmeans.shape[-1]

    data = np.zeros((n, d))
    picks = np.zeros((n,), dtype=np.int8)

    for i in range(n):
        pick = RANDOM.integers(0, nmix)
        picks[i] = pick
        data[i] = RANDOM.multivariate_normal(mixmeans[pick], mixsigmas[pick], size=1)

    return data, mixmeans, mixsigmas, picks

def test1(c, d: int, nmix: int, topk: int):
    print("------------------------------TEST 1------------------------------")
    xtest, _, _, _ = _dataGen(n=1, d=d, nmix=nmix)
    labels = predictLabel(c, xtest, topk=topk)
    print("x",xtest)
    print("nearest centroids (sorted)", c[labels])
    print("labels of neartest centroids", labels)

def test2(c, mixmeans: np.ndarray, mixsigmas: np.ndarray, n: int=1000):
    print("------------------------------TEST 2------------------------------")
    means_closest = predictLabel(c, mixmeans, topk=1)
    print("closest centroid label to each mean:\n", means_closest.flatten())
    
    xtest, _, _, picks = _dataGen(n=n, mixmeans=mixmeans, mixsigmas=mixsigmas)
    labels = predictLabel(c, xtest, topk=1)

    match = np.sum(means_closest[picks] == labels)
    print(match / n)


if __name__ == "__main__":

    n = 10000
    d = 10
    k = 100

    xtrain, mixmeans, mixsigmas, _ = _dataGen(n, d, nmix=k)
    print("mixmeans:\n", mixmeans)

    c = learnkNN(xtrain, k=k)
    print("centroids:\n", c)

    test1(c, d=d, nmix=k, topk=20)    
    test2(c, mixmeans=mixmeans, mixsigmas=mixsigmas, n=n)
    