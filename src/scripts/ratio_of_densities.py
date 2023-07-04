import sys

import numpy as np
from scipy.stats import multivariate_normal, norm

rng = np.random.default_rng(42)

def monte_carlo(n=1000, var=1.0, cov=0.0):
    assert 0 <= cov <= var
    mean = np.zeros(2, dtype=np.float64)
    cov = np.array([[var, cov],
                    [cov, var]], dtype=np.float64)
    samples = rng.multivariate_normal(mean=mean, cov=cov, size=n)
    print("samples:")
    print(samples[:10])
    num = multivariate_normal(
        mean=mean, cov=cov, allow_singular=True
    ).pdf(samples)
    print("joint:")
    print(num[:10])
    den_x = norm(loc=0.0, scale=1.0).pdf(samples[:, 0])
    print("marginal x")
    print(den_x[:10])
    den_z = norm(loc=0.0, scale=1.0).pdf(samples[:, 1])
    print("marginal z")
    print(den_z[:10])
    print("product of marginals")
    print((den_x*den_z)[:10])
    ratio = num/(den_x*den_z)
    return np.mean(ratio)


if __name__=="__main__":
    print(monte_carlo(int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])))

