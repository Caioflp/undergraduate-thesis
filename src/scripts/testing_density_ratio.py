import numpy as np
from scipy.spatial import distance_matrix

from src.models.density_ratio import DensityRatio

rng = np.random.default_rng(42)
rho = 0.5
dim = 2
n_samples = 5


sample_joint = rng.multivariate_normal(
    mean=np.zeros(dim, dtype=np.float64),
    cov=np.array([[1, rho], [rho, 1]], dtype=np.float64),
    size=n_samples
)
sample_independent = np.concatenate(
    [sample_joint[:, 0].reshape(-1, 1), np.roll(sample_joint[0:, 1].reshape(-1, 1), 1)],
    axis=1
)

# print(sample_joint)
# print(sample_independent)

lengthscale = np.median(np.ravel(distance_matrix(sample_joint, sample_joint)))

model = DensityRatio(lengthscale=lengthscale, regularization_weight=0.5)
model.fit(sample_joint, sample_independent)
