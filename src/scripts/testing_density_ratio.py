import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance_matrix
from sklearn.neighbors import KernelDensity
from matplotlib import cm
from matplotlib.ticker import LinearLocator

from src.models.density_ratio import DensityRatio


rng = np.random.default_rng(42)
rho = 0.5
dim = 2
n_samples = 2000


sample_joint = rng.multivariate_normal(
    mean=np.zeros(dim, dtype=np.float64),
    cov=np.array([[1, rho], [rho, 1]], dtype=np.float64),
    size=n_samples
)
sample_independent = np.concatenate(
    [sample_joint[:, 0].reshape(-1, 1), np.roll(sample_joint[:, 1].reshape(-1, 1), 1)],
    axis=1
)

model = DensityRatio(
    regularization="rkhs",
    regularization_weight=0.01
)
model.fit(sample_joint, sample_independent)

bandwidth = 0.2
KDE_joint = KernelDensity(bandwidth=bandwidth)
KDE_x = KernelDensity(bandwidth=bandwidth)
KDE_y = KernelDensity(bandwidth=bandwidth)
KDE_joint.fit(sample_joint)
KDE_x.fit(sample_independent[:, 0].reshape(-1, 1))
KDE_y.fit(sample_independent[:, 1].reshape(-1, 1))

# test_sample = rng.multivariate_normal(
#     mean=np.zeros(dim, dtype=np.float64),
#     cov=np.array([[1, rho], [rho, 1]], dtype=np.float64),
#     size=n_samples
# )
# x_domain = test_sample[:, 0]
# y_domain = test_sample[:, 1]
# x_domain = np.linspace(start=-1.96, stop=1.96, num=100)
x_domain = rng.standard_normal(500)
y_domain = rng.standard_normal(500)
x_coords = x_domain.reshape(-1, 1)
y_coords = y_domain.reshape(-1, 1)
# x_mesh, y_mesh = np.meshgrid(x_domain, y_domain)
# x_coords, y_coords = x_mesh.reshape(-1, 1), y_mesh.reshape(-1, 1)
xy_points = np.concatenate(
    [x_coords, y_coords],
    axis=1
)
# print(x_domain.shape)
# print(y_domain.shape)
# print(x_coords.shape)
# print(y_coords.shape)

exponent = (
    (- rho / (2 * (1 - rho**2)))
    * (rho * (x_coords**2 + y_coords**2) - 2 * x_coords * y_coords)
)
actual_ratio = (1/np.sqrt(1 - rho**2) * np.exp(exponent)).ravel()
estimated_ratio = model(xy_points)

# print(actual_ratio.shape)
# print(estimated_ratio.shape)

KDE_estimated_ratio = np.exp(
    KDE_joint.score_samples(xy_points)
    - KDE_x.score_samples(x_coords)
    - KDE_y.score_samples(y_coords)
)

uLSIF_error = np.square(actual_ratio - np.maximum(estimated_ratio, 0))
KDE_error = np.square(actual_ratio - KDE_estimated_ratio)
print("uLSIF error: ", uLSIF_error.mean())
print("KDE error: ", KDE_error.mean())


# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# # Plot the surface.
# surf = ax.plot_surface(x_mesh, y_mesh, uLSIF_error.reshape(x_mesh.shape), cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)
# surf = ax.plot_surface(x_mesh, y_mesh, KDE_error.reshape(x_mesh.shape), cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)
# 
# # Customize the z axis.
# ax.set_zlim(-1.01, 1.01)
# ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
# ax.zaxis.set_major_formatter('{x:.02f}')

# # Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=5)

# plt.show()
