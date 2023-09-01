import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from src.models.conditional_mean_operator import ConditionalMeanOperator


rng = np.random.default_rng()
rho = 0.5
dim = 2
n_samples = 1000


sample_joint = rng.multivariate_normal(
    mean=np.zeros(dim, dtype=np.float64),
    cov=np.array([[1, rho], [rho, 1]], dtype=np.float64),
    size=n_samples
)
x_sample = sample_joint[:, 0].reshape(-1, 1)
z_sample = sample_joint[:, 1].reshape(-1, 1)


model = ConditionalMeanOperator()
weight, loss = model.find_best_regularization_weight(z_sample, x_sample)
model.fit(z_sample, x_sample)
print("Best weight: ", weight)
print("Best loss: ", loss)

def mean_x_given_z(z):
    return rho * z

def var_x_given_z(z):
    return 1 - rho**2

def mean_x_squared_given_z(z):
    return mean_x_given_z(z)**2 + var_x_given_z(z)

if __name__ == "__main__":
    Z = np.linspace(-1.96, 1.96, 100)

    true_mean = mean_x_given_z(Z)
    estimated_mean = model.predict(x_sample.ravel(), Z.reshape(-1, 1))

    fig_mean, ax_mean = plt.subplots()
    ax_mean.plot(Z, true_mean, color="r", linewidth=1.5, label="True")
    ax_mean.plot(Z, estimated_mean, color="b", linestyle="--", linewidth=1.5, label="Estimated")
    ax_mean.scatter(sample_joint[:, 1], sample_joint[:, 0], color="k", alpha=.2,
                    s=2, label="data")
    ax_mean.set_xlabel(r"$z$")
    ax_mean.set_ylabel(r"$x$")
    ax_mean.set_title(r"$E[X | Z = z]$")
    ax_mean.legend()

    true_square = mean_x_squared_given_z(Z)
    estimated_square = model.predict(x_sample.ravel()**2, Z.reshape(-1, 1))

    fig_square, ax_square = plt.subplots()
    ax_square.plot(Z, true_square, color="r", linewidth=1.5, label="True")
    ax_square.plot(Z, estimated_square, color="b", linestyle="--",
                   linewidth=1.5, label="Estimated")
    ax_square.scatter(sample_joint[:, 1], sample_joint[:, 0]**2, color="k", alpha=.2,
                      s=2, label="data")
    ax_square.set_xlabel(r"$z$")
    ax_square.set_ylabel(r"$x$")
    ax_square.set_title(r"$E[X^2 | Z = z]$")
    ax_square.legend()

    fig_mean.savefig("mean.pdf")
    fig_square.savefig("square.pdf")
