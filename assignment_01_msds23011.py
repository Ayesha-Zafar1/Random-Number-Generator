# **Assignment 1: Random Number Generation**

import numpy as np
from matplotlib import pyplot as plt
from typing import List

class RNG:

    _seed = 123456789  # Initial seed for the generator

    @staticmethod
    def _lcg(a: int = 1664525, c: int = 1013904223, m: int = 2**32) -> float:
        """Linear Congruential Generator (LCG) for uniform [0,1) distribution."""
        RNG._seed = (a * RNG._seed + c) % m
        return RNG._seed / m

    @staticmethod
    def uniform(count: int) -> np.ndarray:
        """Generate random numbers from a uniform distribution using LCG."""
        return np.array([RNG._lcg() for _ in range(count)])

    @staticmethod
    def gaussian(count: int,
                 mean: float = 0,
                 std_dev: float = 1) -> np.ndarray:
        """Generate random numbers using the Box-Muller transform for Gaussian distribution."""

        if count <= 0:
            raise ValueError("Count must be a positive integer.")

        if std_dev < 0:
            raise ValueError("Standard deviation must be non-negative.")

        # Half the count since each Box-Muller transform generates two numbers
        half_count = (count + 1) // 2

        u1 = RNG.uniform(half_count)
        u2 = RNG.uniform(half_count)

        # Computing both z0 and z1
        z0 = np.sqrt(-2.0 * np.log(u1)) * np.cos(2 * np.pi * u2)
        z1 = np.sqrt(-2.0 * np.log(u1)) * np.sin(2 * np.pi * u2)

        # Combine both z0 and z1 and then adjust by mean and standard deviation
        z = np.concatenate([z0, z1])[:count]  # Trim to the exact requested count
        return mean + z * std_dev

    @staticmethod
    def mv_gaussian(mean_vec: List[float],
                    cov_matrix: List[List[float]],
                    count: int) -> np.ndarray:
        """Generate random numbers from a multivariate Gaussian distribution."""

        mean_vec = np.array(mean_vec)
        cov_matrix = np.array(cov_matrix)

        if count <= 0:
            raise ValueError("Count must be a positive integer.")

        if len(mean_vec) != len(cov_matrix):
            raise ValueError("Mean vector and covariance matrix dimensions must match.")

        # Cholesky decomposition to convert covariance matrix
        L = np.linalg.cholesky(cov_matrix)

        # Generating standard normal random variables
        z = np.array([RNG.gaussian(len(mean_vec)) for _ in range(count)])

        # Applying transformation to get correlated Gaussian variables
        return mean_vec + np.dot(z, L.T)

### 2. Generating and Visualizing Random Numbers

import seaborn as sns

# Visualizing the Generated Random Numbers

# 1. Uniform Distribution Visualization
def plot_uniform_distribution():
    uniform_samples = RNG.uniform(1000)
    plt.figure(figsize=(8, 5))
    sns.histplot(uniform_samples, bins=30, color='blue')
    plt.title("Uniform Distribution")
    plt.xlabel("Random Numbers (0 to 1)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

# 2. 1D Gaussian Distribution Visualization
def plot_1d_gaussian_distribution(mean=0, std_dev=1):
    gaussian_samples = RNG.gaussian(1000, mean, std_dev)
    plt.figure(figsize=(8, 5))
    sns.histplot(gaussian_samples, bins=30, kde=True, color='green')
    plt.title(f"1D Gaussian Distribution (mean={mean}, std_dev={std_dev})")
    plt.xlabel("Random Numbers")
    plt.ylabel("Density")
    plt.grid(True)

    # Adding the annotations for mean
    plt.axvline(mean, color='r', linestyle='--', label='Mean')
    plt.legend()

    plt.show()

# 3. 2D Gaussian Distribution Visualization
def plot_2d_gaussian_distribution(mean_vec=[0, 0], cov_matrix=[[1, 0.5], [0.5, 1]]):
    mv_gaussian_samples = RNG.mv_gaussian(mean_vec, cov_matrix, 1000)
    plt.figure(figsize=(8, 5))
    plt.scatter(mv_gaussian_samples[:, 0], mv_gaussian_samples[:, 1], color='red', alpha=0.6)
    plt.title("2D Gaussian Distribution")
    plt.xlabel("X-axis (1st dimension)")
    plt.ylabel("Y-axis (2nd dimension)")
    plt.grid(True)

    # Annotations for mean vector
    plt.axhline(y=mean_vec[1], color='blue', linestyle='--', label='Mean Y')
    plt.axvline(x=mean_vec[0], color='green', linestyle='--', label='Mean X')
    plt.legend()

    plt.show()

# Visualizing the distributions
plot_uniform_distribution()         # Uniform Distribution
plot_1d_gaussian_distribution()     # 1D Gaussian Distribution
plot_2d_gaussian_distribution()     # 2D Gaussian Distribution

