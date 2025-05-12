#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import fractional_matrix_power, logm, expm
import numpy.linalg as LA

# ---------- Utility Functions ----------

def sample_policy(mean, cov_diag, n_samples):
    """Draw samples from N(mean, Σ) with Σ diagonal given by cov_diag."""
    std = np.sqrt(cov_diag)
    return mean + np.random.randn(n_samples, 2) * std

def reward_function(a, a0=np.array([0.0, 0.0]), sigma_R=1.0):
    diff = a - a0
    dist_sq = np.sum(diff**2, axis=1)
    return np.exp(-dist_sq/(2 * sigma_R**2))

def entropy_gaussian(cov_diag):
    """Entropy for a 2D Gaussian with diagonal covariance."""
    d = 2
    return 0.5 * np.log((2 * np.pi * np.e * cov_diag[0]) * (2 * np.pi * np.e * cov_diag[1]))

def kl_gaussian(P_mu, P_Sigma, Q_mu, Q_Sigma):
    d = P_mu.size
    inv_QSigma = LA.inv(Q_Sigma)
    diff = Q_mu - P_mu
    term1 = diff.T @ inv_QSigma @ diff
    term2 = np.trace(inv_QSigma @ P_Sigma)
    term3 = np.log(LA.det(Q_Sigma)/LA.det(P_Sigma))
    return 0.5 * (term1 + term2 + term3 - d)

def geodesic_mean(mu0, mu1, t):
    """Linear interpolation between means."""
    return (1-t)*mu0 + t*mu1

def geodesic_covariance(Sigma0, Sigma1, t):
    """Geodesic for covariance matrices using the affine-invariant metric."""
    # Compute matrix square root and its inverse
    Sigma0_sqrt = fractional_matrix_power(Sigma0, 0.5)
    Sigma0_inv_sqrt = fractional_matrix_power(Sigma0, -0.5)
    middle = Sigma0_inv_sqrt @ Sigma1 @ Sigma0_inv_sqrt
    middle_t = fractional_matrix_power(middle, t)
    return Sigma0_sqrt @ middle_t @ Sigma0_sqrt

# ---------- Simulation of Gradient Updates ----------

def optimize_policy(lambda_val, n_iter=300, n_samples=10000, lr_mean=0.1, lr_cov=0.05):
    # Initialize at attractor for simplicity:
    mean = np.array([0.0, 0.0])
    cov_diag = np.array([0.5, 0.5])
    trajectory = []  # record (mean, cov_diag)
    for t in range(n_iter):
        actions = sample_policy(mean, cov_diag, n_samples)
        R = reward_function(actions)
        diff = actions - mean
        inv_cov = 1.0 / cov_diag  # for diagonal cov, inverse is element-wise
        
        # Policy gradient for mean:
        grad_mean = (R[:, None] * diff * inv_cov).mean(axis=0)
        
        # Gradient for covariance (diagonal case)
        ER = R.mean()
        ER_diff2 = (R[:, None] * (diff**2)).mean(axis=0)
        grad_cov = 0.5 * ER_diff2 / (cov_diag**2) + 0.5 * (lambda_val - ER) / cov_diag
        
        # Update steps
        mean += lr_mean * grad_mean
        cov_diag += lr_cov * grad_cov
        cov_diag = np.maximum(cov_diag, 1e-4)
        
        trajectory.append((mean.copy(), cov_diag.copy()))
    return trajectory

# ---------- Compute Trajectory and Geodesic ----------
#%%

# Run optimization for a given lambda
lambda_val = 0.1
traj = optimize_policy(lambda_val)

# Extract initial and final parameters
mean0, cov0 = traj[0]
meanT, covT = traj[-1]
Sigma0 = np.diag(cov0)
SigmaT = np.diag(covT)

# Parameterize the geodesic: for t in [0,1]
t_values = np.linspace(0, 1, len(traj))
geo_means = [geodesic_mean(mean0, meanT, t) for t in t_values]
geo_covs  = [geodesic_covariance(Sigma0, SigmaT, t) for t in t_values]

# For plotting, extract the standard deviations (sqrt of diagonal elements)
geo_cov_diag = np.array([np.diag(Sigma) for Sigma in geo_covs])
traj_cov_diag = np.array([cov for (_, cov) in traj])
traj_means = np.array([mu for (mu, _) in traj])

# ---------- Visualization ----------

plt.figure(figsize=(10,4))

# Plot trajectory of means in 2D
plt.subplot(1,2,1)
plt.plot(traj_means[:,0], traj_means[:,1], 'o-', label='Gradient trajectory')
plt.plot([mean0[0], meanT[0]], [mean0[1], meanT[1]], 'k--', label='Straight line (geodesic for mean)')
plt.xlabel("μ₁"); plt.ylabel("μ₂")
plt.title("Trajectory of Mean")
plt.legend()

# Plot covariance evolution (standard deviations)
plt.subplot(1,2,2)
plt.plot(t_values, traj_cov_diag[:,0], 'o-', label='σ₁ (trajectory)')
plt.plot(t_values, geo_cov_diag[:,0], 'k--', label='σ₁ (geodesic)')
plt.plot(t_values, traj_cov_diag[:,1], 's-', label='σ₂ (trajectory)')
plt.plot(t_values, geo_cov_diag[:,1], 'r--', label='σ₂ (geodesic)')
plt.xlabel("t"); plt.ylabel("Standard deviation")
plt.title("Covariance evolution")
plt.legend()

plt.tight_layout()
plt.show()

# ---------- Compare Trajectory and Geodesic using KL Divergence ----------

# For each time step, compute the KL divergence between the policy and the geodesic point
kl_vals = []
for (mu_traj, cov_traj), mu_geo, Sigma_geo in zip(traj, geo_means, geo_covs):
    Sigma_traj = np.diag(cov_traj)
    kl_vals.append(kl_gaussian(mu_traj, Sigma_traj, mu_geo, Sigma_geo))

plt.figure()
plt.plot(t_values, kl_vals, 'o-')
plt.xlabel("t")
plt.ylabel("KL Divergence")
plt.title("KL divergence between gradient trajectory and geodesic")
plt.show()

# %%
