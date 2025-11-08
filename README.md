# GMM-From-Scratch-on-Wine-Dataset
"ML models implemented from scratch using NumPy and Pandas only"

🍷 Gaussian Mixture Model (GMM) from Scratch on Wine Dataset

📘 Project Overview

This project demonstrates the implementation of a Gaussian Mixture Model (GMM) from scratch using the Wine dataset.
It shows how probabilistic clustering works using Expectation-Maximization (EM) and how to evaluate and visualize clusters with AIC/BIC, Silhouette Score, and PCA projection.


---

🎯 Objective

To identify natural clusters of wine samples based on chemical composition using a probabilistic unsupervised learning approach — without using any predefined labels.


---

🧠 Theoretical Background

1. Gaussian Mixture Model

A GMM models the data as a weighted sum of multiple Gaussian distributions:

p(x) = $$\sum_{k=1}^{K} \pi_k \, \mathcal{N}(x \mid \mu_k, \Sigma_k)$$

where:

K = number of clusters

 $$\pi_k$$= mixing coefficient for cluster , with $$\pi_k$$=1.

$$\mu_k$$ = mean vector of cluster 

$$\Sigma_k$$ = covariance matrix of cluster 



---

2. Expectation-Maximization (EM) Algorithm

Used to iteratively estimate parameters ($$\pi$$, $$\mu$$, $$\Sigma$$):

E-Step:

Compute the responsibility , i.e., the probability that sample  belongs to cluster :

$$\gamma_{ik}$$ = $$\frac{\pi_k \, \mathcal{N}(x_i \mid \mu_k, \Sigma_k)}{\sum_{j=1}^{K} \pi_j \, \mathcal{N}(x_i \mid \mu_j, \Sigma_j)}$$

M-Step:

Update parameters using the new responsibilities:

$$\N_k$$ = $$\sum_{i=1}^{N} \gamma_{ik}$$

$$\mu_k$$ = $$\frac{1}{N_k} \sum_{i=1}^{N} \gamma_{ik} x_i $$

$$\Sigma_k$$ = $$\frac{1}{N_k} \sum_{i=1}^{N} \gamma_{ik} (x_i - \mu_k)(x_i - \mu_k)^T$$

$$\pi_k$$ = $$\frac{N_k}{N} $$

Repeat until log-likelihood convergence:

$$\mathcal{L}$$ = $$\sum_{i=1}^{N} \log \left( \sum_{k=1}^{K} \pi_k \mathcal{N}(x_i \mid \mu_k, \Sigma_k) \right)$$


---

🧩 Dataset Description

Wine Dataset (UCI Repository / sklearn):

13 chemical analysis features (e.g., alcohol, malic acid, flavanoids, magnesium, etc.)

178 wine samples

3 underlying wine types (used only for validation, not for training)



---

⚙️ Implementation Steps

1. Load & Scale Data

Standardize features using StandardScaler.



2. Initialize GMM Parameters

Random means, covariances, and uniform mixing coefficients.



3. Run EM Algorithm

Alternate between E-Step and M-Step until convergence.



4. Model Selection

Compare GMM models for  to  clusters using:

BIC (Bayesian Information Criterion)

AIC (Akaike Information Criterion)

Silhouette Score




5. Dimensionality Reduction

Apply PCA (Principal Component Analysis) to visualize clusters in 2D space.



6. Cluster Visualization

Plot clusters with ellipses showing Gaussian boundaries.



7. Model Evaluation

Compute Silhouette Score and optionally Adjusted Rand Index (ARI) if true labels available.





---

📊 Evaluation Metrics

Metric	                                                  Description

Silhouette Score:	                                       Measures cluster separation and cohesion. Higher = better.
BIC/AIC:	                                               Penalize model complexity. Lower = better model fit.
ARI (Optional):	                                         Measures how well predicted clusters match true classes.



---

🎨 PCA Projection

PCA was applied to reduce the 13D feature space to 2D for visualization.
Each color in the plot represents one cluster learned by the GMM.

This helps to visually assess how well GMM separated the samples based on their underlying distributions.


---

💡 Insights

GMM successfully identified 3 major clusters corresponding to distinct wine categories.

PCA visualization showed clear cluster separation in the reduced space.

Optimal number of clusters determined via BIC and Silhouette Score ≈ 3, aligning with real labels.

GMM outperformed K-Means in terms of cluster flexibility because it accounts for variance and shape of each cluster.



---

🧭 Real-World Relevance

Domain	       :                                    Application

Finance	    :                                (   Customer segmentation by spending patterns)
Healthcare	     :                             ( Patient risk clustering)
Speech Recognition	  :                         (Modeling sound patterns probabilistically)
Marketing	          :                           (Identifying customer behavior clusters)
Anomaly Detection	   :                          (Points with low probability under all Gaussians are flagged as anomalies)



---

🧠 Future Work

Implement Diagonal / Spherical covariance constraints

Extend to Variational GMM (Bayesian version)

Compare performance with K-Means, DBSCAN, and Hierarchical Clustering



---

🧮 Mathematical Appendix

Formulas derived from the standard GMM-EM approach, applied to multivariate Gaussian densities:

$$\mathcal{N}(x \mid \mu, \Sigma)$$ = $$\frac{1}{(2\pi)^{d/2} |\Sigma|^{1/2}} 
\exp\left(-\frac{1}{2}(x-\mu)^T \Sigma^{-1}(x-\mu)\right)$$


---

🧩 Note on Learning Journey

" These derivations are included for conceptual clarity and personal learning purposes.
The project emphasizes both mathematical understanding and practical implementation as part of my AI/ML internship preparation journey."
