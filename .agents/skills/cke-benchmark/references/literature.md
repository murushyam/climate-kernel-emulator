# Core Literature & Mathematical Formulations for `climate-kernel-emulator`

## 1. Parametric Nonlocal Integration Kernels (Ferretti et al., 2026)
- **Citation**: Ferretti, S. L., Lin, J., Shamekh, S., Baldwin, J. W., Pritchard, M. S., & Beucler, T. (2026). *Data-Driven Integration Kernels for Interpretable Nonlocal Operator Learning*. **Environmental Data Science**, `doi:10.1017/eds.2026.10056` (`arXiv:2603.10305`).
- **Normalized Spherical Integration**:
  $$z_{c,k}(\mathbf{x}, t) = \frac{\sum_{\mathbf{x}'} \tilde{k}_{c,k}(\mathbf{x}, \mathbf{x}') \, u_c(\mathbf{x}', t) \cos\phi(\mathbf{x}')}{\sum_{\mathbf{x}'} \tilde{k}_{c,k}(\mathbf{x}, \mathbf{x}') \cos\phi(\mathbf{x}')}$$
- **Kernel Families**:
  - *Gaussian*: $k_{\text{G}}(\Delta\phi, \Delta\lambda) = \exp\!\left(-\frac{(\Delta\phi - \mu_\phi)^2}{2\sigma_\phi^2} - \frac{(\Delta\lambda_{\text{wrap}} - \mu_\lambda)^2}{2\sigma_\lambda^2}\right)$
  - *Mixture of Gaussians*: $k_{\text{MG}}(\Delta\mathbf{s}) = \sum_{m=1}^M w_m \exp(-\|\Delta\mathbf{s} - \boldsymbol{\mu}_m\|^2 / 2\sigma_m^2)$
  - *Differentiable Top-Hat*: $k_{\text{TH}}(r) = \sigma((r - a)/\varepsilon)\,\sigma((b - r)/\varepsilon)$
  - *Exponential*: $k_{\text{EXP}}(r) = \exp(-r / \tau)$

## 2. Physics-Grounded Fidelity Verification (Suri & Sonnewald, 2026)
- **Citation**: Suri, S., & Sonnewald, M. (2026). *Trusting Machine Learning With Physics: A Fidelity Verification Framework for Complex Systems*. **JGR: Machine Learning and Computation**, 3(6), e2025JH001211, `doi:10.1029/2025JH001211`.
- Combines **LRP-$\varepsilon$** internal pathway relevance and **SHAP** input perturbation attributions evaluated across 4 mechanistic climate regimes (`aerosol_hubs`, `arctic_amplification`, `tropical_itcz`, `southern_ocean`).

## 3. Zero-Shot Probabilistic Forecasting with Covariates (Petralia et al., 2026)
- **Citation**: Petralia, A., et al. / Mistral AI (2026). *Timestral: Time Series Foundation Models with Decoupled Horizon Forecasting*, `arXiv:2610.37585`.
- Passes known-future greenhouse gas and global aerosol trajectories via `known_covariates` of shape `(B, C_known, T_context + prediction_length)` to predict 21 quantiles ($0.01 \dots 0.99$) on CPU.

## 4. Corrective Residual Diffusion & Action-Conditioned World Models
- **IPSL-AID**: Kingston, K., et al. (2026). *IPSL-AID: Generative Diffusion Models for Climate Downscaling from Global to Regional Scales*, `arXiv:2604.03275`.
- **CorrDiff**: Mardani, M., et al. (2024). *Residual Corrective Diffusion Modeling for Km-Scale Atmospheric Downscaling*, **Commun. Earth Environ.**, 5, 124, `doi:10.1038/s43247-024-01291-0`.
- **Earth System World Model**: Wang, Z., et al. (2026). *Earth System World Model for What-If Simulations: A Case Study for Terrestrial Ecosystems*, `arXiv:2609.08855`.

