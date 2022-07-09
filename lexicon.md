# Lexicon

## Categories?

| Term | Usage here |
| ----------- | ----------- |
| affine |Parameter dependence factors from operators, e.g., $H(\bm{\theta})= \sum_n h_n({\theta}) \widehat{H}_n$. |
| emulator | fast &amp; accurate model of the exact system |
| greedy sampling | Serially find snapshot locations $\theta_i$ at largest expected error (using a fast approximation). |
| high fidelity | Highly accurate, usually for costly calculation [Full-Order Model (FOM)]. |
| hyperreduction methods | Approximations to non-linearity or non-affineness (e.g., EIM) |
| intrusive | Non-intrusive treats FOM as black box; intrusive requires coding. |
| multifidelity | |
| offline-online paradigm | Heavy compute done once (offline); cheap to vary parameters (online). |
| proper orthogonal decomposition (POD) | Generically the term POD is used for PCA-type reduction via SVD. In snapshot context, PCA is applied to reduce/orthogonalize snapshot basis. |
| reduced basis methods (RBMs) | Implement snapshot-based projection methods. |
| reduced-order model | General name for an emulator resulting from applying MOR techniques.|
| snapshots | High-fidelity calculations at a set of parameters and/or times.|
| strong form | Differential equations are said to state a problem in a strong form.|
| SVD | Singular value decomposition. |
| weak form | An integral expression such as a functional that implicitly contains a differential equation. |


Feedback would be most welcome.
