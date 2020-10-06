# Hidden Markov Random Field Dirichlet Process Gaussian Mixture Model

## Model descrition


\\[ \begin{eqnarray*}
\pi & \sim & \text{Dir}(K, \alpha) \text{ or GEM}(\eta) \\
                \theta_k & \sim & p (\theta_k)  \\
 \mathbf{z}_{1:N} | \pi& \sim & p (\mathbf{z}_{1:N} | \pi) \\
                 \mathbf{x}_n | \mathbf{z}_n =k, \theta & \sim & p(\mathbf{x}_n|\mathbf{z}_n =k, \theta)
\end{eqnarray*} \\]
