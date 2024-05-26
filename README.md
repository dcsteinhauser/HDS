# Hybrid Differentiable Simulation: Improving Real-world Deployment through Data #

This repository was created for the course Foundations of Reinforcement Learning, held in the spring semester of 2024 at ETH Zurich. 

The purpose of this repository is three-fold:

1. It offers an open-source, parallelized implementation of <cite> Policy Optimization via Differentiable Simulation (PODS)</cite> [1] using <cite>JAX</cite> [2] and the <cite>BRAX</cite> [3] physics engine, since no such implementation is as of yet publicly available. 
2. To explore how well model-based reinforcement learning (particularly ones leveraging differentiable simulation) algorithms generalize to real world applications.
3. To explore whether using real-world data can mitigate the sim2real gap of said methods, as well as the known numerical stability issues that arise due to non-differentiable phenomena such as contact.

## Policy Optimization via Differentiable Simulation (PODS) ##





## Citations ##
[1] 
Mora, M.A.Z., Peychev, M., Ha, S., Vechev, M. &amp; Coros, S.. (2021). PODS: Policy Optimization via Differentiable Simulation. <i>Proceedings of the 38th International Conference on Machine Learning</i>, in <i>Proceedings of Machine Learning Research</i> 139:7805-7817 Available from https://proceedings.mlr.press/v139/mora21a.html.

[2] Bradbury, J., Frostig, R., Hawkins, P., Johnson, M. J., Leary, C., Maclaurin, D., Necula, G., Paszke, A., VanderPlas, J., Wanderman-Milne, S., & Zhang, Q. (2018). JAX: composable transformations of Python+NumPy programs (0.3.13) [Computer software]. http://github.com/google/jax

[3] Freeman, C. D., Frey, E., Raichuk, A., Girgin, S., Mordatch, I., & Bachem, O. (2021). Brax – A Differentiable Physics Engine for Large Scale Rigid Body Simulation.


