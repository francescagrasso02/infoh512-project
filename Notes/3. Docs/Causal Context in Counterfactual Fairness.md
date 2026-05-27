---
Authors:
  - Chen Wang
  - Sarah Erfani
  - Tansu Alpcan
  - Christopher Leckie
---
*PDF*
[[OIL-AD.pdf]]

```cardlink
url: https://arxiv.org/abs/2402.04567
title: "OIL-AD: An Anomaly Detection Framework for Sequential Decision Sequences"
description: "Anomaly detection in decision-making sequences is a challenging problem due to the complexity of normality representation learning and the sequential nature of the task. Most existing methods based on Reinforcement Learning (RL) are difficult to implement in the real world due to unrealistic assumptions, such as having access to environment dynamics, reward signals, and online interactions with the environment. To address these limitations, we propose an unsupervised method named Offline Imitation Learning based Anomaly Detection (OIL-AD), which detects anomalies in decision-making sequences using two extracted behaviour features: action optimality and sequential association. Our offline learning model is an adaptation of behavioural cloning with a transformer policy network, where we modify the training process to learn a Q function and a state value function from normal trajectories. We propose that the Q function and the state value function can provide sufficient information about agents' behavioural data, from which we derive two features for anomaly detection. The intuition behind our method is that the action optimality feature derived from the Q function can differentiate the optimal action from others at each local state, and the sequential association feature derived from the state value function has the potential to maintain the temporal correlations between decisions (state-action pairs). Our experiments show that OIL-AD can achieve outstanding online anomaly detection performance with up to 34.8% improvement in F1 score over comparable baselines."
host: arxiv.org
favicon: https://arxiv.org/static/browse/0.3.4/images/icons/favicon-32x32.png
image: https://arxiv.org/static/browse/0.3.4/images/arxiv-logo-fb.png
```

---

# Resume
