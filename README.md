# Bayesian & Stochastic LSTM for Cyclone Tracking

This repository implements two variants of LSTMs to predict cyclone trajectories (Latitude/Longitude) using the IBTrACS dataset.



## Results



| Model | Train RMSE | Test RMSE |
| :--- | :--- | :--- |
| Stochastic LSTM | 0.1150° | 0.2246° |
| Bayesian LSTM | 0.6948° | 0.9129° |


![slstm](results/slstm.png) 
![blstm](results/blstm.png) 

## Model Comparisons

| Model | Core Mechanism | Training Objective |
| --- | --- | --- |
| **Stochastic LSTM** (`slstm.py`) | Uses Weight Uncertainty (BayesLinear) during inference via stochastic sampling. | Standard **MSE Loss** only. |
| **Bayes LSTM** (`blstm.py`) | Full Variational Inference with weight distributions. | **ELBO**: MSE Loss + Weighted **KL Divergence**. |

---

### Stochastic LSTM
Treats the LSTM weights as distributions but optimizes primarily for point-estimate accuracy. It utilizes the "stochastic" nature by keeping the model in `.train()` mode during inference to sample from the weight distributions 50 times and average the results.

### Bayes LSTM

In addition to stochastic sampling, it includes a **KL Divergence** term in the loss function. This regularizes the learned weights to remain close to a prior distribution, improving uncertainty quantification and preventing overfitting. The performance of the BayesLSTM can be tweaked by changing the `prior_var` and `kl_weight` values.


### Conclusion

Although the Stochastic LSTM shows lower RMSE, it suffers from significant overfitting with a **95% error jump (0.11° to 0.22°)** between training and testing. The Bayesian LSTM achieves stable generalization and a tight error gap of only **31% (0.69° to 0.91°)** due to its variational inference framework, which incorporates KL divergence as a regularization term to penalize the complexity of the weight distributions.

