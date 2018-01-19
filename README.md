## TFDeepSurv
COX Proportional risk model implemented by tensorflow

### Some different to DeepSurv
DeepSurv is a package for Deep COX Proportional risk model, published on Github, our works may differ in:

- Early stabilizing feature importance.
- Consider ties of death time, which means different loss function.
- Estimator for survival function, three estimating algorithm provided.
- Scientific Hyperparameters method, Bayesian Hyperparameters Optimize for neural network(in plan).

### TODO-list
- try Bayesian Hyperparameters Optimization for DNN.(**Must fixed structure of network at first !**)
- Test simulated data and reality data.
- Analyze results of experiments.

### Reference
- LDeepSurv.py : Deep Neural Network for Data without ties of dead time.
- L2DeepSurv.py : Deep Neural Network for Data without/with ties of dead time.
- HyperParametersTuning.py : Tuning Hyperparameters of neural network.
- dataset.py : Generate Simulated data.
- utils.py : general function.
- vision.py : visualize data function.
- notebook/ : note results of experiments.