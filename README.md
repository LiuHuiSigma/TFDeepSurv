## TFDeepSurv
COX Proportional risk model implemented by tensorflow

### Some different to DeepSurv
DeepSurv is a package for Deep COX Proportional risk model, published on Github, our works much differ in:

- The location of Batch Norm layers(X=w*x+b -> BatchNorm -> Activation, as demonstrated on paper)
- Optimizer(Adam Optimizer or others)
- Activation(relu or tanh or others)
- The ties of death time maybe occur

### TODO-list
- Estimate H0 when ties occurred.
- Take ties of dead time into consideration(toughly).