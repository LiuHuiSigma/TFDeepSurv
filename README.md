## TFDeepSurv
COX Proportional risk model implemented by Keras

## Some different to DeepSurv
DeepSurv is a package for Deep COX Proportional risk model, published on Github, our works much differ in:

- using Mini-Batch(should be as Hyper-parameters)
- The location of Batch Norm layers(X=w*x+b -> BatchNorm -> Activation, as demonstrated on paper)
- Optimizer(Adam Optimizer or others)
- Activation(relu or tanh or others)