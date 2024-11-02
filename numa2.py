import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

np.random.seed(0)

X = [[1,2,3,4],
     [2.0,5.0,-1.0,2.0],
     [-1.5,2.7,3.3,-0.8]]



class LayerDense():
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases  =np.zeros((1,n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        
class ActivationReLu:
    def forward(self, inputs):
        self.output = np.maximum(0,inputs)

class SoftmaxActivation:
    def forward (self, inputs):
        exp_values = np.exp(inputs) - np.max(inputs, axis=1 , keepdims=True)
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

class Loss:
    def calculate(self, output, y):
        sample_loses = self.forward(output, y)
        data_loss = np.mean(sample_loses)
        return data_loss

class Loss_CategoricalCrossentrophy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        else:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)
            
        nagative_log_likelihoods = -np.log(correct_confidences)
        return nagative_log_likelihoods
         
        
        
X,y = spiral_data(100,3)

dense1 = LayerDense(2,3)
activation1 = ActivationReLu()

dense2 = LayerDense(3,3)
activation2 = SoftmaxActivation()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[-5:])

loss_function = Loss_CategoricalCrossentrophy()
loss = loss_function.calculate(activation2.output, y)

print("loss: ", loss)
# layer1 = LayerDense(2,5)
# activation1 = ActivationReLu()

# layer1.forward(X)


# activation1.forward(layer1.output)
# print(activation1.output)
