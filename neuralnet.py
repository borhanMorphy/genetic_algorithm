import numpy as np
import functools
from activation import relu, softmax

class _Neuron():
    """
        shape of neuron given as feature_count = fc
        a = [w.T] x [X]
        w : shape of (fc,1) # fc = feature count
        X : shape of (fc,any)
        b : int
    """
    def __init__(self, **kwargs):
        self.weights = np.random.rand(kwargs["feature_count"], 1) # w
        self.inputs = None                                        # X
        self.bias = np.zeros((1,))                                # b
    
    def feed(self, X):
        self.inputs = X

class _Layer:
    """
        shape of Layer given as feature_count = fc, neuron_count = nc
        A = [W.T] x [X]
        W : shape of (fc,nc) # fc = feature count, nc = neuron count
        X : shape of (fc,any)
        b : shape of (nc,1)
    """
    def __init__(self, **kwargs):
        neurons = [_Neuron(feature_count=kwargs["feature_count"]) for i in range(kwargs["neuron_count"])]
        self.W = functools.reduce(lambda a,b : np.append(a, b, axis=1), map(lambda neuron:neuron.weights, neurons))
        self.b = functools.reduce(lambda a,b : np.append(a, b, axis=0), map(lambda neuron:neuron.bias, neurons))
        self.layer_type = kwargs["layer_type"]
    
    def forward_prop(self, X):
        if self.layer_type is "output":
            return softmax( np.dot(self.W.T, X) )
        else:
            return relu( np.dot(self.W.T, X) )

class NN:
    """
        shape of Neural Network as input_shape=ic, dense_shapes=[], output_shape=oc
    """
    def __init__(self, **kwargs):
        self.layers = [
                _Layer(
                    feature_count=kwargs["input_shape"][0], 
                    neuron_count=kwargs["input_shape"][1],
                    layer_type="input"
                )
            ]+[
                _Layer(
                    feature_count=shape[0], 
                    neuron_count=shape[1],
                    layer_type="dense"
                ) for shape in kwargs["dense_shapes"]
            ]+[
                _Layer(
                    feature_count=kwargs["output_shape"][0], 
                    neuron_count=kwargs["output_shape"][1],
                    layer_type="output"
                )
            ]
        print("------- Generating NN -------")
        for index in range(len(self.layers)):
            print("layer: ",index," shape: ",self.layers[index].W.shape," layer type: ",self.layers[index].layer_type)
        print("_____________________________\n")
        
    def prediction(self, A):
        for layer in self.layers:
            A = layer.forward_prop(A)
        return np.argmax(A)
    
    def deploy(self):
        return [layer.W for layer in self.layers]

"""
    takes: [[],[],[]]
    returns: np array as appended column by column
"""
def parse_input_stream(input_stream):
    return functools.reduce(lambda a,b:np.append(a,b,axis=1), map(lambda data:np.array(data).reshape(len(data),1),input_stream))

if __name__ == '__main__':
    ga = NN(input_shape=(4,2), dense_shapes=[(2,5),(5,5)], output_shape=(5,3))
    input_stream = [[10.0,10.0,6.5,3],[10.0,10.0,10.0,5.3]]
    input_stream = functools.reduce(lambda a,b:np.append(a,b,axis=1), map(lambda data:np.array(data).reshape(len(data),1),input_stream))
    choice = ga.prediction( input_stream )
    input_stream = [[10.0,10.0,6.5,3],[10.0,1.1,3.0,3.5]]
    input_stream = functools.reduce(lambda a,b:np.append(a,b,axis=1), map(lambda data:np.array(data).reshape(len(data),1),input_stream))
    choice2 = ga.prediction( input_stream )
    print(choice,"\n")
    print(choice2)