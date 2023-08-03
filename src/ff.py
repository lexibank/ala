"""
Code for simple feed-forward networks.
"""

import numpy as np
from tqdm import tqdm


class FF(object):

    def __init__(
            self, 
            input_layer: int,
            hidden_layer: int,
            output_layer: int,
            verbose: bool=False,
            ):
        self.input_layer = np.random.uniform(-1, 1, (input_layer,
                                                     hidden_layer))
        self.output_layer = np.random.uniform(-1, 1, (hidden_layer,
                                                      output_layer))
        self.input_weights = []
        self.output_weights = []
        self.epoch_loss = []


    def train(self, training_data, epochs, learning_rate=0.01):
        for i in range(epochs):
            loss = 0
            for input_data, output_data in tqdm(
                    training_data, desc="epoch {0}".format(i+1)):
                # forward pass on the network
                predicted, hidden_layer, output_layer = self.forward(
                        self.input_layer,
                        self.output_layer,
                        input_data)

                # error calculation
                total_error = self.get_error(predicted, output_data)

                # backward weight adjustment
                self.backward(
                        total_error,
                        hidden_layer,
                        input_data,
                        learning_rate
                        )
                
                # loss calculation
                loss += get_loss(output_layer, output_data)
            self.epoch_loss.append(loss)
            self.input_weights.append(self.input_layer)
            self.output_weights.append(self.output_layer)
            if self.verbose:
                print("Epoch: {0}, Loss: {1:.2f}".format(i+1, loss))

    def get_error(self, predicted, output_data):
        
        idxs = set([i for i in range(len(output_data)) if output_data[i] == 1])
        idxs_l = len(idxs)
            
        total_error = [
                (p - 1) + (idxs_l - 1) * p if i in idxs else idxs_l * p for i, p in enumerate(predicted)
                ]           
        return  np.array(total_error)

    def get_loss(self, output_layer, output_data):
        if [x for x in output_layer if x > 700]:
            for i in range(len(uvecs)):
                if output_layer[i] > 700:
                    output_layer[i] = 700
    
        sum_1 = -1 * sum(
                [output_layer[i] for i, c in enumerate(output_data) if c == 1]) 
        sum_2 = sum(output_data) * np.log(np.sum(np.exp(output_layer)))
        return sum_1 + sum_2 


    def self.backward(
            self,
            total_error, 
            hidden_layer, 
            input_data,
            learning_rate
            ):
        dl_hidden_in = np.outer(input_data, np.dot(self.output_layer, total_error.T))
        dl_hidden_out = np.outer(hidden_layer, total_error)
    
        self.input_layer = self.input_layer - (learning_rate * dl_hidden_in)
        self.output_layer = self.output_layer - (learning_rate * dl_hidden_out)


    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def forward(self, iweights, oweights, ivecs):
        
        # from input vectors to input weights for first layer
        hidden = np.dot(iweights.T, ivecs)
        # from first layer to output layer
        out = np.dot(oweights.T, hidden)
        # prediction with softmax
        predicted = softmax(out)
    
        return predicted, hidden, out
    
    def predict(x, weights_in, weights_out):
        
        y, hidden, u = forward(weights_in, weights_out, x)
    
        return [i for i, v in enumerate(y) if v >= 0.99]
