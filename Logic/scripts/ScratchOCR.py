import numpy as np
import os

class ScratchOCR:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.lr = learning_rate
        self.w_input_hidden = np.random.randn(hidden_nodes, input_nodes) * 0.01
        self.w_hidden_output = np.random.randn(output_nodes, hidden_nodes) * 0.01
        self.b_hidden = np.zeros((hidden_nodes, 1))
        self.b_output = np.zeros((output_nodes, 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        hidden_inputs = np.dot(self.w_input_hidden, inputs) + self.b_hidden
        hidden_outputs = self.sigmoid(hidden_inputs)
        final_inputs = np.dot(self.w_hidden_output, hidden_outputs) + self.b_output
        final_outputs = self.sigmoid(final_inputs)

        output_errors = targets - final_outputs
        output_grad = output_errors * self.sigmoid_derivative(final_outputs)
        self.w_hidden_output += self.lr * np.dot(output_grad, hidden_outputs.T)
        self.b_output += self.lr * output_grad

        hidden_errors = np.dot(self.w_hidden_output.T, output_grad)
        hidden_grad = hidden_errors * self.sigmoid_derivative(hidden_outputs)
        self.w_input_hidden += self.lr * np.dot(hidden_grad, inputs.T)
        self.b_hidden += self.lr * hidden_grad

    def predict(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T
        hidden = self.sigmoid(np.dot(self.w_input_hidden, inputs) + self.b_hidden)
        final = self.sigmoid(np.dot(self.w_hidden_output, hidden) + self.b_output)
        return final

    def save(self, folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        file_path = os.path.join(folder_path, "weights.npz")
        np.savez(file_path, 
                 w_ih=self.w_input_hidden, 
                 w_ho=self.w_hidden_output,
                 b_h=self.b_hidden,
                 b_o=self.b_output)
        print(f"Model saved to {file_path}")

    def load(self, folder_path):
        file_path = os.path.join(folder_path, "weights.npz")
        data = np.load(file_path)
        self.w_input_hidden = data['w_ih']
        self.w_hidden_output = data['w_ho']
        self.b_hidden = data['b_h']
        self.b_output = data['b_o']
        print(f"Model loaded from {file_path}")