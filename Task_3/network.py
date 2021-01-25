import numpy
import scipy.special


class Network:
    def __init__(self, input_nodes, hidden_nodes, output_nodes,
                 learning_rate=0.1, bias_switch=0, momentum=0,
                 ):
        self.i_nodes = input_nodes
        self.h_nodes = hidden_nodes
        self.o_nodes = output_nodes
        self.lr = learning_rate
        self.bias_switch = bias_switch
        self.w_ih = numpy.random.rand(self.h_nodes, self.i_nodes)
        self.b_ih = numpy.random.rand(self.h_nodes, 1) * bias_switch
        self.w_ho = numpy.random.rand(self.o_nodes, self.h_nodes)
        self.b_ho = numpy.random.rand(self.o_nodes, 1) * bias_switch
        self.momentum = momentum
        self.activation_func = lambda x: scipy.special.expit(x)
        self.derivative_activation_func = lambda x: x * (1.0 - x)
        self.w_ho_old = 0
        self.w_ih_old = 0
        self.b_ih_old = 0
        self.b_ho_old = 0

    def train(self, input_list, target_list):
        inputs = numpy.array(input_list, ndmin=2).T
        targets = numpy.array(target_list, ndmin=2).T

        # forward propagation
        h_inputs = numpy.dot(self.w_ih, inputs) + self.b_ih
        h_outputs = self.activation_func(h_inputs)
        
        #print("HIDDEN")
        #print(h_outputs)
        
        o_inputs = numpy.dot(self.w_ho, h_outputs) + self.b_ho

        o_outputs = self.activation_func(o_inputs)

        # back propagation
        o_errors = targets - o_outputs
        h_errors = numpy.dot(self.w_ho.T, o_errors)

        # output weights tweaking
        # o_outputs
        
        self.w_ho += self.lr * numpy.dot((o_errors * self.derivative_activation_func(o_outputs)), numpy.transpose(h_outputs)) + self.w_ho_old * self.momentum
        self.w_ho_old = self.lr * numpy.dot((o_errors * self.derivative_activation_func(o_outputs)), numpy.transpose(h_outputs))

        # output bias tweaking
        self.b_ho += self.lr * o_errors * self.derivative_activation_func(o_outputs) * self.bias_switch + self.b_ho_old * self.momentum
        self.b_ho_old = self.lr * o_errors * self.derivative_activation_func(o_outputs)
        self.b_ho *= self.bias_switch


        # hidden weights tweaking
        self.w_ih += self.lr * numpy.dot((h_errors * self.derivative_activation_func(h_outputs)), numpy.transpose(inputs)) + self.w_ih_old * self.momentum
        self.w_ih_old = self.lr * numpy.dot((h_errors *self.derivative_activation_func(h_outputs)), numpy.transpose(inputs))

        # hidden bias tweaking
        self.b_ih += self.lr * h_errors *self.derivative_activation_func(h_outputs) + self.b_ih_old * self.momentum
        self.b_ih_old = self.lr * h_errors * self.derivative_activation_func(h_outputs)
        self.b_ih *= self.bias_switch

    def query(self, input_list):
        inputs = numpy.array(input_list, ndmin=2).T
        hidden_inputs = numpy.dot(self.w_ih, inputs) + self.b_ih
        hidden_outputs = self.activation_func(hidden_inputs)
        #print(hidden_outputs)
        final_inputs = numpy.dot(self.w_ho, hidden_outputs) + self.b_ho
        final_outputs = self.activation_func(final_inputs)

        return final_outputs

    def query2(self, input_list):
        inputs = numpy.array(input_list, ndmin=2).T
        hidden_inputs = numpy.dot(self.w_ih, inputs) + self.b_ih
        hidden_outputs = self.activation_func(hidden_inputs)
        print("HIDDEN:")
        print(hidden_outputs)
        final_inputs = numpy.dot(self.w_ho, hidden_outputs) + self.b_ho
        final_outputs = self.activation_func(final_inputs)
        print("FINAL:")
        print(final_outputs)

        return final_outputs, hidden_outputs
