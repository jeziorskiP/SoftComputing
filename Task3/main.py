#SISE2 PJ
from network import Network
from helper import *
import numpy
from sys import argv
from time import time
import matplotlib.pyplot as plt
from argument_parser import ArgumentParser


argument_parser = ArgumentParser()
acceptable_error = argument_parser.get_acceptable_error()
learning_rate =argument_parser.get_learning_rate()
momentum = argument_parser.get_momentum()
hidden_size = argument_parser.get_hidden_size()
bias_switch =argument_parser.get_bias()

print_parameters(acceptable_error, learning_rate, momentum, hidden_size, bias_switch)

# start
input_list = []
target_list = []
network = Network(4, hidden_size, 4, learning_rate, bias_switch, momentum)

input_list =    [ [1,0,0,0],
                  [0,1,0,0],
                  [0,0,1,0],
                  [0,0,0,1]
                ]

target_list =    [ [1,0,0,0],
                  [0,1,0,0],
                  [0,0,1,0],
                  [0,0,0,1]
                ]
temp1 = []
temp2 = []

f_outs = []
h_outs1 = []
h_outs2 = []
# train
epoch = 0
error = 10
error_ar = []
start = time() * 1000
while epoch < 150000 and error > acceptable_error:
    error_ar.clear()
    for j in range(len(input_list)):
        r = numpy.random.randint(0, len(input_list))
        network.train(input_list[r], target_list[r])
        error_ar.append(mean_square_error(network.query, input_list, target_list))
    error = numpy.sum(error_ar) / len(error_ar)
    if epoch % 100 == 0:
        print(str(epoch) + '\t\terror = ' + str(error))
        if hidden_size == 2:
            final_outputs, hidden_outputs = network.query2(input_list)
            #print(hidden_outputs)
            h_outs1.append(hidden_outputs[0])
            h_outs2.append(hidden_outputs[1])
        temp1.append(epoch)
        temp2.append(error)

    epoch += 1
stop = time() * 1000

print('\n\n\n' + str(epoch) + '\t\terror = ' + str(error) + '\ttime\t' + str(format(stop - start, '.3f')))
if hidden_size == 2:
    plotHidden(h_outs1, h_outs2)


plt.plot(temp1,temp2 )
plt.xlabel("epoch")
plt.ylabel("MSE")
plt.show()
# TESTOWANIE
input_list = []
target_list = []

input_list =    [ [1,0,0,0],
                  [0,1,0,0],
                  [0,0,1,0],
                  [0,0,0,1]
                ]

error = 10
error_ar = []
for j in range(len(input_list)):
    error_ar.append(mean_square_error(network.query, input_list, target_list))
error = numpy.sum(error_ar) / len(error_ar)

print('\n\nerror = ' + str(error)+"\n\n")

print("\n\nNetwork answers")
for i in range(len(input_list)):
    print("INPUT: "+str(input_list[i]) + " \nResult: \n"+str(network.query2(input_list[i])))
