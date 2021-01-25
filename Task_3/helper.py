import numpy

import matplotlib.pyplot as plt
def print_parameters(acceptable_error, learning_rate,momentum,hidden_size, bias_switch):
    print("\nParameters:\n")
    print('acceptable error:\t' + str(acceptable_error) + '\n' +
          'learning rate:   \t' + str(learning_rate) + '\n' +
          'momentum:        \t' + str(momentum) + '\n' +
          'hidden size:     \t' + str(hidden_size) + '\n' +
          'bias switch:     \t' + str(bias_switch) + '\n')

def mean_square_error(func, xinput, xoutput):
    ans = 0
    for x in range(len(xoutput)):
        ans += ((func(xinput[x]).T - xoutput[x]) ** 2)
    return numpy.sum(ans) / len(xoutput)
    
def plot(X, Y, title, saveFile = ""):
    plt.plot(X,Y)
    plt.xlabel("epoch")
    plt.ylabel(title)
    plt.show()
    if saveFile == "":
        plt.savefig(saveFile)
        
def plotHidden(node1, node2):
    line1 = []
    line2 = []
    line3 = []
    line4 = []
    epoch = []
    k = 0
    print("----------")
    for i in node1:
        
        line1.append(i[0])
        line2.append(i[1])
        line3.append(i[2])
        line4.append(i[3])
        epoch.append(k)
        k+=1
    plt.plot(epoch, line1, label = "line 1")
    plt.plot(epoch, line2, label = "line 2")
    plt.plot(epoch, line3, label = "line 3")
    plt.plot(epoch, line4, label = "line 4")
    plt.show()
    
    line1.clear()
    line2.clear()
    line3.clear()
    line4.clear()

    print("----------")
    for i in node2:
        line1.append(i[0])
        line2.append(i[1])
        line3.append(i[2])
        line4.append(i[3])
    plt.plot(epoch, line1, label = "line 1")
    plt.plot(epoch, line2, label = "line 2")
    plt.plot(epoch, line3, label = "line 3")
    plt.plot(epoch, line4, label = "line 4")
    plt.show()
