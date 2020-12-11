import numpy

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