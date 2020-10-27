import pandas as pd
from sklearn.model_selection import train_test_split
import random
import numpy as np 

"""
pattern1 -> N=2 | pat=2 
pattern11 -> N=2 | pat=1 
pattern2 -> N=10 | pat=10 
pattern3 -> N=10 | pat=50 
pattern4 -> N=10 | pat=50 
pattern5 -> N=10 | pat=50 
pattern6 -> N=10 | pat=50 
"""

#parameters to set!
way = 1     #1 -> data from file. 2-> random data
N = 10      #inputs
pat = 10    #patterns
K = 10     #epoch
lr = 0.001  #learning rate
Wmin = 1.0  #lower weight range
Wmax = 2.0  #upper weight range
Pmin = 1.0  #lower input value range
Pmax = 2.0  #upper input value range
fileName = "patterns2"  #data source 

if(way == 1):
    data=pd.read_csv("./patterns/" + fileName+".txt", header=None)

elif(way == 2):
    data = pd.DataFrame(np.random.uniform(Pmin,Pmax, size=(pat,N+1)))   #N inputs and one extra column as output
else:
    print("error")

print(data)
#copy last column. classes = desired output
classes = data[data.columns[-1]]

#drop last column(desired output) from dataset
#data = only inputs
data = data.iloc[:,:-1]

#Concert data to numpy array
X = data.to_numpy()
Y = classes.to_numpy()

#Generate/initialize weights
#Random value from given range
weights = [random.uniform(Wmin, Wmax) for i in range(N)]

#Show first weights
print(weights)

#for each epoch, calculate
for k in range(K):
    print("\nEpoch: ", k)
    #for each pattern(pat), calculate output (input multiply by weights and sum everything), then calc Error as Delta and modyfy weights.
    for p in range(pat): 
        print("-------P: ", p, " ---------")
        #bierzemy pattern - jednen wiersz
        #lets get one pattern, one row
        row = X[p]  
        
        #Mnozymy kazdy element patterna przez wage
        #multiply each input (from that pattern/row) by weights
        out = np.multiply( weights, row)
        
        #suma z waga*X
        #Sum everthing (weights*X)
        #output = neuron's output
        output = np.sum(out)            
        
        #calculate error /delta
        #err = difference between expected value and received value(output)
        err = Y[p] - output
        
        #Change weights
        #new weight = last weight + lr*err*input
        weights = weights + lr*(Y[p] - output)*row
        
        print("weights ",weights)

print("\n-------------------ANS-------------------")
print("Received Weights \n",weights)
ans=pd.read_csv("./ans/" + fileName+"-ans.txt", header=None)
print("\nPatterns results(weights to reach)")
print(ans.to_numpy()[0])

diff = weights - ans.to_numpy()
print("\nDifferences between received values and patterns result" )
print(diff[0])
