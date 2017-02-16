# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 18:35:23 2015

@author: paresh
"""

"""
Feed-forward neural networks trained using backpropagation
based on code from http://rolisz.ro/2013/04/18/neural-networks-in-python/
"""
 
 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cross_validation import train_test_split 
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelBinarizer
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.grid_search import GridSearchCV
from sklearn import svm
from sklearn.neural_network import BernoulliRBM

def tanh(x):
    return np.tanh(x)
 
def tanh_deriv(x):
    return 1.0 - np.tanh(x)**2
 
def logistic(x):
    return 1/(1 + np.exp(-x))
 
def logistic_derivative(x):
    return logistic(x)*(1-logistic(x))
 
class NeuralNetwork:
    def __init__(self, layers, activation='tanh') :
        """
        layers: A list containing the number of units in each layer.
                Should contain at least two values
        activation: The activation function to be used. Can be
                "logistic" or "tanh"
        """
        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv
        self.num_layers = len(layers) - 1
        self.weights = [ np.random.randn(layers[i - 1] + 1, layers[i] + 1)/10 for i in range(1, len(layers) - 1) ]
        self.weights.append(np.random.randn(layers[-2] + 1, layers[-1])/10)
 
    def forward(self, x) :
        """
        compute the activation of each layer in the network
        """
        a = [x]
        for i in range(self.num_layers) :
            a.append(self.activation(np.dot(a[i], self.weights[i])))
        return a
 
    def backward(self, y, a) :
        """
        compute the deltas for example i
        """
        deltas = [(y - a[-1]) * self.activation_deriv(a[-1])]
        for l in range(len(a) - 2, 0, -1): # we need to begin at the second to last layer
            deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_deriv(a[l]))
        deltas.reverse()
        return deltas
 
    def fit(self, X, y, learning_rate=0.2, epochs=50):
        X = np.asarray(X)
        temp = np.ones( (X.shape[0], X.shape[1]+1))
        temp[:, 0:-1] = X  # adding the bias unit to the input layer
        X = temp
        y = np.asarray(y)
 
        for k in range(epochs):
            if k%10==0 : print "***************** ", k, "epochs  ***************"            
            I = np.random.permutation(X.shape[0])
            for i in I :
                a = self.forward(X[i])
                deltas = self.backward(y[i], a)
                # update the weights using the activations and deltas:
                for i in range(len(self.weights)):
                    layer = np.atleast_2d(a[i])
                    delta = np.atleast_2d(deltas[i])
                    self.weights[i] += learning_rate * layer.T.dot(delta)
 
    def predict(self, x):
        x = np.asarray(x)
        temp = np.ones(x.shape[0]+1)
        temp[0:-1] = x
        a = temp
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a
 
def test_digits() :
    digits = load_digits()
    X = digits.data
    y = digits.target
    X /= X.max()
    return X,y
     
if __name__=='__main__' :
    X,y = test_digits()    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    labels_train = LabelBinarizer().fit_transform(y_train)
    labels_test = LabelBinarizer().fit_transform(y_test)    
    """
    Plot network accuracy as a function of the number of hidden units 
    for a single-layer network
    """
    accuracy=[]
    arr_hidden_units=[]
    for j in range (1,13):
        k=2**j
        nn = NeuralNetwork([64,k,10],'logistic') 
        nn.fit(X_train,labels_train,epochs=100)
        predictions = []
        for i in range(X_test.shape[0]) :
            o = nn.predict(X_test[i])
            predictions.append(np.argmax(o))
        print confusion_matrix(y_test,predictions)
        arr_confusion = confusion_matrix(y_test,predictions)
        accuracy.append(round(((float(np.trace(arr_confusion)))/(float(np.sum(arr_confusion)))),3))
        print k,np.sum(arr_confusion),np.trace(arr_confusion)
        arr_hidden_units.append(k)
    print accuracy
    fig,ax = plt.subplots()
    ax.set_xscale('symlog', basex=2)
    ax.plot(arr_hidden_units,accuracy)
    plt.show()
    """
    Plot network accuracy as a function of the number of hidden units
    for a two-layer network    
    """
    arr_confusion2=[]
    accuracy2=[]
    arr_firstlayer=[]
    arr_secondlayer=[]
    for j in range (1,11):
        k=2**j
        for l in range(1,11):
            m=2**l
            nn_twolayer = NeuralNetwork([64,k,m,10],'logistic') 
            nn_twolayer.fit(X_train,labels_train,epochs=100)
            predictions = []
            for i in range(X_test.shape[0]) :
                o = nn_twolayer.predict(X_test[i])
                predictions.append(np.argmax(o))
            print confusion_matrix(y_test,predictions)
            arr_confusion2 = confusion_matrix(y_test,predictions)
            accuracy2.append(round(((float(np.trace(arr_confusion2)))/(float(np.sum(arr_confusion2)))),3))
            print k,np.sum(arr_confusion2),np.trace(arr_confusion2)
            arr_secondlayer.append(m)
            arr_firstlayer.append(k)
    #plot surface graph of accuracy vs elements hidden layer 1 vs elements hidden layer 2
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X = np.reshape(arr_firstlayer,(-1,10))
    X = np.log2(X)
    Y = np.reshape(arr_secondlayer,(-1,10))
    Y = np.log2(Y)   
    xticks = [2,4,8,16,32,64,128,256,512,1024]
    yticks = [2,4,8,16,32,64,128,256,512,1024]
    ax.set_xticks(np.log2(xticks))
    ax.set_xticklabels(xticks)
    ax.set_yticks(np.log2(yticks))
    ax.set_yticklabels(yticks)
    Z = np.reshape(accuracy2,(-1,10))
    ax.plot_surface(X,Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    plt.show()