from random import randint

import numpy as np
from tqdm import trange

########### TO-DO ###########
# 1. Implement perceptron using the weights self.w
#   --> See: def perc(self,X):
# 2. Implement perceptron update
#   --> See: self.w = self.w
# 3. Implement prediction
#   --> See: def predict(self, X):
#       - X is data

class Perceptron:
    """Implements the single layer perceptron.
    """

    def __init__(self, lr=0.5, epochs=100):


        self.lr = lr
        self.epochs = epochs
        self.w = None


    def perc(self,X):

        p = X.shape
        if len(p) == 1 :
            #x = np.insert(X, 0, 1)
            #w2 = np.insert(self.w, 0, 0)
            z = np.dot(self.w, X)
            z = 1 if z >= 0 else -1
            #print(z)
            #return z
            return z
        else:
            #feat, images = X.shape
            #w2 = np.insert(self.w, 0, 0)
            #z = np.dot(self.w,
            #           np.vstack([np.ones(images), X]))
            z = np.dot(self.w, X)
            z[z >= 0] = 1
            z[z < 0] = -1
            #print(z)
            return z

    def fit(self, X, y):

        # n_observations -> number of training examples
        # m_features -> number of features
        n_observations, m_features = X.shape

        #Initialize weights with zero
        self.w = np.zeros(m_features)

        #make sure that labels are only -1 and 1
        y_ = np.array([1 if i > 0 else  -1 for i in y])

        # Empty list to store how many examples were
        # misclassified at every iteration.
        miss_classifications = []

        # Training.
        for epoch in trange(self.epochs):

            # predict all items from the dataset
            predictions = self.perc(np.transpose(X, [1, 0]))
            # compare with gt
            errors = y_ - predictions

            if ((errors == 0).all()):
                print(f'No errors after {epoch} epochs. Training successful!')
            else:
                #sample one prediction at random
                n = randint(0,n_observations-1)
                prediction_for_update = self.perc(X[n,:])
                # update the weights of the perceptron from the random sample
                # self.w += self.lr * ()
                self.w += self.lr*(y_[n]-prediction_for_update)*X[n,:]
            # Appending number of misclassified examples
            # at every iteration.
            miss_classifications.append(errors.shape[0] - np.sum(errors==0))
            #error_epoch = predictions
            #print(error_epoch)

        return miss_classifications


    def predict(self, X):
        test_pred = self.perc(np.transpose(X, [1, 0]))
        return test_pred

