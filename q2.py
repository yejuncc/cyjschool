import numpy as np 

from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt

np.random.seed(1847)

class BatchSampler(object):
    '''
    A (very) simple wrapper to randomly sample batches without replacement.

    You shouldn't need to touch this.
    '''
    
    def __init__(self, data, targets, batch_size):
        self.num_points = data.shape[0]
        self.features = data.shape[1]
        self.batch_size = batch_size

        self.data = data
        self.targets = targets

        self.indices = np.arange(self.num_points)

    def random_batch_indices(self, m=None):
        '''
        Get random batch indices without replacement from the dataset.

        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        if m is None:
            indices = np.random.choice(self.indices, self.batch_size, replace=False)
        else:
            indices = np.random.choice(self.indices, m, replace=False)
        return indices 

    def get_batch(self, m=None):
        '''
        Get a random batch without replacement from the dataset.

        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        indices = self.random_batch_indices(m)
        X_batch = np.take(self.data, indices, 0)
        y_batch = self.targets[indices]
        return X_batch, y_batch  


class GDOptimizer(object):
    '''
    A gradient descent optimizer with momentum

        lr - learning rate
        beta - momentum hyperparameter
    '''

    def __init__(self, lr, beta):
        self.lr = lr
        self.beta = beta
        self.vel = 0.0

    def update_params(self, params, grad):
        # Update parameters using GD with momentum and return
        # the updated parameters
        update_value = self.beta * self.vel - self.lr * grad
        self.vel = update_value
        params += update_value
        return params


class SVM(object):
    '''
    A Support Vector Machine
    '''

    def __init__(self, c, feature_count):
        self.c = c
        self.w = np.random.normal(0.0, 0.1, feature_count+1)
        
    def hinge_loss(self, X, y):
        '''
        Compute the hinge-loss for input data X (shape (n, m)) with target y (shape (n,)).

        Returns a length-n vector containing the hinge-loss per data point.
        '''
        # Implement hinge loss
        # Add bias term to X
        X = np.concatenate((np.ones((np.shape(X)[0], 1)), X), axis=1)
        l = 1 - y * (X.dot(self.w.transpose()))
        l[l<0] = 0
        return l


    def grad(self, X, y):
        '''
        Compute the gradient of the SVM objective for input data X (shape (n, m))
        with target y (shape (n,))

        Returns the gradient with respect to the SVM parameters (shape (m,)).
        '''
        # Compute (sub-)gradient of SVM objective
        hinge_loss = self.hinge_loss(X,y)
        X = np.concatenate((np.ones((np.shape(X)[0], 1)), X), axis=1)
        grad = []
        for i in range(np.shape(X)[0]):
            if hinge_loss[i] > 0:
                g = - y[i]*(X[i])
                grad.append(g)

        grad = np.array(grad)
        sum = np.sum(grad,axis=0)
        grad = self.w +self.c * sum / np.shape(y)
        return grad

    def classify(self, X):
        '''
        Classify new input data matrix (shape (n,m)).

        Returns the predicted class labels (shape (n,))
        '''
        # Classify points as +1 or -1
        X = np.concatenate((np.ones((np.shape(X)[0], 1)), X), axis=1)
        labels = X.dot(self.w.transpose())
        labels[labels >= 0] = 1
        labels[labels < 0] = -1
        return labels

def load_data():
    '''
    Load MNIST data (4 and 9 only) and split into train and test
    '''
    mnist = fetch_mldata('MNIST original', data_home='./data')
    label_4 = (mnist.target == 4)
    label_9 = (mnist.target == 9)

    data_4, targets_4 = mnist.data[label_4], np.ones(np.sum(label_4))
    data_9, targets_9 = mnist.data[label_9], -np.ones(np.sum(label_9))

    data = np.concatenate([data_4, data_9], 0)
    data = data / 255.0
    targets = np.concatenate([targets_4, targets_9], 0)

    permuted = np.random.permutation(data.shape[0])
    train_size = int(np.floor(data.shape[0] * 0.8))

    train_data, train_targets = data[permuted[:train_size]], targets[permuted[:train_size]]
    test_data, test_targets = data[permuted[train_size:]], targets[permuted[train_size:]]
    print("Data Loaded")
    print("Train size: {}".format(train_size))
    print("Test size: {}".format(data.shape[0] - train_size))
    print("-------------------------------")
    return train_data, train_targets, test_data, test_targets

def optimize_test_function(optimizer, w_init=10.0, steps=200):
    '''
    Optimize the simple quadratic test function and return the parameter history.
    '''
    def func(x):
        return 0.01 * x * x

    def func_grad(x):
        return 0.02 * x

    w = w_init
    w_history = [w_init]

    for _ in range(steps):
        # Optimize and update the history
        grad = func_grad(w)
        w = optimizer.update_params(w, grad)
        w_history.append(w)
    return w_history

def optimize_svm(train_data, train_targets, penalty, optimizer, batchsize, iters):
    '''
    Optimize the SVM with the given hyperparameters. Return the trained SVM.

    SVM weights can be updated using the attribute 'w'. i.e. 'svm.w = updated_weights'
    '''
    batchsample = BatchSampler(train_data,train_targets,batchsize)
    feature_count = np.shape(train_data)[1]
    svm = SVM(penalty, feature_count)
    for i in range(iters):
        x_batch, y_batch = batchsample.get_batch()
        grad = svm.grad(x_batch,y_batch)
        # update w
        svm.w = optimizer.update_params(svm.w, grad)
    return svm

# helper function
def plot(y_values1, y_values2, i):
    x = []
    for num in range(i):
        x.append(num)
    plt.plot(x, y_values1)
    plt.plot(x, y_values2)
    plt.show()


# helper function for accuracy
def accuracy(classify, labels):
    correct = 0
    total = np.shape(classify)[0]
    for i in range(total):
        if classify[i] == labels[i]:
            correct += 1
    return correct/total

if __name__ == '__main__':
    train_data, train_targets, test_data, test_targets = load_data()
    # ======================= 2.1 =====================
    # 2.1 beta = 0.0
    optimizer0 = GDOptimizer(1.0, 0.0)
    w_updates = optimize_test_function(optimizer0)
    # # 2.1 beta = 0.9
    optimizer1 = GDOptimizer(1.0, 0.9)
    w_updates1 = optimize_test_function(optimizer1)
    # plot the result for 2.1
    plot(w_updates,w_updates1,201)

    # ======================= 2.2, 2.3 =====================
    optimizer2 = GDOptimizer(0.05,0)
    op_svm2 = optimize_svm(train_data,train_targets, 1.0, optimizer2, 100, 500)
    train_acc = accuracy(op_svm2.classify(train_data),train_targets)
    hinge_loss = np.average(op_svm2.hinge_loss(train_data,train_targets))

    test_acc = accuracy(test_targets,op_svm2.classify(test_data))
    hinge_loss_test = np.average(op_svm2.hinge_loss(test_data,test_targets))


    print("2.3 accuracy and loss for alpha = 0.05, C = 1.0, m = 100, T = 500 and beta = 0" )
    print("training acccuracy is ...", train_acc)
    print("testing acccuracy is ...", test_acc)
    print("training hinge loss is ...", hinge_loss)
    print("testing hinge loss is ...", hinge_loss_test)
    # plot
    plot = op_svm2.w[1:].reshape((28, 28))
    plt.imshow(plot)
    plt.show()




    optimizer3 = GDOptimizer(0.05, 0.1)
    op_svm3 = optimize_svm(train_data, train_targets, 1.0, optimizer3, 100, 500)
    train_acc2 = accuracy(train_targets,op_svm3.classify(train_data))
    test_acc2 = accuracy(test_targets,op_svm3.classify(test_data))

    hinge_loss2 = np.average(op_svm3.hinge_loss(train_data,train_targets))
    hinge_loss_test2 = np.average(op_svm3.hinge_loss(test_data,test_targets))
    print("2.3 accuracy and loss for alpha = 0.05, C = 1.0, m = 100, T = 500 and beta = 0.1" )
    print("training acccuracy is ...", train_acc2)
    print("testing acccuracy is ...", test_acc2)
    print("training hinge loss is ...", hinge_loss2)
    print("testing hinge loss is ...", hinge_loss_test2)
    # plot w
    plot2 = op_svm3.w[1:].reshape((28, 28))
    plt.imshow(plot2)
    plt.show()