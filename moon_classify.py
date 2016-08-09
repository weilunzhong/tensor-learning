import tensorflow as tf
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import matplotlib
import matplotlib.pyplot as plt


matplotlib.rcParams['figure.figsize'] = (10.0, 8.0)
#plot the generated dataset
np.random.seed(0)
X, y = sklearn.datasets.make_moons(1000, noise=0.20)
print X.shape
plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)

# Helper function to plot a decision boundary.
# If you don't fully understand this function don't worry, it just generates the contour plot below.
def plot_decision_boundary(pred_func):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)

def calculate_loss(model):
    W1, W2, b1, b2 = model['W1'], model['W2'], model['b1'], model['b2']
    #forward propagation
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_score = np.exp(z2)
    probs = exp_score / np.sum(exp_score, axis=1, keepdims=True)
    correct_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(correct_logprobs)
    data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1./num_examples * data_loss

def predict(model, x):
    W1, W2, b1, b2 = model['W1'], model['W2'], model['b1'], model['b2']
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_score = np.exp(z2)
    probs = exp_score / np.sum(exp_score, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)

def build_model(nn_hdim, num_passes=20000, print_loss=False):
    np.random.seed(0)
    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, nn_output_dim))

    for idx in xrange(0, num_passes):
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_score = np.exp(z2)
        probs = exp_score / np.sum(exp_score, axis=1, keepdims=True)

        delta3 = probs
        delta3[range(num_examples), y] -= 1
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)

        dW2 += reg_lambda * W2
        dW1 += reg_lambda * W1

        W1 += -epsilon * dW1
        W2 += -epsilon * dW2
        b1 += -epsilon * db1
        b2 += -epsilon * db2

        model = {"W1": W1, "W2": W2, "b1": b1, "b2": b2}

        if print_loss and idx % 1000 == 0:
            print "Loss after iteration {0}: {1}".format(idx, calculate_loss(model))
    return model



def network():
    global num_examples, nn_input_dim, nn_output_dim, epsilon, reg_lambda
    num_examples = len(X)
    nn_input_dim = 2
    nn_output_dim = 2
    num_hidden_layer = 4

    epsilon = 0.01
    reg_lambda = 0.01

    model = build_model(num_hidden_layer, print_loss=True)
    plot_decision_boundary(lambda x: predict(model, x))
    plt.title("Decesion Bouondry for hidden layer size {0}".format(num_hidden_layer))

    plt.show()

def main():
    clf = sklearn.linear_model.LogisticRegressionCV()
    print clf.fit(X,y)

    plot_decision_boundary(lambda x: clf.predict(x))
    plt.title("logistic Regression")

    plt.show()

if __name__ == "__main__":
    network()
