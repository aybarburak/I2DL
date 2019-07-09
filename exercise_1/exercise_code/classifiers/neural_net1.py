"""Two Layer Network."""
# pylint: disable=invalid-name
import numpy as np


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network. The net has an input dimension
    of N, a hidden layer dimension of H, and performs classification over C
    classes. We train the network with a softmax loss function and L2
    regularization on the weight matrices. The network uses a ReLU nonlinearity
    after the first fully connected layer.

    In other words, the network has the following architecture:

    input - fully connected layer - ReLU - fully connected layer - softmax

    The outputs of the second fully-connected layer are the scores for each
    class.
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        self.best_params = self.params

    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural
        network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each
          y[i] is an integer in the range 0 <= y[i] < C. This parameter is
          optional; if it is not passed then we only return scores, and if it is
          passed then we instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c]
        is the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of
          training samples.
        - grads: Dictionary mapping parameter names to gradients of those
          parameters  with respect to the loss function; has the same keys as
          self.params.
        """
        # pylint: disable=too-many-locals
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, _ = X.shape

        # Compute the forward pass
        scores = None
        ########################################################################
        # TODO: Perform the forward pass, computing the class scores for the   #
        # input. Store the result in the scores variable, which should be an   #
        # array of shape (N, C).                                               #         
        ########################################################################

        Z1 = X.dot(W1) + b1 # (N * is) *(is * hs) = (N * hs)
        A1 = np.maximum(Z1, 0) # Apply the RELU activation
        Z2 = A1.dot(W2) + b2
        scores = Z2

        ########################################################################
        #                              END OF YOUR CODE                        #
        ########################################################################

        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # Compute the loss
        loss = None
        ########################################################################
        # TODO: Finish the forward pass, and compute the loss. This should     #
        # include both the data loss and L2 regularization for W1 and W2. Store#
        # the result in the variable loss, which should be a scalar. Use the   #
        # Softmax classifier loss. So that your results match ours, multiply   #
        # the regularization loss by 0.5                                       #
        ########################################################################

        y_mat = np.zeros((N, W2.shape[1]))
        y_mat[range(N), y] = 1

        maxCol = np.max(Z2, axis = 1, keepdims = True)
        exp_scores = np.exp(Z2 - maxCol)
        scores = A2 = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        loss = (-1 / N) * np.sum(y_mat * np.log(scores)) # We divide by N here, and not in regularization
        loss += reg / 2. * (np.sum(np.square(W1)) + np.sum(np.square(W2))) # Regularization term

        ########################################################################
        #                              END OF YOUR CODE                        #
        ########################################################################

        # Backward pass: compute gradients
        grads = {}
        ########################################################################
        # TODO: Compute the backward pass, computing the derivatives of the    #
        # weights and biases. Store the results in the grads dictionary. For   #
        # example, grads['W1'] should store the gradient on W1, and be a matrix#
        # of same size                                                         #
        ########################################################################

        # I got the equations from Andrew NG's video for training neural networks:
        # Gradient descent for Neural Networks
        # https://www.coursera.org/learn/neural-networks-deep-learning/lecture/Wh8NI/gradient-descent-for-neural-networks
        dZ2 = A2
        dZ2[range(N), y] -= 1
        dW2 = (1 / N) * np.dot(A1.T, dZ2) + reg * W2 # (hiddenS, outputS)
        db2 = (1 / N) * np.sum(dZ2, axis = 0, keepdims = True) # (1, outputS)

        reluZ1 = Z1
        reluZ1[reluZ1 < 0] = 0
        reluZ1[reluZ1 > 0] = 1

        dZ1 = dZ2.dot(W2.T) * reluZ1 # (N * hs)
        dW1 = (1 / N) * X.T.dot(dZ1) + reg * W1
        db1 = (1 / N) * np.sum(dZ1, axis = 0, keepdims = True)

        grads['W2'] = dW2
        grads['b2'] = db2.reshape(b2.shape)
        grads['W1'] = dW1
        grads['b1'] = db1.reshape(b1.shape)

        ########################################################################
        #                              END OF YOUR CODE                        #
        ########################################################################

        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=1e-5, num_iters=100,
              batch_size=200, verbose=False, iterations_per_epoch = 20):
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means
          that X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning
          rate after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        # pylint: disable=too-many-arguments, too-many-locals
        num_train = X.shape[0]
        dim = X.shape[1]
        iterations_per_epoch = max(num_train // batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        loss_history_epoch = []
        train_acc_history = []
        val_acc_history = []

        best_loss = 20
        flipped = False

        for it in range(num_iters):
            X_batch = None
            y_batch = None

            ####################################################################
            # TODO: Create a random minibatch of training data and labels,     #
            # storing hem in X_batch and y_batch respectively.                 #
            ####################################################################
            X_batch = np.zeros((batch_size, dim))
            y_batch = np.zeros((batch_size), dtype=int)

            mini_idx = np.random.choice(num_train, batch_size)
            for i in range(mini_idx.shape[0]):
                X_batch[i] = X[mini_idx[i]]
                y_batch[i] = y[mini_idx[i]]

            ####################################################################
            #                             END OF YOUR CODE                     #
            ####################################################################

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            if loss < best_loss:
                best_loss = loss
                self.best_params = self.params

            ####################################################################
            # TODO: Use the gradients in the grads dictionary to update the    #
            # parameters of the network (stored in the dictionary self.params) #
            # using stochastic gradient descent. You'll need to use the        #
            # gradients stored in the grads dictionary defined above.          #
            ####################################################################

            self.params['W1'] -= learning_rate * grads['W1']
            self.params['b1'] -= learning_rate * grads['b1']
            self.params['W2'] -= learning_rate * grads['W2']
            self.params['b2'] -= learning_rate * grads['b2']

            ####################################################################
            #                             END OF YOUR CODE                     #
            ####################################################################

            if verbose and it % 10 == 0:
                print('iteration %d / %d: loss %f, learning rate %f' % (it, num_iters, loss, learning_rate))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)
                loss_history_epoch.append(loss)

                # Decay learning rate
                learning_rate *= learning_rate_decay

                """if loss < 2:
                    learning_rate *= learning_rate_decay
                elif len(loss_history_epoch) > 2 and loss_history_epoch[-3] > loss_history_epoch[-2] \
                        and loss_history_epoch[-2] > loss_history_epoch[-1]:
                    learning_rate /= learning_rate_decay # Increase your speed
                else:
                    learning_rate *= learning_rate_decay # Decrease your speed """

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each
          of the elements of X. For all i, y_pred[i] = c means that X[i] is
          predicted to have class c, where 0 <= c < C.
        """
        y_pred = None

        ########################################################################
        # TODO: Implement this function; it should be VERY simple!             #
        ########################################################################
        y_pred = np.zeros(X.shape[0])
        scores = self.loss(X)
        y_pred = np.argmax(scores, axis=1)  # No need to do softmax
        ########################################################################
        #                              END OF YOUR CODE                        #
        ########################################################################

        return y_pred


def neuralnetwork_hyperparameter_tuning(X_train, y_train, X_val, y_val):
    best_net = None # store the best model into this 

    ############################################################################
    # TODO: Tune hyperparameters using the validation set. Store your best     #
    # trained model in best_net.                                               #
    #                                                                          #
    # To help debug your network, it may help to use visualizations similar to #
    # the  ones we used above; these visualizations will have significant      #
    # qualitative differences from the ones we saw above for the poorly tuned  #
    # network.                                                                 #
    #                                                                          #
    # Tweaking hyperparameters by hand can be fun, but you might find it useful#
    # to  write code to sweep through possible combinations of hyperparameters #
    # automatically like we did on the previous exercises.                     #
    ############################################################################
    input_size = X_train.shape[1]
    output_size = y_train.shape[0]
    print(y_train.shape[0])

    # Parameters to tune
    hidden_size = 500 # variable
    batch_size = 200
    learning_rate = 1e-3;
    decay_rate = 0.95
    iterations_per_epoch = 50

    best_net = TwoLayerNet(input_size, hidden_size, 10)
    stats = best_net.train(X_train, y_train, X_val, y_val,
              learning_rate=learning_rate, learning_rate_decay=decay_rate,
              reg=0.31, num_iters=3000,
              batch_size=batch_size, verbose=True, iterations_per_epoch = iterations_per_epoch)

    best_net.params = best_net.best_params

    # Plot the loss function and train / validation accuracies
    import matplotlib.pyplot as plt

    plt.subplots(nrows=2, ncols=1)

    plt.subplot(2, 1, 1)
    plt.plot(stats['loss_history'])
    plt.title('Loss history')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    plt.subplot(2, 1, 2)
    plt.plot(stats['train_acc_history'], label='train')
    plt.plot(stats['val_acc_history'], label='val')
    plt.title('Classification accuracy history')
    plt.xlabel('Epoch')
    plt.ylabel('Clasification accuracy')

    plt.tight_layout()
    plt.show()

    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################
    return best_net
