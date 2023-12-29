import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class ELUClassifier:
    """
    sklearn style implementation of a multiclass classifier using numpy. Classifier uses ELU activation
    and converts the logits to multiclass probabilities using softmax

    network pipeline ->

    z = wx + b
    z = elu(z)
    z = batch_norm(z) (if batch norm is implemented)
    z = dropout(z)
    p = softmax(z)
    loss = crossentropy(y,p)
    """

    def __init__(self, lr=1e-2, epochs=100, l2=2e-4, batch_size=None, dropout_rate=0.04, batch_norm=False,
                 end_lr=None, sgd_momentum=False, optimizer='sgd', patience=3):
        """
        :param lr: learning rate
        :param epochs: number training epochs
        :param l2: l2 regularization term
        :param batch_size: training batch size (if None, classifier computes the batch size)
        :param end_lr: ending learning rate for linear lr decay (if None, lr is constant)
        :param sgd_momentum: to use momentum if optimizer is sgd
        :param optimizer: parameter optimizaton algorithm ('sgd' or 'adam')
        :param patience: early stopping parameter
        :param dropout_rate: shuts off provided percent of weights with a boolean mask for regularization
        :param batch_norm: implements batch normalization after elu activation
        """

        # input params
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.end_lr = end_lr
        self.batch = batch_size
        self.end_lr = end_lr
        self.momentum = sgd_momentum
        self.opt = optimizer
        self.patience = patience
        self.drop = dropout_rate
        self.norm = batch_norm

        # track loss
        self.losses = {'train': [], 'val': []}

        # model and optimizer params
        self.weights = None
        self.bias = None
        self.w_velocity = 0
        self.b_velocity = 0
        self.w_m1 = 0
        self.w_m2 = 0
        self.b_m1 = 0
        self.b_m2 = 0
        self.classes = None

        # batch norm params
        self.gamma = None
        self.beta = None
        self.running_mean = None
        self.running_var = None

    @staticmethod
    def initalize_weights(inp, out):

        """Glorot initialization"""

        np.random.seed(10)
        std = (2 / inp) ** 0.5
        return np.random.normal(0, std, (inp, out))

    def softmax(self, logits):
        """
        softmax implementation using numpy, converts vectors to scaler probabilites
        """
        max_logits = np.max(logits, axis=-1, keepdims=True)
        logits -= max_logits
        return np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)

    def compute_loss(self, prob, label, training=True):
        """
        computes the categorical crossentropy loss
        applies l2 regularization if training is True
        """
        loss = -np.mean(np.log(prob + 1e-7)[np.arange(len(label)), label])
        if training:
            l2_reg = self.l2 * 0.5 * np.sum(self.weights ** 2)
            loss += l2_reg
        return loss

    def fit(self, x, y):

        """training the classifier"""

        # split data in train and val
        x, xval, y, yval = train_test_split(x, y, train_size=0.85, random_state=0)

        # initialize params
        self.classes = len(np.unique(y))
        self.weights = self.initalize_weights(x.shape[-1], self.classes)
        self.bias = np.zeros((self.classes,))

        # initialize batch norm params (initialized if batch norm is True)
        if self.norm:
            self.gamma, self.running_var = np.ones(self.classes), np.ones(self.classes)
            self.beta, self.running_mean = np.zeros(self.classes), np.zeros(self.classes)

        # set batch size and train steps
        self.batch = self.batch if self.batch is not None else self.set_batch_size(x)
        n_batches = len(x) // self.batch if len(x) % self.batch == 0 else len(x) // self.batch + 1

        batch_norm_params = {'gamma': self.gamma, 'beta': self.beta,
                             'running_mean': self.running_mean, 'running_var': self.running_var}

        params = [self.weights, self.bias]

        patience = self.patience

        # linear lr decay
        if self.end_lr is not None:
            lr_decay = np.linspace(self.lr, self.end_lr, n_batches)

        for e in tqdm(range(self.epochs)):
            loss = 0

            for i in range(n_batches):
                xbatch = x[i * self.batch:(i + 1) * self.batch]
                ybatch = y[i * self.batch:(i + 1) * self.batch]

                # forward pass

                # compute logits and apply elu
                z = np.clip(xbatch @ self.weights + self.bias, -700, 700)
                elu = np.where(z > 0, z, 1.6732 * (np.exp(z) - 1))
                norm = elu

                # apply batch norm
                if self.norm:
                    norm = self.batch_norm(elu)

                # apply dropout
                dropout_mask = np.linspace(0, 1, len(ybatch) * self.classes) >= self.drop
                np.random.shuffle(dropout_mask)
                dropout_mask = dropout_mask.reshape((len(ybatch), self.classes))
                drop = norm * dropout_mask

                # compute probs
                probs = self.softmax(drop)

                # compute loss
                loss += self.compute_loss(probs, ybatch)

                # one hot encode target label
                ohe = np.zeros((len(ybatch), self.classes))
                ohe[np.arange(len(ybatch)), ybatch] = 1

                # gradient calculations

                # dL/dz = dL/dS * dS/ddrop * ddrop/dnorm
                dL_dnorm = (probs - ohe) * dropout_mask / len(ybatch)
                if self.norm:
                    dL_dgamma = np.mean(dL_dnorm * (elu - self.running_mean) / (self.running_var + 1e-5) ** 0.5, axis=0)
                    dL_dbeta = np.mean(dL_dnorm, axis=0)

                # dL/dz = dL/dz * dnorm/dselu * dselu/dz
                dL_dz = (dL_dnorm * np.where(z > 0, 1, 1.6732 * np.exp(z)))
                if self.norm:
                    dL_dz *= (self.gamma / (self.running_var + 1e-5) ** 0.5)

                dL_dw = xbatch.T @ dL_dz + self.l2 * self.weights
                dL_db = np.mean(dL_dz, axis=0)

                if self.end_lr is not None:
                    self.lr = lr_decay[i]

                # backward pass

                # update batch norm params
                if self.norm:
                    self.gamma = self.gamma - self.lr * dL_dgamma
                    self.beta = self.beta - self.lr * dL_dbeta

                # update weights and bias with the optimizer
                if self.opt == 'sgd':
                    if self.momentum:
                        self.w_velocity = self.w_velocity * 0.9 - self.lr * dL_dw
                        self.b_velocity = self.b_velocity * 0.9 - self.lr * dL_db
                        self.weights += self.w_velocity
                        self.bias += self.b_velocity
                    else:
                        self.weights = self.weights - self.lr * dL_dw
                        self.bias = self.bias - self.lr * dL_db

                elif self.opt == 'adam':
                    self.w_m1 = self.w_m1 * 0.9 + (1 - 0.9) * dL_dw
                    self.w_m2 = self.w_m2 * 0.999 + (1 - 0.999) * (dL_dw ** 2)
                    w_m1 = self.w_m1 / (1 - 0.9 ** (i + 1))
                    w_m2 = self.w_m2 / (1 - 0.999 ** (i + 1))
                    self.weights = self.weights - self.lr * (w_m1 / (w_m2 ** 0.5 + 1e-8))
                    self.b_m1 = self.b_m1 * 0.9 + (1 - 0.9) * dL_db
                    self.b_m2 = self.b_m2 * 0.999 + (1 - 0.999) * (dL_db ** 2)
                    b_m1 = self.b_m1 / (1 - 0.9 ** (i + 1))
                    b_m2 = self.b_m2 / (1 - 0.999 ** (i + 1))
                    self.bias = self.bias - self.lr * (b_m1 / (b_m2 ** 0.5 + 1e-8))

            # track mean training loss
            self.losses['train'].append(loss / n_batches)

            # run inference
            z = np.clip(xval @ self.weights + self.bias, -700, 700)
            z = np.where(z > 0, z, 1.6732 * (np.exp(z) - 1))
            if self.norm:
                z = self.batch_norm(z, training=False)
            probs = self.softmax(z)
            loss = self.compute_loss(probs, yval, training=False)  # val_loss

            # best params saved
            if e == 0 or loss < min(self.losses['val']):
                params = [self.weights, self.bias]
                if self.norm:
                    batch_norm_params = {'gamma': self.gamma, 'beta': self.beta,
                                         'running_mean': self.running_mean,
                                         'running_var': self.running_var}

                # early stopping
                if e > 0:
                    if loss >= self.losses['val'][-1]:
                        patience -= 1
                    else:
                        patience = self.patience

            # record val loss
            self.losses['val'].append(loss)

            if patience == 0:
                self.weights, self.bias = params
                if self.norm:
                    self.gamma = batch_norm_params['gamma']
                    self.beta = batch_norm_params['beta']
                    self.running_mean = batch_norm_params['running_mean']
                    self.running_var = batch_norm_params['running_var']
                break

            # shuffle data
            tmp = np.concatenate((x, y[:, np.newaxis]), axis=1)
            np.random.shuffle(tmp)
            x, y = tmp[:, :-1], tmp[:, -1].reshape(-1).astype('int')

    def predict(self, x):
        """
        predict the probability of each class
        """
        z = np.clip(x @ self.weights + self.bias, -700, 700)
        z = np.where(z > 0, z, 1.6732 * (np.exp(z) - 1))
        if self.norm:
            z = self.batch_norm(z, False)
        return self.softmax(z)

    def plot_loss(self):
        """
        plot the train and valid losses
        """
        pd.DataFrame(self.losses).plot()
        plt.show()

    def set_batch_size(self, x):
        """
        determines a batch size if not provided
        """
        _, f = x.shape
        k = np.round(np.log2(f))
        return int(2 ** k)

    def batch_norm(self, z, training=True):
        """
        implements batch normalization
        calculates running statistics during training with batch momentum = 0.9
        running_stat = running_stat * batch_moment + (1 - batch_moment) * stat(z)
        Z = (z - r_mean) / (r_var + epsilon) ** 0.5
        normalization = gamma X Z + beta
        """
        if training:
            self.running_mean = 0.9 * self.running_mean + 0.1 * np.mean(z, axis=0)
            self.running_var = 0.9 * self.running_var + 0.1 * np.var(z, axis=0, )

        norm = self.gamma * (z - self.running_mean) / (self.running_var + 1e-5) ** 0.5 + self.beta
        return norm
