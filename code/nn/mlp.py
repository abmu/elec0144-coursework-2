import numpy as np
from .optim import Optimiser
from enum import Enum
from dataclasses import dataclass


ACTIVATION = {
    'linear': lambda x: x,
    'tanh': lambda x: np.tanh(x),
    'relu': lambda x: np.maximum(0, x),
    'sigmoid': lambda x: 1 / (1 + np.exp(-x))
}

DERIVATIVE = {
    'linear': lambda x: np.ones_like(x),
    'tanh': lambda x: 1 - np.tanh(x) ** 2,
    'relu': lambda x: (x > 0).astype(x.dtype),
    'sigmoid': lambda x: (s := ACTIVATION['sigmoid'](x)) * (1 - s)
}


class Task(Enum):
    CLASSIFICATION = 'classification'
    REGRESSION = 'regression'


def _to_task_enum(value: str | Task) -> Task:
    """
    Ensure the value is a valid Task enum

    Args:
        value: Input string or Task

    Returns:
        A Task enum
    """
    if isinstance(value, Task):
        return value
    return Task(value)


@dataclass
class TrainingResult:
    train_losses: list[np.float64]
    val_losses: list[np.float64]
    train_accs: list[np.float64]
    val_accs: list[np.float64]


class MultilayerPerceptron:
    def __init__(self, layers: list[tuple[int, str]], optimiser: Optimiser, task: str = Task.REGRESSION.value, l2_reg: float = 0.0, seed: int = 144) -> None:
        self.layers = layers.copy()  # [(layer size, activation function), ...]
        self.optimiser = optimiser
        self.task = _to_task_enum(task)
        self.weights = [None] * len(layers)
        self.biases = [None] * len(layers)
        self.x_mean, self.x_std = None, None
        self.l2_reg = l2_reg  # L2-Regularisation lambda, recommended to set as 1e-2 -- higher value => more regularisation
        self.rng = np.random.default_rng(seed)  # random generator
        self._init_params()


    def _init_params(self) -> None:
        """
        Initialises the weights and biases of the neural network

        Args:
            seed: Seed value to be used for random generator
        """
        # Initialise weights and biases with random values
        # NOTE: 0-index will not be used
        for i in range(1, len(self.layers)):
            prev_layer = self.layers[i-1][0]
            next_layer = self.layers[i][0]
            func = self.layers[i][1]

            if func == 'relu':
                # He initialisation
                std = np.sqrt(2.0 / prev_layer)
            else:  # linear, tanh, sigmoid
                # Glorot initialisation
                std = np.sqrt(2.0 / (prev_layer + next_layer))

            w = self.rng.normal(0, std, size=(next_layer, prev_layer))
            b = np.zeros((next_layer, 1))
            self.weights[i] = w
            self.biases[i] = b


    def _forward(self, input: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Performs a forward pass through the neural network.

        Args:
            input: Inputs for the input layer of the neural network

        Returns:
            A cache (in the form of a list of tuples) which contains the pre-activation and activation value of every neuron in each layer of the network
        """
        cache = [None] * len(self.layers)

        # Input layer
        x = input.reshape(-1, 1)
        cache[0] = (np.array([]), x)  # no activation function for first layer

        # Hidden layers
        for i in range(1, len(self.layers)-1):
            w = self.weights[i]
            b = self.biases[i]
            z = w @ x + b  # pre-activation value of layer
            func = self.layers[i][1]
            a = ACTIVATION[func](z)
            cache[i] = (z, a)
            x = a

        # Output layer
        w = self.weights[-1]
        b = self.biases[-1]
        z = w @ x + b
        func = self.layers[-1][1]
        a = ACTIVATION[func](z)
        cache[-1] = (z, a)

        return cache


    def _loss(self, output: np.ndarray, actual: np.ndarray) -> np.float64:
        """
        Calculate the loss for a single sample

        Args:
            output: Output layer of neural network
            actual: Actual truth value

        Returns
            The loss value calculated between the network output and actual value
        """
        actual = actual.reshape(-1, 1)
        error = 0.5 * np.sum((output - actual)**2)  # instantaneous squared error

        reg_loss = 0.0
        if self.l2_reg > 0:
            for w in self.weights[1:]:
                reg_loss += 0.5 * self.l2_reg * np.sum(w ** 2)
        
        return error + reg_loss 


    def _backprop(self, cache: list[tuple[np.ndarray, np.ndarray]], y: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Performs a backwards pass through the neural network.

        Args:
            cache: The pre-activation and activation value of every neuron in each layer of the network based on the inputs
            y: Actual truth value outputs for given inputs

        Returns
            The gradients for the weights and biases after the backwards pass
        """
        grad_w = [None] * len(self.layers)
        grad_b = [None] * len(self.layers)

        # Output layer
        y = y.reshape(-1, 1)
        z, a = cache[-1]
        func = self.layers[-1][1]
        a_prev = cache[-2][1]

        dL_da = a - y  # derivative of loss function with respect to activation layer
        da_dz = DERIVATIVE[func](z)  # derivative of activation layer with respect to pre-activation values
        delta = dL_da * da_dz
        
        dz_dw = a_prev  # derivative of pre-activation values with respect to weights
        dz_db = np.ones_like(self.biases[-1])  # derivative of pre-activation values with respect to biases
        dz_dprev = self.weights[-1]  # derivative of last layer pre-activation values with respect to previous activation layer

        # Loss function gradients
        grad_w[-1] = delta @ dz_dw.T
        if self.l2_reg > 0:
            grad_w[-1] += self.l2_reg * self.weights[-1]

        grad_b[-1] = delta * dz_db
        grad_prev = dz_dprev.T @ delta

        # Hidden layers
        for i in range(len(self.layers)-2, 0, -1):
            z, a = cache[i]
            func = self.layers[i][1]
            a_prev = cache[i-1][1]

            dL_da = grad_prev
            da_dz = DERIVATIVE[func](z)
            delta = dL_da * da_dz

            grad_w[i] = delta @ a_prev.T
            if self.l2_reg > 0:
                grad_w[i] += self.l2_reg * self.weights[i]

            grad_b[i] = delta
            grad_prev = self.weights[i].T @ delta

        return grad_w, grad_b
    

    @property
    def weights_biases(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Create a deep copy of the weights and biases

        Returns:
            A tuple of a deep copy of the weights and biases
        """
        n = len(self.weights)
        weights = [None] + [self.weights[i].copy() for i in range(1, n)]
        biases = [None] + [self.biases[i].copy() for i in range(1, n)]
        return weights, biases


    def _normalise_xs(self, xs: np.ndarray) -> np.ndarray:
        """
        Normalise the input

        Args:
            xs: Input values

        Returns:
            Normalised output values
        """
        # if self.x_min is None or self.x_max is None:
        #     return xs
        # return 2 * (xs - self.x_min) / (self.x_max - self.x_min) - 1
        if self.x_mean is None or self.x_std is None:
            return xs
        return (xs - self.x_mean) / (self.x_std + 1e-8)
    

    def _accuracy(self, xs: np.ndarray, ys: np.ndarray) -> float:
        """
        Compute the classifcation accuracy

        Args:
            xs: Input values
            ys: Truth output values

        Return:
            Accuracy measured in range [0, 1]
        """
        ys_pred = self.predict(xs)

        # Convert from 3 outputs to 1 maximum value output -- predicted class
        ys_true_cls = self.to_classification(ys)
        ys_pred_cls = self.to_classification(ys_pred)

        return np.mean(ys_pred_cls == ys_true_cls)
    

    @staticmethod
    def to_classification(ys: np.ndarray) -> np.ndarray:
        """
        Convert from n outputs to 1 maximum value output -- predicted class

        Args:
            ys: Output values

        Returns:
            The classifications
        """
        ys_cls = ys.argmax(axis=1)
        return ys_cls
    

    def train(self, iterations: int, train_data: tuple[np.ndarray, np.ndarray], val_data: tuple[np.ndarray, np.ndarray] = None, train_loss_goal: float = 1e-4, val_patience: int = 10) -> TrainingResult:
        """
        Train the neural netwrok on a given data set

        Args:
            iterations: The number of training iterations to run
            train_data: Training input values and output truth values
            val_data: Validation input values and output truth values
            train_loss_goal: The early stopping threshold for training
            val_patience: How long to allow no improvement on validation loss

        Returns:
            The average train losses per iteration, average validation losses, training accuracy, and validation accuracy
        """
        # Normalise input based on training data only
        xs_train, ys_train = train_data
        self.x_mean, self.x_std = xs_train.mean(axis=0), xs_train.std(axis=0)
        xs_train_norm = self._normalise_xs(xs_train)

        if val_data:
            xs_val, ys_val = val_data
            xs_val_norm = self._normalise_xs(xs_val)

            # Best validation loss
            best_val_loss = np.inf
            best_val_epoch = -1
            best_val_weights, best_val_biases = self.weights_biases
            val_patience_count = 0

        # Train and validate neural network
        self.optimiser.reset()
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []

        for epoch in range(1, iterations+1):

            # Training
            total_train_loss = 0

            for i, (x, y) in enumerate(zip(xs_train_norm, ys_train)):
                cache = self._forward(x)
                _, y_pred = cache[-1]
                total_train_loss += self._loss(y_pred, y)

                grad_w, grad_b = self._backprop(cache, y)
                self.weights, self.biases = self.optimiser.update(*self.weights_biases, grad_w, grad_b)
            
            mean_train_loss = total_train_loss / len(xs_train)
            train_losses.append(mean_train_loss)

            # Validation
            if val_data:
                total_val_loss = 0

                for i, (x, y) in enumerate(zip(xs_val_norm, ys_val)):
                    cache = self._forward(x)
                    _, y_pred = cache[-1]
                    total_val_loss += self._loss(y_pred, y)

                mean_val_loss = total_val_loss / len(xs_val)
                val_losses.append(mean_val_loss)

                # Check if validation loss has improved
                if mean_val_loss < best_val_loss:
                    best_val_loss = mean_val_loss
                    best_val_epoch = epoch
                    best_val_weights, best_val_biases = self.weights_biases
                    val_patience_count = 0
                else:
                    val_patience_count += 1

                if val_patience_count >= val_patience:
                    # Validation loss no longer improving, so break
                    self.weights, self.biases = best_val_weights, best_val_biases
                    print(f'[STOPPING!] Epoch: {epoch}, Best Validation Loss = {best_val_loss:.6f} at epoch {best_val_epoch}')
                    break

            if self.task == Task.CLASSIFICATION:
                # Training accuracy
                train_acc = self._accuracy(xs_train, ys_train)
                train_accs.append(train_acc)

                # Validation accuracy
                if val_data:
                    val_acc = self._accuracy(xs_val, ys_val)
                    val_accs.append(val_acc)

            if not val_data and mean_train_loss < train_loss_goal:
                # Stop training early -- if there is no validation set and training loss goal reached
                print(f'[STOPPING!] Epoch: {epoch} | Train Loss = {mean_train_loss} < Train Goal = {train_loss_goal}')
                break

            # Progress output
            if epoch % 1000 == 0:
                val_loss_msg = f' | Validation Loss = {mean_val_loss:.6f}' if val_data else ''
                train_acc_msg = f', Train Acc = {train_acc:.6f}' if self.task == Task.CLASSIFICATION else ''
                val_acc_msg = f', Validation Acc = {val_acc:.6f}' if self.task == Task.CLASSIFICATION and val_data else ''
                print(f'Epoch: {epoch} | Train Loss = {mean_train_loss:.6f}{train_acc_msg}{val_loss_msg}{val_acc_msg}')

        return TrainingResult(
            train_losses=train_losses,
            val_losses=val_losses,
            train_accs=train_accs,
            val_accs=val_accs
        )
    

    def predict(self, xs: np.ndarray) -> np.ndarray:
        """
        Predicts the output ys for a given input xs

        Args:
            xs: Input values

        Returns:
            Predicted output values for the given input
        """
        xs_norm = self._normalise_xs(xs)
        return np.array([self._forward(x)[-1][1].flatten() for x in xs_norm])
