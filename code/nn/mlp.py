import numpy as np
from .optim import Optimiser


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


class MultilayerPerceptron:
    def __init__(self, layers: list[tuple[int, str]], optimiser: Optimiser) -> None:
        self.layers = layers  # [(layer size, activation function), ...]
        self.optimiser = optimiser
        self.weights = [None] * len(layers)
        self.biases = [None] * len(layers)
        self.x_mean, self.x_std = None, None
        self._init_params()


    def _init_params(self, seed: int = 144) -> None:
        """
        Initialises the weights and biases of the neural network

        Args:
            seed: Seed value to be used for random generator
        """
        # Initialise weights and biases with random values
        # NOTE: 0-index will not be used
        rng = np.random.default_rng(seed)  # random generator
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

            w = rng.normal(0, std, size=(next_layer, prev_layer))
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
        input_size = self.layers[0][0]
        x = input.reshape(input_size, 1)
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
        return 0.5 * np.sum((output - actual)**2)


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
            grad_b[i] = delta
            grad_prev = self.weights[i].T @ delta

        return grad_w, grad_b
    

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
    

    def train(self, xs: np.ndarray, ys: np.ndarray, iterations: int, loss_goal: float = 1e-3) -> list[np.float64]:
        """
        Train the neural netwrok on a given data set

        Args:
            xs: Input values
            ys: Truth output values
            iterations: The number of training iterations to run
            loss_goal: The early stopping threshold for training

        Returns:
            A list of the total losses per iteration 
        """
        # Normalise input
        self.x_mean, self.x_std = xs.mean(), xs.std()
        xs_norm = self._normalise_xs(xs)

        # Train neural network
        self.optimiser.reset()
        losses = []
        for epoch in range(1, iterations+1):
            error_sse = 0  # sum of squared errors

            for i, (x, y) in enumerate(zip(xs_norm, ys)):
                cache = self._forward(x)
                _, y_pred = cache[-1]
                error_sse += self._loss(y_pred, y)

                grad_w, grad_b = self._backprop(cache, y)
                self.optimiser.update(self.weights, self.biases, grad_w, grad_b)
                
            losses.append(error_sse)

            if error_sse < loss_goal:
                # Stop training early
                print(f'Epoch: {epoch}, Loss = {error_sse:.4f} < Early stopping threshold = {loss_goal:.4f}')
                break

            if epoch % 1000 == 0:
                print(f'Epoch: {epoch}, Loss = {error_sse:.4f}')
        
        return losses
    

    def predict(self, xs: np.ndarray) -> np.ndarray:
        """
        Predicts the output ys for a given input xs

        Args:
            xs: Input values

        Returns:
            Predicted output values for the given input
        """
        xs_norm = self._normalise_xs(xs)
        return np.array([self._forward(x)[-1][1].item() for x in xs_norm])
