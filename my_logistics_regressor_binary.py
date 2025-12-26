import numpy as np


class MyCustomBinaryLogisticRegression:

    def __init__(self, learning_rate=0.045, epochs=10000, verbose=True):
        self.__learning_rate = learning_rate
        self.verbose_ = verbose
        self.__epochs = epochs
        self.__is_already_fitted = False
        self.__weights = None

    def fit(self, X, Y):

        if self.__is_already_fitted:
            raise RuntimeError("Model has been already fitted!")

        # adding that 1's column to multiply with bias

        column_to_add = np.ones(shape=(X.shape[0], 1))
        X_NEW = np.hstack((column_to_add, X))

        weights = np.ones(shape=(X_NEW.shape[1],))

        for i in range(self.__epochs):
            errors = []

            for col in range(X_NEW.shape[1]):
                error = 0

                for row in range(X_NEW.shape[0]):
                    y_true = Y[row]
                    coefficients = X_NEW[row]

                    z = np.dot(coefficients, weights)

                    y_pred = 1 / (1 + np.exp(-z))
                    error += (y_pred - y_true) * X_NEW[row][col]

                errors.append(error)

            new_weights = weights - self.__learning_rate * np.array(errors)

            if np.all(abs(new_weights - weights) <= 1e-12):
                if self.verbose_:
                    print(f"Converged after {i+1} epochs")
                weights = new_weights
                break

            weights = new_weights

        self.__weights = weights
        self.__is_already_fitted = True

    def predict(self, X):

        if not self.__is_already_fitted:
            raise RuntimeError("Model should be fit first!")

        predictions = []

        for row_index in range(X.shape[0]):
            coefficients = [1]
            coefficients.extend(X[row_index, :].flatten().tolist())
            coefficients = np.array(coefficients)

            predictions.append(1 if np.dot(coefficients, self.__weights) >= 0 else 0)

        return np.array(predictions)

    def get_weights(self):
        if not self.__is_already_fitted:
            raise RuntimeError("Model should be fit first!")

        return self.__weights[1:]

    def get_bias(self):
        if not self.__is_already_fitted:
            raise RuntimeError("Model should be fit first!")

        return self.__weights[0]
