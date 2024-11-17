import pandas as pd
import numpy as np
from typing import Dict, List

class NaiveBayes:
    def __init__(self):
        self.parameters = {}
        self.probabilities = {}
        self.classes = []
    
    def _set_classes(self, y: pd.Series):
        """Sets the classes of the model.

        Args:
            y (pd.Series): Target variable.
        """
        self.classes = np.unique(y)
    
    def _gauss_likelihood(feature: pd.Series, mean: float, std: float) -> pd.Series:
        """Calculates the likelihood of a feature given a mean and a standard deviation.

        Args:
            feature (pd.Series): Feature to calculate the likelihood.
            mean (float): Mean of the values of the feature per class.
            std (float): Standard deviation of the values of the feature per class.

        Returns:
            pd.Series: Likelihood of the feature given the mean and the standard deviation.
        """
        exponent = np.exp(-1/2*((feature - mean)/std)**2)
        function = 1/(std*np.sqrt(2*np.pi))*exponent
        return function

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Calculates the mean and the standard deviation of the features per class.

        Returns:
            Dict[str, Dict[str, Dict[str, float]]]: Mean and standard deviation of the features per class.
        """
        self._set_classes(y_train)
        parameters = {}
        for class_ in self.classes:
            parameters[class_] = {
                'apriori': len(y_train[y_train == class_])/len(y_train)
            }

            for feature in X_train.columns:
                parameters[class_][feature] = {}
                parameters[class_][feature]['mean'] = X_train[y_train == class_][feature].mean()
                parameters[class_][feature]['std'] = X_train[y_train == class_][feature].std()
            
        self.parameters = parameters

        return parameters
    
    def predict_prob(self, X_test: pd.DataFrame) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Predicts the probability of each class given the features.

        Returns:
            Dict[str, Dict[str, Dict[str, float]]]: Mean and standard deviation of the features per class.
        """
        probabilities = {}
        for class_ in self.classes:
            likelihood = 1
            for feat in X_test.columns:
                likelihood*=self._gauss_likelihood(X_test[feat], self.parameters[class_][feat]['mean'], self.parameters[class_][feat]['std'])
            probabilities[class_] = likelihood*self.parameters[class_]['apriori']
        return probabilities
        