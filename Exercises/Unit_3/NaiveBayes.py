import pandas as pd
import numpy as np
from typing import Dict, List
from tqdm import tqdm

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
    
    def _gauss_likelihood(self, feature: pd.Series, mean: float, std: float) -> pd.Series:
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

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        """Predicts the class of the features.

        Args:
            X_test (pd.DataFrame): Features.

        Returns:
            pd.Series: Predicted class.
        """
        probabilities = pd.DataFrame(self.predict_prob(X_test))
        return probabilities.idxmax(axis="columns")
    
    def _cross_validation_split(self, k: int, data: pd.DataFrame) -> List[Dict[str, pd.DataFrame]]:
        """Splits the dataset into k folds.

        Args:
            k (int): Number of folds.
            X (pd.DataFrame): Features.
            y (pd.Series): Target.

        Returns:
            List[Dict[str, pd.DataFrame]]: List of dictionaries with the train and test sets.
        """
        data = data.sample(frac=1).reset_index(drop=True)
        data['fold'] = data.index % k
        folds = []
        for i in range(k):
            test = data[data['fold'] == i].drop(columns='fold')
            train = data[data['fold'] != i].drop(columns='fold')
            folds.append({'train': train, 'test': test})
        return folds
    
    def accuracy_metric(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        """Calculates the accuracy of the model.

        Args:
            y_true (pd.Series): True target.
            y_pred (pd.Series): Predicted target.

        Returns:
            float: Accuracy of the model.
        """
        return (y_true == y_pred).sum()/len(y_true)
        
    def cross_validation_evaluate(self, k: int, data: pd.DataFrame) -> List[float]:
        """Evaluates the model using cross-validation.

        Args:
            k (int): Number of folds.
            X (pd.DataFrame): Features.
            y (pd.Series): Target.

        Returns:
            List[float]: List of accuracies.
        """
        folds = self._cross_validation_split(k=k, data=data)
        accuracies = []
        for fold in tqdm(folds, desc='Folds', total=k):
            X_train = fold['train'].iloc[:, :-1]
            y_train = fold['train']['class']
            X_test = fold['test'].iloc[:, :-1]
            y_test = fold['test']['class']

            self.fit(X_train=X_train, y_train=y_train)
            y_pred = self.predict(X_test=X_test)
            
            accuracy = self.accuracy_metric(y_true=y_test, y_pred=y_pred)
            accuracies.append(accuracy)
            mean_accuracy = np.mean(accuracies)
        return accuracies, mean_accuracy

if __name__ == '__main__':
    from ucimlrepo import fetch_ucirepo
    iris_dataset = fetch_ucirepo(id=53)
    X_iris = iris_dataset.data.features
    y_iris = iris_dataset.data.targets['class']
    iris_df = pd.concat([X_iris, y_iris], axis=1)

    # Cross-validation evaluation
    nb = NaiveBayes()
    cv_ev = nb.cross_validation_evaluate(k=5, data=iris_df)
    print(cv_ev)