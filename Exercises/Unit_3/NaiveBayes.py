import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from tqdm import tqdm

class NaiveBayes:
    def __init__(self):
        self.parameters = {}
        self.probabilities = {}
        self.classes = []
    
    def _set_classes(self, y: pd.Series):
        """Sets the classes of the model.

        Parameters
        ----------
        y : pd.Series
            Target variable.
        """
        self.classes = np.unique(y)
    
    def _gauss_likelihood(self, feature: pd.Series, mean: float, std: float) -> pd.Series:
        """Calculates the likelihood of a feature given a mean and a standard deviation.

        Parameters
        ----------
        feature : pd.Series
            Feature to calculate the likelihood.
        mean : float
            Mean of the values of the feature per class.
        std : float
            Standard deviation of the values of the feature per class.

        Returns
        -------
        pd.Series
            Likelihood of the feature given the mean and the standard deviation.
        """
        exponent = np.exp(-1/2*((feature - mean)/std)**2)
        function = 1/(std*np.sqrt(2*np.pi))*exponent
        return function
    
    def _discret_likelihood(self, feature: pd.Series) -> pd.Series:
        """Calculates the likelihood of each value for discret features.

        Parameters
        ----------
        feature : pd.Series
            Feature to calculate the likelihood.

        Returns
        -------
        pd.Series
            Likelihood of each value in the feature.
        """
        value_counts = feature.value_counts()
        likelihood = feature.map(value_counts / len(feature))
        return likelihood

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Calculates the mean and the standard deviation of the features per class.
        As well as the discrete likelihood of the cathegories that belong to discreat features.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training data.
        y_train : pd.Series
            Target of training data.
        """
        self._set_classes(y_train)
        parameters = {}
        for class_ in self.classes:
            parameters[class_] = {
                'apriori': len(y_train[y_train == class_])/len(y_train)
            }
            for feature in X_train.select_dtypes(include='number').columns:
                parameters[class_][feature] = {}
                parameters[class_][feature]['mean'] = X_train[y_train == class_][feature].mean()
                parameters[class_][feature]['std'] = X_train[y_train == class_][feature].std()

            for feature in X_train.select_dtypes(include='object').columns:
                parameters[class_][feature] = {}
                subset = X_train[y_train == class_][feature]
                parameters[class_][feature]['likelihood'] = subset.value_counts()/len(subset)
            
        self.parameters = parameters
    
    def predict_prob(self, X_test: pd.DataFrame) -> Dict[str, pd.Series]:
        """Predicts the probability of each class given the features.

        Parameters
        ----------
        X_test : pd.DataFrame
            Test data.

        Returns
        -------
        Dict[str, pd.Series]
            Probabilities per class.
        """
        probabilities = {}
        for class_ in self.classes:
            likelihood = 1
            for feat in X_test.columns:
                if X_test[feat].dtype == 'object':
                    likelihood*=self._discret_likelihood(X_test[feat])
                else:
                    likelihood*=self._gauss_likelihood(X_test[feat], self.parameters[class_][feat]['mean'], self.parameters[class_][feat]['std'])
            probabilities[class_] = likelihood*self.parameters[class_]['apriori']

        return probabilities

    def predict_log_prob(self, X_test: pd.DataFrame) -> Dict[str, pd.Series]:
        """Predicts the log probability of each class given the features.

        Parameters
        ----------
        X_test : pd.DataFrame
            Test data.

        Returns
        -------
        Dict[str, pd.Series]
            Log-probabilities per class.
        """
        log_probabilities = {}
        for class_ in self.classes:
            likelihood = 0
            for feat in X_test.columns:
                if X_test[feat].dtype == 'object':
                    likelihood+=np.log(self._discret_likelihood(X_test[feat]))
                else:
                    likelihood+=np.log(self._gauss_likelihood(X_test[feat], self.parameters[class_][feat]['mean'], self.parameters[class_][feat]['std']))
            log_probabilities[class_] = likelihood+np.log(self.parameters[class_]['apriori'])
        return log_probabilities

    def predict(self, X_test: pd.DataFrame, method: Optional[str] = None) -> pd.Series:
        """Predicts the class of the features.

        Parameters
        ----------
        X_test : pd.DataFrame
            Test data.
        method : None | str, optional
            Method to predict the class. This can be None or 'log'. 
            Defaults to None.

        Returns
        -------
        pd.Series
            Predicted class.
        """
        if method == 'log':
            probabilities = pd.DataFrame(self.predict_log_prob(X_test))
        else:
            probabilities = pd.DataFrame(self.predict_prob(X_test))

        return probabilities.idxmax(axis="columns")
    
    def _cross_validation_split(self, k: int, data: pd.DataFrame) -> List[Dict[str, pd.DataFrame]]:
        """Splits the dataset into k folds.

        Parameters
        ----------
        k : int
            Number of folds.
        data : pd.DataFrame
            Dataset.

        Returns
        -------
        List[Dict[str, pd.DataFrame]]
            List of dictionaries with the train and test sets.
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

        Parameters
        ----------
        y_true : pd.Series
            True target.
        y_pred : pd.Series
            Predicted target.

        Returns
        -------
        float
            Accuracy of the model.
        """
        return (y_true == y_pred).sum()/len(y_true)
        
    def cross_validation_evaluate(self, k: int, data: pd.DataFrame, method: Optional[str] = None) -> tuple[List[float], float]:
        """Evaluates the model using cross-validation.

        Parameters
        ----------
        k : int
            Number of folds.
        data : pd.DataFrame
            Dataset.
        method : None | str, optional
            Method to predict the class. This can be None or 'log'. 
            Defaults to None.

        Returns
        -------
        tuple[List[float], float]
            Tuple with List of accuracies and mean accuracy.
        """
        folds = self._cross_validation_split(k=k, data=data)
        accuracies = []
        for fold in tqdm(folds, desc='Folds', total=k):
            X_train = fold['train'].iloc[:, :-1]
            y_train = fold['train']['class']
            X_test = fold['test'].iloc[:, :-1]
            y_test = fold['test']['class']

            self.fit(X_train=X_train, y_train=y_train)
            y_pred = self.predict(X_test=X_test, method=method)
            
            accuracy = self.accuracy_metric(y_true=y_test, y_pred=y_pred)
            accuracies.append(accuracy)
            mean_accuracy = np.mean(accuracies)
        return accuracies, mean_accuracy

if __name__ == '__main__':
    # print("Loading data...")
    # from ucimlrepo import fetch_ucirepo
    # iris_dataset = fetch_ucirepo(id=53)
    # X_iris = iris_dataset.data.features
    # y_iris = iris_dataset.data.targets['class']
    # iris_df = pd.concat([X_iris, y_iris], axis=1)

    test_df = pd.DataFrame(
        {
            'Colores' : ['Rojo', 'Verde', 'Rojo', 'Azul', 'Verde'], 
            'Tamaño': ['Grande', 'Pequeño', 'Pequeño', 'Grande', 'Grande'],
            'Length': [5, 10, 15, 20, 30],    
            'class': ['A', 'B', 'A', 'B', 'A']
        }
    )

    print("Fitting the models...")
    nb = NaiveBayes()
    nb.fit(X_train=test_df.iloc[:, :-1], y_train=test_df['class'])
    print(test_df)
    result = test_df.groupby('class')['Colores'].value_counts().unstack(fill_value=0).reindex(columns=test_df['Colores'].unique(), fill_value=0)
    print(result)
    alpha=1
    laplace_smth_result = (result+alpha) / (result.sum(axis=1) + alpha*len(test_df['Colores'].unique()))
    print(len(test_df['Colores'].unique()).values[:, None])
    # print("Evaluating the model with no log-probabilities...")
    # cv_ev = nb.cross_validation_evaluate(k=5, data=iris_df)

    # print("Evaluating the model with log-probabilities...")
    # cv_ev_log = nb.cross_validation_evaluate(k=5, data=iris_df, method='log')
    
    # print(cv_ev)
    # print(cv_ev_log)