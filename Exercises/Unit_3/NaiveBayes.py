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
        # Impose a lower bound on the probabilities to avoid extremeley small
        # values that can lead to values of zero, therefore indefinite values
        # when calculating the log.
        return np.maximum(function, 1e-10)
    
    def _discret_likelihood_with_laplace_smth(self, X_train: pd.DataFrame, y_train: pd.Series, column: str, alpha:int = 1) -> Dict[str, Dict[str, float]]:
        """Calculates the likelihood of each value for discret features using Laplace smoothing.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training data.
        y_train : pd.Series
            Training target.
        column : str
            Column to calculate the likelihood.
        alpha : int, optional
            Smoothing parameter. Defaults to 1.

        Returns
        -------
        pd.Series
            Likelihood of each value for the feature.
        """        
        dataset = pd.concat([X_train, y_train], axis=1)
        result = dataset.groupby(dataset.columns[-1])[column].value_counts().unstack(fill_value=0)
        result = result.reindex(columns=dataset[column].unique(), fill_value=0)

        # `.values[:, None]` converts Unidimensional array to bidimensional array. From (2,) to (2, 1)
        laplace_smth_result = (result+alpha) / (result.sum(axis=1) + alpha*len(X_train[column].unique())).values[:, None]
        laplace_smth_result_dict = laplace_smth_result.to_dict(orient='index')
        return laplace_smth_result_dict

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
                laplace_smth_result_dict = self._discret_likelihood_with_laplace_smth(X_train, y_train, feature)
                parameters[class_][feature] = laplace_smth_result_dict[class_] 
            
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
                    likelihood*=X_test[feat].map(nb.parameters[class_][feat])
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
                    likelihood+=np.log(X_test[feat].map(nb.parameters[class_][feat]))
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
        """Calculates the accuracy of a prediction.

        Parameters
        ----------
        y_true : pd.Series
            True target.
        y_pred : pd.Series
            Predicted target.

        Returns
        -------
        float
            Accuracy of a prediction.
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
            y_train = fold['train'].iloc[:, -1]
            X_test = fold['test'].iloc[:, :-1]
            y_test = fold['test'].iloc[:, -1]

            self.fit(X_train=X_train, y_train=y_train)
            y_pred = self.predict(X_test=X_test, method=method)
            
            accuracy = self.accuracy_metric(y_true=y_test, y_pred=y_pred)
            accuracies.append(accuracy)
            mean_accuracy = np.mean(accuracies)
        return accuracies, mean_accuracy

if __name__ == '__main__':
    print("Loading datasets...\n\n")
    from ucimlrepo import fetch_ucirepo

    iris_dataset = fetch_ucirepo(id=53)
    X_iris = iris_dataset.data.features
    y_iris = iris_dataset.data.targets['class']
    iris_df = pd.concat([X_iris, y_iris], axis=1)

    bank_marketing = fetch_ucirepo(id=222)
    X_bank_marketing = bank_marketing.data.features
    y_bank_marketing = bank_marketing.data.targets['y']
    bank_marketing_df = pd.concat([X_bank_marketing, y_bank_marketing], axis=1)
    bank_marketing_df_no_misssing = bank_marketing_df.dropna(subset=['job', 'education'])
    bank_marketing_df_no_misssing = bank_marketing_df_no_misssing.drop(columns=['contact', 'poutcome'])
    X_bank_marketing_df_no_misssing = bank_marketing_df_no_misssing.iloc[:, :-1]
    y_bank_marketing_df_no_misssing = bank_marketing_df_no_misssing['y']

    print("## IRIS DATASET ##\n")
    print("Fitting the models...\n")
    nb = NaiveBayes()

    print("Evaluating the model with NO log-probabilities...\n")
    cv_ev = nb.cross_validation_evaluate(k=5, data=iris_df)
    for i, fold_acc in enumerate(cv_ev[0]):
        print(f"Fold {i} - Accuracy: {fold_acc}")
    print(f"Mean Accuracy: {cv_ev[1]}\n\n")

    print("Evaluating the model with log-probabilities...\n")
    cv_ev_log = nb.cross_validation_evaluate(k=5, data=iris_df, method='log')
    for i, fold_acc in enumerate(cv_ev_log[0]):
        print(f"Fold {i} - Accuracy: {fold_acc}")
    print(f"Mean Accuracy: {cv_ev_log[1]}\n\n")

    print("## BANK MARKETING DATASET ##\n")
    print("Fitting the models...\n")
    nb = NaiveBayes()

    print("Evaluating the model with NO log-probabilities...\n")
    cv_ev = nb.cross_validation_evaluate(k=5, data=bank_marketing_df_no_misssing)
    for i, fold_acc in enumerate(cv_ev[0]):
        print(f"Fold {i} - Accuracy: {fold_acc}")
    print(f"Mean Accuracy: {cv_ev[1]}\n\n")

    print("Evaluating the model with log-probabilities...\n")
    cv_ev_log = nb.cross_validation_evaluate(k=5, data=bank_marketing_df_no_misssing, method='log')
    for i, fold_acc in enumerate(cv_ev_log[0]):
        print(f"Fold {i} - Accuracy: {fold_acc}")
    print(f"Mean Accuracy: {cv_ev_log[1]}\n\n")

    # ## Resultados de la evalaución del modelo con el dataset IRIS ##
    # Evaluating the model with NO log-probabilities...

    # Fold 0 - Accuracy: 1.0
    # Fold 1 - Accuracy: 0.9666666666666667
    # Fold 2 - Accuracy: 0.9666666666666667
    # Fold 3 - Accuracy: 0.9333333333333333
    # Fold 4 - Accuracy: 0.9666666666666667
    # Mean Accuracy: 0.9666666666666668

    # Estos resultados indican que el modelo tiene un buen desempeño en términos de Accuracy (Predicciones correctas / Total de predicciones)   
    # en la clasificación de las flores del dataset Iris. (Setosa, Versicolor, Virginica).
    # Esto quiere decir que el dataset contiene información suficiente para que el modelo pueda aprender y generalizar correctamente
    # a partir del cálculo de las probabilidades condicionales de las features.


    # Evaluating the model with log-probabilities...

    # Fold 0 - Accuracy: 0.9666666666666667
    # Fold 1 - Accuracy: 0.9666666666666667
    # Fold 2 - Accuracy: 0.8333333333333334
    # Fold 3 - Accuracy: 1.0
    # Fold 4 - Accuracy: 1.0
    # Mean Accuracy: 0.9533333333333334

    # En este caso, se emplea el logaritmo de las probabilidades para evitar problemas de underflow,
    # dado que las probabilidades son valores muy pequeños pudiendo ser cero en algunos casos.
    # El desempeño es similar al modelo sin logaritmos


    # ## Resultados de la evalaución del modelo con el dataset BANK MARKETING ##
    # Evaluating the model with NO log-probabilities...

    # Fold 0 - Accuracy: 0.8901493228382915
    # Fold 1 - Accuracy: 0.8752170390091446
    # Fold 2 - Accuracy: 0.8878342400740826
    # Fold 3 - Accuracy: 0.8797175272053717
    # Fold 4 - Accuracy: 0.878212549201204
    # Mean Accuracy: 0.8822261356656188

    # En este caso, el modelo tiene un desempeño aceptable en términos de Accuracy 
    # en la clasificación de los clientes (Sí se suscribe a un depósito, No se suscribe a un depósito)
    # Dado que el dataset contiene más complejidad en términos de la cantidad de features y la cantidad de datos,
    # el modelo tiene un desempeño menos preciso que en el caso del dataset Iris.


    # Evaluating the model with log-probabilities...

    # Fold 0 - Accuracy: 0.8845931242041903
    # Fold 1 - Accuracy: 0.8789211714318786
    # Fold 2 - Accuracy: 0.8845931242041903
    # Fold 3 - Accuracy: 0.8790229219726788
    # Fold 4 - Accuracy: 0.8851586015281315
    # Mean Accuracy: 0.8824577886682141

    # En este caso, se emplea el logaritmo de las probabilidades para evitar problemas de underflow,
    # El desempeño es similar al modelo sin logaritmos.