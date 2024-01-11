from itertools import combinations
from typing import Optional

import pandas as pd  # type: ignore
from joblib import Parallel, delayed  # type: ignore
from sklearn.base import ClassifierMixin  # type: ignore
from sklearn.compose import ColumnTransformer  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # type: ignore
from sklearn.svm import SVC  # type: ignore


class Synolytic:

    def __init__(self,
                 classifier_str: str,
                 probability: bool = False,
                 random_state: Optional[int] = None,
                 numeric_cols: Optional[list] = None,
                 category_cols: Optional[list] = None
                 ):
        self.classifier_str = classifier_str
        self.probability = probability
        self.random_state = random_state
        self.numeric_cols = numeric_cols
        self.category_cols = category_cols
        self.nodes_tpl_list: Optional[list] = None
        self.preprocessor: Optional[ColumnTransformer] = None
        self.clf_dict: Optional[dict] = None
        self.predicts: Optional[list] = None
        self.graph_df: Optional[pd.DataFrame] = None

    def fit(self,
            X_train: pd.DataFrame,
            y_train: pd.Series) -> None:
        """
        Preprocesses the data, fits classifiers, and creates a graph.

        Parameters:
        X_train (pd.DataFrame): The input features.
        y_train (pd.Series): The target variable.

        Returns:
        None
        """
        # Preprocess numeric and category data
        transformers_list = []

        if self.numeric_cols is not None:
            transformers_list.append(('num', StandardScaler(), self.numeric_cols))
        if self.category_cols is not None:
            transformers_list.append(('cat', OneHotEncoder(), self.category_cols))

        self.preprocessor = ColumnTransformer(transformers=transformers_list, remainder='passthrough')
        self.preprocessor.fit(X_train)

        X_train_processed = pd.DataFrame(columns=self.preprocessor.get_feature_names_out(),
                                         data=self.preprocessor.transform(X_train))

        # Create pairs of features for all features
        self.nodes_tpl_list = list(combinations(iterable=X_train_processed.columns, r=2))

        # Reset classifier list for each fit call
        self.clf_dict = {}

        def aux_fit_clf(idx: int, df: pd.DataFrame, feature_name_1: str, feature_name_2: str,
                        y_train: pd.Series) -> tuple:
            """
            Fits a classifier to a pair of features.

            Parameters:
            idx (int): The index of the pair of features.
            df (pd.DataFrame): The processed features.
            feature_name_1 (str): The name of the first feature in the pair.
            feature_name_2 (str): The name of the second feature in the pair.
            y_train (pd.Series): The target variable.

            Returns:
            tuple: A tuple containing the index, feature names, and the fitted classifier.
            """
            clf = SVC(probability=self.probability, class_weight='balanced', random_state=self.random_state) \
                if self.classifier_str == 'svc' \
                else LogisticRegression(class_weight='balanced', random_state=self.random_state) \
                if self.classifier_str == 'logreg' else _raise(exception_type=ValueError, msg='Unknown classifier')
            clf.fit(df[[feature_name_1, feature_name_2]], y_train)
            return idx, feature_name_1, feature_name_2, clf

        # Fill tpl_list on all CPU kernels
        tpl_list = Parallel(n_jobs=-1,
                            verbose=0,
                            prefer='processes')(delayed(aux_fit_clf)(idx=idx,
                                                                     df=X_train_processed,
                                                                     feature_name_1=feature_1,
                                                                     feature_name_2=feature_2,
                                                                     y_train=y_train) for
                                                idx, (feature_1, feature_2) in
                                                enumerate(self.nodes_tpl_list))

        self.graph_df = pd.DataFrame(columns=['p1', 'p2'], data=self.nodes_tpl_list)
        self.clf_dict = {idx: [feature_1, feature_2, clf] for idx, feature_1, feature_2, clf in tpl_list}

    def predict(self, X_test: pd.Series) -> pd.DataFrame:
        """
        Predict the output for the given test data.

        Parameters:
            X_test (pd.Series): The test data to make predictions on.

        Returns:
            pd.DataFrame: The predicted output for the test data.
        """
        # Process the test data using the preprocessor
        X_test_processed = pd.DataFrame(index=X_test.index,
                                        columns=self.preprocessor.get_feature_names_out(),  # type: ignore
                                        data=self.preprocessor.transform(X_test))  # type: ignore

        def aux_predict_clf(X_test: pd.DataFrame, clf: ClassifierMixin) -> tuple:
            """
            Auxiliary function to make predictions using a classifier.

            Parameters:
                X_test (pd.DataFrame): The processed test data.
                clf (ClassifierMixin): The classifier to make predictions with.

            Returns:
                tuple: The predicted output as a tuple.
            """
            # Make predictions using the classifier
            return tuple(clf.predict_proba(X_test)[:, 1]) if self.probability \
                else tuple(clf.predict(X_test))

        # Make predictions for each classifier in clf_dict and update the graph dataframe
        self.graph_df.loc[:, X_test_processed.index] = \
            [aux_predict_clf(X_test=X_test_processed[[val[0], val[1]]], clf=val[2])
                for _, val in self.clf_dict.items()]  # type: ignore

        return self.graph_df


def _raise(exception_type, msg):
    raise exception_type(msg)
