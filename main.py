from functools import reduce

import hydra
import pandas as pd  # type: ignore
from omegaconf import DictConfig
from sklearn.model_selection import StratifiedKFold  # type: ignore
from tqdm import tqdm

from synolytic import Synolytic  # type: ignore


@hydra.main(version_base='1.3', config_path="conf", config_name="config")
def main(cfg : DictConfig) -> None:
    @hydra.main(version_base='1.3', config_path="conf", config_name="config")
    def main(cfg: DictConfig) -> None:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(cfg.data['data_fpath'], sep=cfg.data['separator'])

        # Get the numeric columns (all columns except the target)
        numeric_cols = [col for col in df.columns if col != cfg.data['target']]

        # Create a Synolytic object
        gr = Synolytic(
            classifier_str='svc',
            probability=True,
            random_state=cfg.random_state,
            numeric_cols=numeric_cols,
            category_cols=None
        )

        # Create a StratifiedKFold object
        skf = StratifiedKFold(n_splits=cfg.data['n_splits'], random_state=None)

        # Drop the target column from the DataFrame to get the features
        features_df = df.drop(columns=[cfg.data['target']])

        # Get the target column
        target = df[cfg.data['target']]

        # Create an empty list to store the results
        results_list = []

        print('Building synolytic graph...')

        # Perform k-fold cross-validation
        for train_index, test_index in tqdm(skf.split(features_df, target), total=skf.get_n_splits(), desc="k-fold"):
            Xtrain, Xtest = features_df.loc[train_index], features_df.loc[test_index]
            ytrain, ytest = target.loc[train_index], target.loc[test_index]

            # Fit the Synolytic model on the training data
            gr.fit(X_train=Xtrain, y_train=ytrain)

            # Predict the labels for the test data
            results_list.append(gr.predict(X_test=Xtest))

        # Merge the results into a single DataFrame
        results_df = reduce(lambda x, y: pd.merge(x, y, on=['p1', 'p2'], how='inner'), results_list)

        # Save the results to a CSV file
        results_df.to_csv(cfg.data['results_fpath'], sep=cfg.data['separator'], index=False)

        print('All done!')


if __name__ == "__main__":
    main()