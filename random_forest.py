import numpy as np
from collections import Counter
from DecTree import DecisionTree

class RandomForest:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        criterion: str = "entropy",
        max_features: None | str = "sqrt",
    ) -> None:
        self.n_estimators = n_estimators #n trees in forest
        self.max_depth = max_depth #max depth of tree
        self.criterion = criterion #criterion choice
        self.max_features = max_features #number of features to consider

    def bootstrap(self, X, y):
        samples = X.shape[0] 
        idx = np.random.choice(samples, samples, replace=True) #selects random values w replacement
        return X[idx], y[idx]

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.trees = []

        for _ in range(self.n_estimators):
            X_bootstrap, y_bootstrap = self.bootstrap(X,y)

            tree = DecisionTree(
                max_depth=self.max_depth,
                criterion=self.criterion,
                #max_features=self.max_features
            )
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)

    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = np.array([tree.predict(X) for tree in self.trees])

        final_predictions = []
        for i in range(X.shape[0]):
            votes = predictions[:,i]
            final_predictions.append(Counter(votes).most_common(1)[0][0])
        return np.array(final_predictions)



'---------------------------------------Test-----------------------------------------------------------------'
if __name__ == "__main__":
    # Test the RandomForest class on a synthetic dataset
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    seed = 0

    np.random.seed(seed)

    X, y = make_classification(
        n_samples=100, n_features=10, random_state=seed, n_classes=2
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=seed, shuffle=True
    )

    rf = RandomForest(
        n_estimators=20, max_depth=5, criterion="entropy", max_features="sqrt"
    )
    rf.fit(X_train, y_train)

    print(f"Training accuracy: {accuracy_score(y_train, rf.predict(X_train))}")
    print(f"Validation accuracy: {accuracy_score(y_val, rf.predict(X_val))}")
