import numpy as np
from typing import Self
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from math import sqrt, log2

"____________________________________"
'''Hjelpefunksjoner'''

def count(y: np.ndarray) -> np.ndarray:
    if y.size == 0:
        return np.array([])
    unique_val, counts = np.unique(y, return_counts=True) #teller alle de unike verdiene
    probs = counts /len(y) #gjør om til proporsjoner/prosentandel
    return probs #returnerer
 

def gini_index(y: np.ndarray) -> float:
    if len(y) == 0:
        return 0.0
    probs = count(y)
    gini = 1.0 - np.sum(probs**2)
    return gini
    

def entropy(y: np.ndarray) -> float:
    if len(y) ==0: #håndterer hvis tom
        return 0.0
    probs = count(y)
    entro = -np.sum(probs * np.log2(probs)) #entropi formula
    return entro
    

def most_common(y: np.ndarray) -> int:
    unique_val, counts = np.unique(y, return_counts=True) #akk som i count, får vi de unike ''originale'' verdiene og deres count
    common_id = np.argmax(counts) #finner indexen av den med flest verdier av seg selv
    return unique_val[common_id]


"___________________________________________"
'''Class '''

class Node:
    def __init__(
        self,
        feature: int = 0,
        threshold: float = 0.0,
        left: int | Self | None = None,
        right: int | Self | None = None,
        value: int | None = None,
    ) -> None:
        self.feature = feature 
        self.threshold = threshold
        self.left = left #left child node (feature <= threshold)
        self.right = right
        self.value = value #kun for leaf nodes

    def is_leaf(self) -> bool:
        return self.value is not None


class DecisionTree:
    def __init__(
        self,
        max_depth: int | None = None,
        criterion: str = "entropy",
        max_features=None
    ) -> None:
        self.root = None
        self.criterion = criterion
        self.max_depth = max_depth
        self.max_features = None

    def _impurity(self, y: np.ndarray) -> float: #kalkulerer impurity basert på valgt kriterie (entropy eller gini)
        if self.criterion == "entropy":
            return entropy(y)
        elif self.criterion == "gini":
            return gini_index(y)
        else:
            raise ValueError("Criterion must be 'entropy' or 'gini'")

    '''Information Gain'''
    def _information_gain(self, y: np.ndarray, y_left: np.ndarray, y_right: np.ndarray) -> float:
        #impurity = self.gini_index if criterion == "gini" else self.entropy
        
        n = len(y) #antall samples
        n_left, n_right = len(y_left), len(y_right)

        if n_left == 0 or n_right ==0: #hvis en av sidene er tomme så er det 0 information gain
            return 0.0

        parent_impurity = self._impurity(y) #før splitt
        child_impurity = (n_left / n) * self._impurity(y_left) + (n_right / n) * self._impurity(y_right) #regner ut impurity fra begge sider

        return parent_impurity - child_impurity #information gain     


    "Splitter data"
    def split(self, X, y, feature_index, threshold):
        left_mask = X[: , feature_index] < threshold
        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[~left_mask], y[~left_mask]
        return X_left, y_left, X_right, y_right


    '''Best split'''

    def best_split(self, X: np.ndarray, y: np.ndarray):
        best_feature, best_threshold, best_info_gain = None, None, -float('inf')

        n_features = X.shape[1]

        if self.max_features is None:
            feature_indices = range(n_features)
        elif self.max_features == 'sqrt':
            feature_indices = np.random.choice(n_features, int(sqrt(n_features)), replace=False)
        elif self.max_features == "log2":
            feature_indices = np.random.choice(n_features, int(log2(n_features)), replace=False)
        else:
            feature_indices = range(n_features)

        for feature_index in feature_indices:
            possible_thresholds = np.unique(X[:, feature_index])

            for threshold in possible_thresholds:
                X_left, y_left, X_right, y_right = self.split(X, y, feature_index, threshold)
                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                info_gain = self._information_gain(y, y_left, y_right)
                if info_gain > best_info_gain:
                    best_feature, best_threshold, best_info_gain = feature_index, threshold, info_gain

        return best_feature, best_threshold, best_info_gain


    '''Build Tree'''   
    def _build_tree(self, X, y, depth=0, max_depth=None, min_samples_split = 2):
        n_samples = len(y)
        unique_classes = np.unique(y)

        #sjekker om current node burde være leaf node
        if len(unique_classes) == 1 or n_samples < min_samples_split or (self.max_depth and depth >= self.max_depth):
            leaf_value = self.calculate_leaf_value(y)
            return Node(value=leaf_value)
        
        #finner beste split
        best_feature, best_threshold, best_info_gain = self.best_split(X, y)
        if best_info_gain == -float('inf'):
            leaf_value = self.calculate_leaf_value(y)
            return Node(value=leaf_value)
        
        #splitter dataset og bygger høyre og venstre subtrees
        X_left, y_left, X_right, y_right = self.split(X, y, best_feature, best_threshold)
        left_subtree = self._build_tree(X_left, y_left, depth + 1, max_depth, min_samples_split)
        right_subtree = self._build_tree(X_right, y_right, depth + 1, max_depth, min_samples_split)

        return Node(feature=best_feature, threshold=best_threshold, left=left_subtree, right=right_subtree)

    '''Calculate'''
    def calculate_leaf_value(self, y):
        return np.bincount(y.astype(int)).argmax()


    '''Fit/Train'''
    def fit(self,X: np.ndarray,y: np.ndarray,max_depth=None,min_samples_split=2,max_features=None):
        if max_depth is not None:
            self.max_depth = max_depth
        self.max_features = max_features
        self.root = self._build_tree(X, y)


    '''Predict'''
    def _predict_input(self, x, node): 
        if node is None:
            return 0
        if node.value is not None:
            return node.value
        if x[node.feature] < node.threshold:
            return self._predict_input(x, node.left)
        return self._predict_input(x, node.right)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.root is None:
            raise ValueError("The model isn't trained.")
        
        preds = np.array([self._predict_input(x, self.root) for x in X])
        return preds
    

'''Test'''
if __name__ == "__main__":
    # Test the DecisionTree class on a synthetic dataset
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

    # Expect the training accuracy to be 1.0 when max_depth=None
    rf = DecisionTree(max_depth=None, criterion="gini")

    rf.fit(X_train, y_train)

    print(f"Training accuracy: {accuracy_score(y_train, rf.predict(X_train))}")
    print(f"Validation accuracy: {accuracy_score(y_val, rf.predict(X_val))}")