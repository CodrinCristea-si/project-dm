import math

import numpy as np
import random


class Node:
    """
    Component of the decision tree representing a node: each node can contain another node as an attribute,
    except when the decision tree has reached the terminal node

    Parameters:

    predicted_class: int
        the predicted class is specified by taking the mode of the classes in the node during training.

    Attributes:

    feature_index: int
        the index of the feature of the fitted data where the split will occur for the node
    threshold: float
        the value split ('less than' and 'more than') for the chosen feature
    left: <class Node>
        the left child Node that will be grown that fulfils the condition 'less than' threshold
    right: <class Node>
        the right child Node that will be grown that fulfils the condition 'more than' threshold
    """

    def __init__(self, predicted_class):
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None


class DecisionTree:
    """
    A decision tree classifier

    Parameters

    max_depth: int, default=None
        the maximum depth of the tree. If None, then nodes are expanded until all leaves are the purest
    max_features: int, float, default=None
        at each split from parent node to child nodes only the max_features are considered to find the threshold split
        if it is float and <1: max_features take the proportion of the features in the dataset.
        if it is None: max_features take the sqrt of the n_features
    random_state: int, default=None
        controls the randomness of the estimator: the features are always randomly permuted at each split

    Attributes:

    tree: <class Node>
        The root node which obtains all other sub-nodes which are recursively stored as attributes
    """
    def __init__(self, max_depth=None, max_features=None, random_state=None):
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state
        self.tree = None
        # will be initialized during training
        self.n_features = None
        self.n_classes = None

    def fit(self, X, y):
        """
        trains a decision tree on X and y

        Parameters:

        X: np.array
            the set of feature variables of the training dataset

        y: np.array
            the target variable of the training dataset

        Returns:
        None
        """

        self.n_classes = len(set(y))
        self.n_features = X.shape[1]

        if self.max_features is None:
            self.max_features = int(math.sqrt(self.n_features))

        if isinstance(self.max_features, float) and self.max_features <= 1:
            self.max_features = int(self.max_features * self.n_features)

        # create tree for the dataset
        self.tree = self.grow_tree(X, y, self.random_state)

    def predict(self, X):
        """
        Predict the class for each test example in a test set

        Parameters:

        X: np.array
            The set of feature variables of the test dataset

        Returns:

        predicted_classes: np.array
            The numpy array of predict class for each test example
        """
        predicted_classes = np.array([self.predict_example(inputs) for inputs in X])

        return predicted_classes

    def best_split(self, X, y, random_state):
        """
        Finds the optimal feature index and threshold value for the split at the parent node, which are then used to decide the split of
        training examples of features/targets into smaller subsets.

        Parameters:

        X: np.array
            Subset of all the training examples of features at the parent node.
        y: np.array
            Subset of all the training examples of targets at the parent node.
        random_state: int, default=None

        Returns:
        best_feat_id: int, None
            The feature index considered for split at parent node.

        best_threshold: float, None
            The threshold value at the feature considered for split at parent node.
        """
        m = len(y)
        if m <= 1:
            return None, None

        num_class_parent = [np.sum(y == c) for c in range(self.n_classes)]
        best_gini = 1.0 - sum((n / m) ** 2 for n in num_class_parent)
        if best_gini == 0:
            return None, None

        best_feat_id, best_threshold = None, None

        random.seed(random_state)
        feat_indices = random.sample(range(self.n_features), self.max_features)

        for feat_id in feat_indices:

            sorted_column = sorted(set(X[:, feat_id]))
            threshold_values = [np.mean([a, b]) for a, b in zip(sorted_column, sorted_column[1:])]

            for threshold in threshold_values:

                left_y = y[X[:, feat_id] < threshold]
                right_y = y[X[:, feat_id] > threshold]

                num_class_left = [np.sum(left_y == c) for c in range(self.n_classes)]
                num_class_right = [np.sum(right_y == c) for c in range(self.n_classes)]

                gini_left = 1.0 - sum((n / len(left_y)) ** 2 for n in num_class_left)
                gini_right = 1.0 - sum((n / len(right_y)) ** 2 for n in num_class_right)

                gini = (len(left_y) / m) * gini_left + (len(right_y) / m) * gini_right

                if gini < best_gini:
                    best_gini = gini
                    best_feat_id = feat_id
                    best_threshold = threshold

        return best_feat_id, best_threshold

    def grow_tree(self, X, y, random_state, depth=0):
        """
        Recursive function to continuously generate nodes. At each recursion step, a parent node is formed and recursively split
        into left child node and right child node IF the maximum depth is not reached or the parent node is less pure than
        the average gini of child nodes.

        Parameters:

        X: np.array
            Subset of all the training examples of features at the parent node.
        y: np.array
            Subset of all the training examples of targets at the parent node.
        random_state: int, default=None
        depth: int
            The number of times a branch has split.

        Returns:

        node: <class Node>
            The instantiated Node, with its corresponding attributes.
        """
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes)]
        predicted_class = np.argmax(num_samples_per_class)

        node = Node(predicted_class=predicted_class)

        if (self.max_depth is None) or (depth < self.max_depth):
            id, thr = self.best_split(X, y, random_state)

            if id is not None:
                if random_state is not None:
                    random_state += 1

                indices_left = X[:, id] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]

                node.feature_index = id
                node.threshold = thr
                node.left = self.grow_tree(X_left, y_left, random_state, depth + 1)
                node.right = self.grow_tree(X_right, y_right, random_state, depth + 1)

        return node

    def predict_example(self, inputs):
        """
        Generate the predicted class of a single row of test example based on the feature indices and thresholds that have been stored
        in all the nodes.

        Parameters:

        inputs: An row of test examples containing the all the features that have been trained on.

        Returns:

        node.predicted_class: int
            The stored attribute - predicted_class - of the terminal node.
        """
        node = self.tree

        while node.left:
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right

        return node.predicted_class

