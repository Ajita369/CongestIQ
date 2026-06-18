import numpy as np

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature         # index of feature
        self.threshold = threshold     # threshold value
        self.left = left               # left child (<= threshold)
        self.right = right             # right child (> threshold)
        self.value = value             # prediction value (if leaf)

    def is_leaf(self):
        return self.value is not None

class PureDecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2, mode='classifier'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.mode = mode
        self.root = None

    def fit(self, X, y):
        # X: 2D numpy array, y: 1D numpy array
        self.root = self._build_tree(X, y, depth=0)
        return self

    def _build_tree(self, X, y, depth):
        num_samples, num_features = X.shape
        
        # Check stopping criteria
        if (depth >= self.max_depth or 
            num_samples < self.min_samples_split or 
            len(np.unique(y)) == 1):
            
            leaf_value = self._calculate_leaf_value(y)
            return Node(value=leaf_value)

        # Find the best split
        best_feat, best_thresh = self._best_split(X, y, num_samples, num_features)
        
        # If no split improves impurity/variance, create leaf
        if best_feat is None:
            leaf_value = self._calculate_leaf_value(y)
            return Node(value=leaf_value)

        # Split current node
        left_idx = X[:, best_feat] <= best_thresh
        right_idx = X[:, best_feat] > best_thresh
        
        # Build children
        left_child = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right_child = self._build_tree(X[right_idx], y[right_idx], depth + 1)
        
        return Node(feature=best_feat, threshold=best_thresh, left=left_child, right=right_child)

    def _best_split(self, X, y, num_samples, num_features):
        best_gain = -1 if self.mode == 'classifier' else float('inf')
        split_idx, split_thresh = None, None

        # Calculate base impurity/variance
        if self.mode == 'classifier':
            base_impurity = self._gini(y)
        else:
            base_impurity = self._variance(y)

        for feat_idx in range(num_features):
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            
            # If too many thresholds, sample them to speed up training
            if len(thresholds) > 30:
                thresholds = np.percentile(X_column, np.linspace(0, 100, 30))

            for thresh in thresholds:
                left_idx = X_column <= thresh
                right_idx = X_column > thresh
                
                if sum(left_idx) == 0 or sum(right_idx) == 0:
                    continue

                if self.mode == 'classifier':
                    # Calculate information gain (Gini reduction)
                    gain = self._information_gain(y, left_idx, right_idx, base_impurity)
                    if gain > best_gain:
                        best_gain = gain
                        split_idx = feat_idx
                        split_thresh = thresh
                else:
                    # Calculate combined variance (weighted sum of variances)
                    combined_var = self._combined_variance(y, left_idx, right_idx)
                    if combined_var < best_gain:
                        best_gain = combined_var
                        split_idx = feat_idx
                        split_thresh = thresh

        return split_idx, split_thresh

    def _information_gain(self, y, left_idx, right_idx, base_impurity):
        # Weigthed Gini
        n = len(y)
        n_l, n_r = sum(left_idx), sum(right_idx)
        
        gini_l = self._gini(y[left_idx])
        gini_r = self._gini(y[right_idx])
        
        child_gini = (n_l / n) * gini_l + (n_r / n) * gini_r
        return base_impurity - child_gini

    def _combined_variance(self, y, left_idx, right_idx):
        n = len(y)
        n_l, n_r = sum(left_idx), sum(right_idx)
        
        var_l = self._variance(y[left_idx])
        var_r = self._variance(y[right_idx])
        
        return (n_l / n) * var_l + (n_r / n) * var_r

    def _gini(self, y):
        m = len(y)
        if m == 0:
            return 0
        counts = np.bincount(y.astype(int))
        probabilities = counts / m
        return 1.0 - np.sum(probabilities ** 2)

    def _variance(self, y):
        if len(y) == 0:
            return 0
        return np.var(y)

    def _calculate_leaf_value(self, y):
        if len(y) == 0:
            return 0
        if self.mode == 'classifier':
            # Majority vote
            return np.argmax(np.bincount(y.astype(int)))
        else:
            # Mean
            return np.mean(y)

    def predict(self, X):
        # X: 2D numpy array
        return np.array([self._predict_row(self.root, row) for row in X])

    def _predict_row(self, node, x):
        if node.is_leaf():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._predict_row(node.left, x)
        return self._predict_row(node.right, x)

    def predict_proba(self, X):
        # Only for classifier
        assert self.mode == 'classifier', "predict_proba is only available for classifier"
        return np.array([self._predict_proba_row(self.root, row) for row in X])

    def _predict_proba_row(self, node, x):
        # Traverses to leaf, then returns the probability distribution (or binary prob)
        # To simplify, we return the probability of class 1
        # In this implementation, value is the majority class, but we can compute distribution
        if node.is_leaf():
            # Return value as simple probability: if value is 1, return 1.0, else 0.0
            # (or we could store class counts at leaf, but this is sufficient for MVP)
            return float(node.value)
        
        if x[node.feature] <= node.threshold:
            return self._predict_proba_row(node.left, x)
        return self._predict_proba_row(node.right, x)


class PureRandomForest:
    def __init__(self, n_estimators=5, max_depth=10, min_samples_split=2, mode='classifier', sample_subspace=0.8):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.mode = mode
        self.sample_subspace = sample_subspace
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        n_samples = X.shape[0]
        
        for _ in range(self.n_estimators):
            # Bootstrap sampling
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_sample = X[indices]
            y_sample = y[indices]
            
            tree = PureDecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                mode=self.mode
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
        return self

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        # predictions shape: (n_estimators, n_samples)
        if self.mode == 'classifier':
            # Majority vote
            # Transpose to (n_samples, n_estimators)
            predictions = predictions.T
            return np.array([np.argmax(np.bincount(row.astype(int))) for row in predictions])
        else:
            # Mean
            return np.mean(predictions, axis=0)

    def predict_proba(self, X):
        assert self.mode == 'classifier', "predict_proba only for classifier"
        # Average probability from trees
        probs = np.array([tree.predict_proba(X) for tree in self.trees])
        return np.mean(probs, axis=0)
