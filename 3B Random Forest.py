import numpy as np
import pandas as pd
from collections import Counter
from multiprocessing import Pool
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split, KFold
import time
import os
from PIL import Image
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


class TreeNode:
    def __init__(self, feature_index=None, threshold=None, left_branch=None, right_branch=None, *, leaf_value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left_branch = left_branch
        self.right_branch = right_branch
        self.leaf_value = leaf_value

    def is_leaf(self):
        return self.leaf_value is not None


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=None, num_features=None):
        print(f"[DecisionTree Init] Initializing DecisionTree with max_depth={max_depth}, min_samples_split={min_samples_split}, num_features={num_features}")
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.num_features = num_features
        self.root = None

    def fit(self, X, y):
        print("[DecisionTree Fit] Starting to build the tree...")
        self.num_features = X.shape[1] if self.num_features is None else min(X.shape[1], self.num_features)
        self.root = self._build_tree(X, y)
        print("[DecisionTree Fit] Tree building completed.")

    def _build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        unique_labels = np.unique(y)
        print(f"[Tree Build] Depth: {depth}, Samples: {num_samples}, Unique Labels: {unique_labels}")

        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or len(unique_labels) == 1 or num_samples < self.min_samples_split:
            leaf_value = self._majority_vote(y)
            print(f"[Tree Build] Creating leaf node with value: {leaf_value}")
            return TreeNode(leaf_value=leaf_value)

        # Select random subset of features
        feature_indices = np.random.choice(num_features, self.num_features, replace=False)
        best_feature, best_threshold = self._get_best_split(X, y, feature_indices)

        if best_feature is None:
            leaf_value = self._majority_vote(y)
            print(f"[Tree Build] No valid split found. Creating leaf node with value: {leaf_value}")
            return TreeNode(leaf_value=leaf_value)

        # Partition the data
        left_indices, right_indices = self._partition(X[:, best_feature], best_threshold)
        print(f"[Tree Build] Best Split: Feature {best_feature}, Threshold {best_threshold}, Left samples {len(left_indices)}, Right samples {len(right_indices)}")

        # Check for empty splits and create leaf nodes if necessary
        if len(left_indices) == 0 or len(right_indices) == 0:
            leaf_value = self._majority_vote(y)
            print(f"[Tree Build] Split resulted in empty child. Creating leaf node with value: {leaf_value}")
            return TreeNode(leaf_value=leaf_value)

        # Recursively build left and right branches
        left_branch = self._build_tree(X[left_indices, :], y[left_indices], depth + 1)
        right_branch = self._build_tree(X[right_indices, :], y[right_indices], depth + 1)
        return TreeNode(feature_index=best_feature, threshold=best_threshold, left_branch=left_branch, right_branch=right_branch)

    def _get_best_split(self, X, y, feature_indices):
        best_gain = -1
        best_feature, best_threshold = None, None
    
        for feature_index in feature_indices:
            feature_column = X[:, feature_index]
            
            # Compute 10 quantiles for the feature column
            quantiles = np.linspace(0.1, 1.0, 10)
            thresholds = np.quantile(feature_column, quantiles)
    
            for threshold in thresholds:
                gain = self._information_gain(y, feature_column, threshold)
    
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_index
                    best_threshold = threshold
    
        return best_feature, best_threshold

    def _information_gain(self, y, feature_column, threshold):
        parent_entropy = self._entropy(y)

        left_indices, right_indices = self._partition(feature_column, threshold)

        if len(left_indices) == 0 or len(right_indices) == 0:
            return 0

        n = len(y)
        n_left, n_right = len(left_indices), len(right_indices)
        left_entropy = self._entropy(y[left_indices])
        right_entropy = self._entropy(y[right_indices])
        child_entropy = (n_left / n) * left_entropy + (n_right / n) * right_entropy

        information_gain = parent_entropy - child_entropy
        return information_gain

    def _partition(self, feature_column, threshold):
        left_indices = np.argwhere(feature_column <= threshold).flatten()
        right_indices = np.argwhere(feature_column > threshold).flatten()
        return left_indices, right_indices

    def _entropy(self, y):
        histogram = np.bincount(y)
        probabilities = histogram / len(y)
        entropy = -np.sum([p * np.log(p) for p in probabilities if p > 0])
        return entropy

    def _majority_vote(self, y):
        if len(y) == 0:
            print("[Majority Vote] Empty y encountered. Returning default class 0.")
            return 0  # Default class; adjust as needed
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def predict(self, X):
        print(f"[DecisionTree Predict] Making predictions on data of shape: {X.shape}")
        predictions = np.array([self._classify_sample(x, self.root) for x in X])
        print("[DecisionTree Predict] Predictions completed.")
        return predictions

    def _classify_sample(self, x, node):
        if node.is_leaf():
            return node.leaf_value

        if x[node.feature_index] <= node.threshold:
            return self._classify_sample(x, node.left_branch)
        return self._classify_sample(x, node.right_branch)

    def evaluate(self, X, y_true):
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y_true)
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        return accuracy, precision, recall, f1

    def count_nodes(self, node=None):
        node = node or self.root
        if node is None or node.is_leaf():
            return 1
        return 1 + self.count_nodes(node.left_branch) + self.count_nodes(node.right_branch)

# Define RandomForestClassifier class
class RandomForestClassifier:
    def __init__(self, num_trees=10, max_depth=None, min_samples_split=2, num_features=None):
        print(f"[RandomForest Init] Initializing RandomForest with num_trees={num_trees}, max_depth={max_depth}, min_samples_split={min_samples_split}, num_features={num_features}")
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.num_features = num_features
        self.trees = []

    def fit(self, X, y):
        print("[RandomForest Fit] Starting training of RandomForest...")
        self.trees = []
        args = [(X, y, self.max_depth, self.min_samples_split, self.num_features) for _ in range(self.num_trees)]
        with Pool(processes=min(10, self.num_trees)) as pool:
            self.trees = pool.starmap(self._train_tree, args)
        print("[RandomForest Fit] Finished training all trees.")

    @staticmethod
    def _train_tree(X, y, max_depth, min_samples_split, num_features):
        tree = DecisionTree(max_depth=max_depth, min_samples_split=min_samples_split, num_features=num_features)
        X_sample, y_sample = RandomForestClassifier.bootstrap_samples(X, y)
        tree.fit(X_sample, y_sample)
        return tree

    @staticmethod
    def bootstrap_samples(X, y):
        sample_count = X.shape[0]
        indexes = np.random.choice(sample_count, sample_count, replace=True)
        return X[indexes], y[indexes]

    def most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def predict(self, X):
        print(f"[RandomForest Predict] Making predictions on data of shape: {X.shape}")
        # Collect predictions from all trees
        predictions = np.array([tree.predict(X) for tree in self.trees])
        # Transpose to get predictions for each sample across all trees
        tree_prediction_mat = np.swapaxes(predictions, 0, 1)
        # Majority vote
        majority_votes = np.array([self.most_common_label(predict) for predict in tree_prediction_mat])
        print("[RandomForest Predict] Predictions completed.")
        return majority_votes

# Define data loading function
def load_images_from_folder(folder_path, image_size=(28, 28)):
    print(f"[Data Loading] Loading images from {folder_path}...")
    data, labels = [], []
    label_encoder = LabelEncoder()
    try:
        folder_names = sorted(os.listdir(folder_path))
    except FileNotFoundError:
        print(f"[Data Loading] Folder path '{folder_path}' not found.")
        return np.array([]), np.array([]), label_encoder
    folder_names = [fn for fn in folder_names if os.path.isdir(os.path.join(folder_path, fn))]
    label_encoder.fit(folder_names)
    print(f"[Data Loading] Found {len(folder_names)} folders (classes).")

    for label_name in folder_names:
        label_folder = os.path.join(folder_path, label_name)
        print(f"[Data Loading] Processing class '{label_name}'...")
        for filename in os.listdir(label_folder):
            img_path = os.path.join(label_folder, filename)
            if not img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            try:
                img = Image.open(img_path).convert('L')  # Convert to grayscale
                img = img.resize(image_size)
                img_array = np.array(img).flatten() / 255.0  # Normalize
                data.append(img_array)
                labels.append(label_name)
            except Exception as e:
                print(f"[Data Loading] Error loading image {img_path}: {e}")

    if len(data) == 0:
        print("[Data Loading] No images loaded. Please check the dataset directory.")
        return np.array([]), np.array([]), label_encoder

    data = np.array(data)
    labels = label_encoder.transform(labels)
    print(f"[Data Loading] Loaded {len(data)} images from {len(folder_names)} classes.")
    return data, labels, label_encoder

# Define model evaluation function
def evaluate_model(y_true, y_pred, label_encoder):
    print("[Evaluation] Evaluating the model...")
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    print(f"[Evaluation] Model evaluation complete. Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    unique_labels = np.unique(y_true)
    target_names = [label_encoder.classes_[i] for i in unique_labels]
    print(classification_report(y_true, y_pred, target_names=target_names))

# Main function
def main():
    '''
    start_time = time.time()
    folder_path = '/Users/suryansh/Downloads/archive 2/new_Sampled_znewtrain'  # Update the path to your dataset
    image_size = (28, 28)
    test_size = 0.2

    print("[Pipeline] Starting the Multiclass Random Forest Pipeline...")
    X, y, label_encoder = load_images_from_folder(folder_path, image_size)
    if X.size == 0:
        print("[Pipeline] No data to train on. Exiting.")
        return

    print("[Pipeline] Scaling the data...")
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    print(f"[Pipeline] Data shape after scaling: {X.shape}")

    print("[Pipeline] Checking class distribution...")
    unique, counts = np.unique(y, return_counts=True)
    class_distribution = dict(zip(label_encoder.classes_, counts))
    print("[Pipeline] Class distribution:")
    for cls, count in class_distribution.items():
        print(f"  Class '{cls}': {count} samples")

    print("[Pipeline] Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    print(f"[Pipeline] Training set size: {X_train.shape[0]}, Testing set size: {X_test.shape[0]}")

    # Define hyperparameter grid
    print("[Pipeline] Starting grid search for hyperparameters...")
    num_trees_values = [10, 20]
    max_depth_values = [None, 10, 20]
    min_samples_split_values = [2, 5]
    num_features_values = [None, 'sqrt', 'log2']  # 'sqrt' and 'log2' are typical choices

    best_accuracy = 0
    best_params = {}

    # Grid search over hyperparameters
    for num_trees in num_trees_values:
        for max_depth in max_depth_values:
            for min_samples_split in min_samples_split_values:
                for num_features in num_features_values:
                    # Determine number of features to consider at each split
                    if isinstance(num_features, int):
                        nf = num_features
                    elif num_features == 'sqrt':
                        nf = int(np.sqrt(X_train.shape[1]))
                    elif num_features == 'log2':
                        nf = int(np.log2(X_train.shape[1]))
                    else:
                        nf = None  # Consider all features

                    print(f"[Grid Search] Training with num_trees={num_trees}, max_depth={max_depth}, min_samples_split={min_samples_split}, num_features={nf}...")
                    rf = RandomForestClassifier(
                        num_trees=num_trees,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        num_features=nf
                    )
                    rf.fit(X_train, y_train)
                    y_pred = rf.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    print(f"[Grid Search] Accuracy: {accuracy:.4f}")
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_params = {
                            'num_trees': num_trees,
                            'max_depth': max_depth,
                            'min_samples_split': min_samples_split,
                            'num_features': nf
                        }
                        print(f"[Grid Search] New best parameters found: {best_params}, Accuracy: {best_accuracy:.4f}")

    if best_params:
        print(f"[Grid Search] Best parameters: {best_params}, Best Accuracy: {best_accuracy:.4f}")
    else:
        print("[Grid Search] No improvement found. Using default parameters.")

    # Train final model with best hyperparameters
    print("[Pipeline] Training final model with best hyperparameters...")
    rf_final = RandomForestClassifier(
        num_trees=best_params.get('num_trees', 10),
        max_depth=best_params.get('max_depth', None),
        min_samples_split=best_params.get('min_samples_split', 2),
        num_features=best_params.get('num_features', None)
    )
    rf_final.fit(X_train, y_train)
    print("[Pipeline] Making predictions on the test set...")
    y_pred_final = rf_final.predict(X_test)
    evaluate_model(y_test, y_pred_final, label_encoder)
    print("[Pipeline] Pipeline complete.")
    print(f"Total time taken: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
'''