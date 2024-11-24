import numpy as np
import cvxopt
import cvxopt.solvers
import os
from PIL import Image
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def rbf_kernel_matrix(X1, X2, sigma=2):
    sq_dists = (
        np.sum(X1**2, axis=1).reshape(-1, 1) +
        np.sum(X2**2, axis=1) -
        2 * np.dot(X1, X2.T)
    )
    return np.exp(-sq_dists / (2 * sigma ** 2))

class SoftMarginSVMQP:
    def __init__(self, C, kernel='linear', sigma=2):
        self.C = float(C)
        self.kernel_type = kernel
        self.sigma = sigma
        self.support_vectors = None
        self.support_vector_labels = None
        self.alphas = None
        self.b = 0
        self.W = None  # Only used for linear kernel case

    def fit(self, X, y):
        self.X = X
        self.y = y
        n_samples, n_features = X.shape
        print(f"[SVM Fit] Number of samples: {n_samples}, Number of features: {n_features}")

        # Compute the Kernel matrix
        if self.kernel_type == 'linear':
            K = np.dot(X, X.T)
        elif self.kernel_type == 'rbf':
            K = rbf_kernel_matrix(X, X, self.sigma)
        else:
            raise ValueError(f"Unsupported kernel type '{self.kernel_type}'. Use 'linear' or 'rbf'.")

        # Setup the parameters for cvxopt
        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(-np.ones(n_samples))

        # G and h for inequality constraints
        G_std = cvxopt.matrix(-np.eye(n_samples))
        G_slack = cvxopt.matrix(np.eye(n_samples))
        G = cvxopt.matrix(np.vstack((G_std, G_slack)))
        h_std = cvxopt.matrix(np.zeros(n_samples))
        h_slack = cvxopt.matrix(np.ones(n_samples) * self.C)
        h = cvxopt.matrix(np.vstack((h_std, h_slack)))

        # A and b for equality constraints
        A = cvxopt.matrix(y.reshape(1, -1).astype(float))
        b = cvxopt.matrix(0.0)

        # Solve QP problem
        
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        alphas = np.ravel(solution['x'])
        # Support vectors have non zero lagrange multipliers
        threshold = 1e-5
        support_indices = alphas > threshold
        self.alphas = alphas[support_indices]
        self.support_vectors = X[support_indices]
        self.support_vector_labels = y[support_indices]
        

        # Compute b
        if len(self.alphas) > 0:
            K_sv = K[np.ix_(support_indices, support_indices)]
            self.b = np.mean(
                self.support_vector_labels - np.sum(self.alphas * self.support_vector_labels * K_sv, axis=1)
            )
            
        else:
            print("[SVM Fit] No support vectors found. Cannot compute bias term b.")

        # Compute weight vector for linear kernel
        if self.kernel_type == 'linear':
            self.W = np.sum(
                self.alphas[:, None] * self.support_vector_labels[:, None] * self.support_vectors, axis=0
            )
            
        else:
            self.W = None

    def decision_function(self, X):
        if self.kernel_type == 'linear':
            return np.dot(X, self.W) + self.b
        elif self.kernel_type == 'rbf':
            K_pred = rbf_kernel_matrix(X, self.support_vectors, self.sigma)
            return np.dot(K_pred, self.alphas * self.support_vector_labels) + self.b

    def predict(self, X):
        print(f"[SVM Predict] Making predictions on data of shape: {X.shape}")
        decision_values = self.decision_function(X)
        predictions = np.sign(decision_values)
        return predictions

class MulticlassSVM:
    def __init__(self, C, kernel='linear', sigma=2):
        self.C = float(C)
        self.kernel_type = kernel
        self.sigma = sigma
        self.classifiers = {}
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        for cls in self.classes_:
            y_binary = np.where(y == cls, 1, -1)
            svm = SoftMarginSVMQP(C=self.C, kernel=self.kernel_type, sigma=self.sigma)
            svm.fit(X, y_binary)
            self.classifiers[cls] = svm

    def predict(self, X):
        scores = np.zeros((X.shape[0], len(self.classes_)))
        for idx, cls in enumerate(self.classes_):
            svm = self.classifiers[cls]
            decision_function = svm.decision_function(X)
            scores[:, idx] = decision_function
        predictions = self.classes_[np.argmax(scores, axis=1)]
        return predictions

def load_images_from_folder(folder_path, image_size=(28, 28)):
    data, labels = [], []
    label_encoder = LabelEncoder()
    folder_names = sorted(os.listdir(folder_path))
    folder_names = [fn for fn in folder_names if os.path.isdir(os.path.join(folder_path, fn))]
    label_encoder.fit(folder_names)

    for label_name in folder_names:
        label_folder = os.path.join(folder_path, label_name)
        for filename in os.listdir(label_folder):
            img_path = os.path.join(label_folder, filename)
            if not img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            try:
                img = Image.open(img_path).convert('L')
                img = img.resize(image_size)
                img_array = np.array(img).flatten() / 255.0
                data.append(img_array)
                labels.append(label_name)
            except Exception as e:
                print(f"[Data Loading] Error loading image {img_path}: {e}")

    data = np.array(data)
    labels = label_encoder.transform(labels)
    return data, labels, label_encoder

def evaluate_model(y_true, y_pred, label_encoder):
    print("[Evaluation] Evaluating the model...")
    accuracy = accuracy_score(y_true, y_pred)
    print(f"[Evaluation] Model evaluation complete. Accuracy: {accuracy:.4f}")
    print("Classification Report:")

    unique_labels = np.unique(y_true)
    target_names = [label_encoder.classes_[i] for i in unique_labels]
    print(classification_report(y_true, y_pred, target_names=target_names))

def main():
    folder_path = '/Users/suryansh/Downloads/iuwewh/DATASET'  # Update the path to your dataset
    image_size = (28, 28)
    test_size = 0.2
    X, y, label_encoder = load_images_from_folder(folder_path, image_size)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    

    unique, counts = np.unique(y, return_counts=True)
    class_distribution = dict(zip(label_encoder.classes_, counts))
    
    #for cls, count in class_distribution.items():
    #    print(f"  Class '{cls}': {count} samples")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    

    # Grid Search for optimal C and sigma
    print("[Pipeline] Starting grid search for hyperparameters...")
    C_values = [0.1, 1, 10, 100]
    sigma_values = [0.5, 1, 2, 5]
    best_accuracy = 0
    best_C, best_sigma = None, None

    for C in C_values:
        for sigma in sigma_values:
            print(f"[Grid Search] Training with C={C}, sigma={sigma}...")
            svm = MulticlassSVM(C=C, kernel='rbf', sigma=sigma)
            svm.fit(X_train, y_train)
            y_pred = svm.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"[Grid Search] Accuracy with C={C}, sigma={sigma}: {accuracy:.4f}")
            if accuracy > best_accuracy:
                best_accuracy, best_C, best_sigma = accuracy, C, sigma
                print(f"[Grid Search] New best parameters found: C={best_C}, sigma={best_sigma}, Accuracy={best_accuracy:.4f}")

    if best_C is not None and best_sigma is not None:
        print(f"[Grid Search] Best C: {best_C}, Best sigma: {best_sigma}, Best Accuracy: {best_accuracy:.4f}")
    else:
        print("[Grid Search] No improvement found during grid search. Using default parameters C=1.0, sigma=2.")

    # Train final model with best parameters
    print("[Pipeline] Training final model with best hyperparameters...")
    svm_final = MulticlassSVM(C=best_C if best_C else 1.0, kernel='rbf', sigma=best_sigma if best_sigma else 2)
    svm_final.fit(X_train, y_train)
    print("[Pipeline] Making predictions on the test set...")
    y_pred_final = svm_final.predict(X_test)
    evaluate_model(y_test, y_pred_final, label_encoder)
    print("[Pipeline] Pipeline complete.")

if __name__ == "__main__":
    main()