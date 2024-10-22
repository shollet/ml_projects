import numpy as np
import pandas as pd
import os

# Set random seed for reproducibility
np.random.seed(42)

class DataLoader:
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.stop_words = set([
            'the', 'and', 'is', 'in', 'to', 'of', 'a', 'for', 'on', 'with', 'as', 'by',
            'at', 'from', 'it', 'this', 'that', 'an', 'be', 'are', 'was', 'were', 'or',
            'which', 'but', 'not', 'can', 'has', 'have', 'had', 'will', 'would', 'should',
            'could', 'may', 'might', 'do', 'does', 'did', 'been', 'being', 'if', 'their',
            'they', 'them', 'we', 'our', 'us', 'you', 'your', 'he', 'she', 'him', 'her',
            'his', 'hers', 'its', 'about', 'also', 'up', 'out', 'so', 'what', 'when', 'who',
            'whom', 'how', 'why', 'all', 'any', 'no', 'other', 'some', 'such', 'only', 'new',
            'more', 'most', 'over', 'after', 'before', 'between', 'into', 'than', 'these',
            'those', 'very', 'just', 'like'
        ])
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.vocab_map = None
        self.index_to_word = None
        self.class_weights = None
        self.IDF = None
        self.mean = None
        self.std = None
        self.X_train_tfidf = None
        self.X_test_tfidf = None
        self.train_indices = None
        self.val_indices = None
        self.X_train_split = None
        self.y_train_split = None
        self.X_val_split = None
        self.y_val_split = None

    def load_data(self):
        """Load training and test data along with labels and vocabulary map."""
        # Build file paths
        data_train_path = os.path.join(self.data_folder, 'data_train.npy')
        data_test_path = os.path.join(self.data_folder, 'data_test.npy')
        labels_train_npy_path = os.path.join(self.data_folder, 'labels_train.npy')
        labels_train_csv_path = os.path.join(self.data_folder, 'label_train.csv')
        vocab_map_path = os.path.join(self.data_folder, 'vocab_map.npy')

        # Load training and test data
        self.X_train = np.load(data_train_path, allow_pickle=True)
        self.X_test = np.load(data_test_path, allow_pickle=True)

        # Load labels
        if os.path.exists(labels_train_npy_path):
            self.y_train = np.load(labels_train_npy_path, allow_pickle=True).astype(int)
        elif os.path.exists(labels_train_csv_path):
            labels_df = pd.read_csv(labels_train_csv_path)
            if 'label' in labels_df.columns:
                self.y_train = labels_df['label'].values.astype(int)
            elif 'Label' in labels_df.columns:
                self.y_train = labels_df['Label'].values.astype(int)
            else:
                self.y_train = labels_df.iloc[:, 1].values.astype(int)
        else:
            raise FileNotFoundError("Labels file not found in the data directory.")

        # Load vocabulary map
        self.vocab_map = self.load_vocab_map(vocab_map_path)

        # Create index_to_word mapping
        self.index_to_word = self.create_index_to_word_mapping(self.vocab_map)

        print("Data loading complete.")
        print(f"X_train shape: {self.X_train.shape}")
        print(f"X_test shape: {self.X_test.shape}")
        print(f"y_train shape: {self.y_train.shape}")
        print(f"Unique labels in y_train: {np.unique(self.y_train)}")

    def load_vocab_map(self, vocab_map_path):
        """Load the vocabulary map from file."""
        vocab_data = np.load(vocab_map_path, allow_pickle=True)

        # Handle different possible structures of vocab_data
        if isinstance(vocab_data, dict):
            vocab_map = vocab_data
        elif isinstance(vocab_data, np.ndarray):
            if vocab_data.dtype == object:
                if isinstance(vocab_data[0], (tuple, list)):
                    vocab_map = dict(vocab_data)
                elif isinstance(vocab_data[0], str):
                    vocab_map = {idx: word for idx, word in enumerate(vocab_data)}
                else:
                    raise ValueError("Unhandled data type in vocab_data[0]")
            else:
                raise ValueError("Unhandled data type in vocab_data")
        else:
            raise ValueError("Unexpected data format in vocab_map.npy")

        return vocab_map

    def create_index_to_word_mapping(self, vocab_map):
        """Create a mapping from index to word."""
        if isinstance(next(iter(vocab_map.keys())), int):
            # Keys are indices
            index_to_word = vocab_map
        else:
            # Keys are words; invert the mapping
            index_to_word = {index: word for word, index in vocab_map.items()}
        return index_to_word

    def preprocess_data(self):
        """Perform all data preprocessing steps."""
        self.remove_stop_words()
        self.compute_tfidf()
        self.feature_selection()
        self.standardize_features()
        self.add_bias_term()
        self.compute_class_weights()

    def remove_stop_words(self):
        """Remove stop words from the dataset."""
        # Identify indices of stop words
        stop_word_indices = [index for index, word in self.index_to_word.items() if word.lower() in self.stop_words]

        # Remove stop words from data
        keep_indices = [i for i in range(self.X_train.shape[1]) if i not in stop_word_indices]

        self.X_train = self.X_train[:, keep_indices]
        self.X_test = self.X_test[:, keep_indices]

        # Update index_to_word mapping
        self.index_to_word = {new_idx: self.index_to_word[old_idx] for new_idx, old_idx in enumerate(keep_indices)}

        print(f"Stop words removed. Remaining features: {self.X_train.shape[1]}")

    def compute_tfidf(self):
        """Compute TF-IDF feature representation."""
        N = self.X_train.shape[0]

        # Compute Document Frequency (DF)
        DF = np.sum(self.X_train > 0, axis=0)

        # Compute Inverse Document Frequency (IDF)
        self.IDF = np.log((N + 1) / (DF + 1)) + 1  # Adding 1 to avoid division by zero

        # Compute TF-IDF for training and test data
        self.X_train_tfidf = self.X_train * self.IDF
        self.X_test_tfidf = self.X_test * self.IDF

        print("TF-IDF computation complete.")

    def feature_selection(self):
        """Perform feature selection based on document frequency thresholds."""
        N = self.X_train.shape[0]

        # Compute Document Frequency (DF)
        DF = np.sum(self.X_train > 0, axis=0)

        # Define thresholds
        DF_threshold_low = 5
        DF_threshold_high = N * 0.9

        # Select features within the thresholds
        selected_features = np.where((DF > DF_threshold_low) & (DF < DF_threshold_high))[0]

        # Update training and test data
        self.X_train_tfidf = self.X_train_tfidf[:, selected_features]
        self.X_test_tfidf = self.X_test_tfidf[:, selected_features]

        # Update IDF
        self.IDF = self.IDF[selected_features]

        print(f"Feature selection complete. Selected features: {self.X_train_tfidf.shape[1]}")

    def standardize_features(self):
        """Standardize features using z-score normalization."""
        # Compute mean and std from training data
        self.mean = np.mean(self.X_train_tfidf, axis=0)
        self.std = np.std(self.X_train_tfidf, axis=0) + 1e-8  # Avoid division by zero

        # Standardize training data
        self.X_train_tfidf = (self.X_train_tfidf - self.mean) / self.std

        # Standardize test data using training mean and std
        self.X_test_tfidf = (self.X_test_tfidf - self.mean) / self.std

        print("Feature standardization complete.")

    def add_bias_term(self):
        """Add a bias term to the feature matrices."""
        self.X_train_tfidf = np.hstack((np.ones((self.X_train_tfidf.shape[0], 1)), self.X_train_tfidf))
        self.X_test_tfidf = np.hstack((np.ones((self.X_test_tfidf.shape[0], 1)), self.X_test_tfidf))

        print("Bias term added to feature matrices.")

    def compute_class_weights(self):
        """Compute class weights to handle class imbalance."""
        class_counts = np.bincount(self.y_train)
        total_samples = len(self.y_train)
        self.class_weights = {0: total_samples / class_counts[0], 1: total_samples / class_counts[1]}

        print(f"Class weights computed: {self.class_weights}")

    def split_data(self, validation_size=0.2, shuffle=True):
        """Split data into training and validation sets."""
        m = len(self.y_train)
        indices = np.arange(m)
        if shuffle:
            np.random.shuffle(indices)
        split_point = int(m * (1 - validation_size))
        self.train_indices = indices[:split_point]
        self.val_indices = indices[split_point:]
        self.X_train_split = self.X_train_tfidf[self.train_indices]
        self.y_train_split = self.y_train[self.train_indices]
        self.X_val_split = self.X_train_tfidf[self.val_indices]
        self.y_val_split = self.y_train[self.val_indices]

        print(f"Data split into training ({len(self.train_indices)}) and validation ({len(self.val_indices)}) sets.")

    def get_train_validation_data(self):
        """Return the training and validation data splits."""
        return self.X_train_split, self.y_train_split, self.X_val_split, self.y_val_split

    def get_test_data(self):
        """Return the preprocessed test data."""
        return self.X_test_tfidf

class LogisticRegressionModel:
    def __init__(self, class_weights, learning_rate=0.0005, num_iterations=5000, lambda_=0.005,
                 batch_size=128, early_stopping_rounds=20):
        self.class_weights = class_weights
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.lambda_ = lambda_
        self.batch_size = batch_size
        self.early_stopping_rounds = early_stopping_rounds
        self.theta = None
        self.best_threshold = 0.5
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8

    @staticmethod
    def sigmoid(z):
        """Compute the sigmoid function."""
        return 1 / (1 + np.exp(-z))

    def compute_cost(self, X, y):
        """Compute the regularized cost function."""
        m = len(y)
        h = self.sigmoid(X @ self.theta)
        epsilon = self.epsilon

        # Compute weighted cost
        cost = -(1/m) * (
            self.class_weights[1] * np.dot(y, np.log(h + epsilon)) +
            self.class_weights[0] * np.dot((1 - y), np.log(1 - h + epsilon))
        )
        reg_term = (self.lambda_ / (2 * m)) * np.sum(self.theta[1:] ** 2)
        return cost + reg_term

    def compute_gradient(self, X, y):
        """Compute the gradient of the cost function."""
        m = len(y)
        h = self.sigmoid(X @ self.theta)
        weights = np.vectorize(self.class_weights.get)(y)
        gradient = (1/m) * (X.T @ ((h - y) * weights))
        gradient[1:] += (self.lambda_ / m) * self.theta[1:]
        return gradient

    def train(self, X_train, y_train, X_val, y_val):
        """Train the logistic regression model using Adam optimizer."""
        m, n = X_train.shape
        self.theta = np.zeros(n)
        m_t = np.zeros(n)
        v_t = np.zeros(n)

        cost_history = []
        val_cost_history = []
        best_val_cost = float('inf')
        best_theta = None
        no_improvement_counter = 0

        print("Starting training...")

        for i in range(1, self.num_iterations + 1):
            permutation = np.random.permutation(m)
            X_shuffled = X_train[permutation]
            y_shuffled = y_train[permutation]

            for j in range(0, m, self.batch_size):
                X_batch = X_shuffled[j:j + self.batch_size]
                y_batch = y_shuffled[j:j + self.batch_size]

                gradient = self.compute_gradient(X_batch, y_batch)

                # Update biased first and second moment estimates
                m_t = self.beta1 * m_t + (1 - self.beta1) * gradient
                v_t = self.beta2 * v_t + (1 - self.beta2) * (gradient ** 2)

                # Compute bias-corrected estimates
                m_hat = m_t / (1 - self.beta1 ** i)
                v_hat = v_t / (1 - self.beta2 ** i)

                # Update parameters
                self.theta -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

            # Compute costs
            cost = self.compute_cost(X_train, y_train)
            val_cost = self.compute_cost(X_val, y_val)
            cost_history.append(cost)
            val_cost_history.append(val_cost)

            # Early stopping
            if val_cost < best_val_cost - 1e-5:
                best_val_cost = val_cost
                best_theta = self.theta.copy()
                no_improvement_counter = 0
            else:
                no_improvement_counter += 1

            if no_improvement_counter >= self.early_stopping_rounds:
                print(f"Early stopping at iteration {i}")
                self.theta = best_theta
                break

            if i % 500 == 0 or i == self.num_iterations:
                print(f"Iteration {i}: Training cost = {cost}, Validation cost = {val_cost}")

        print("Training complete.")

    def find_best_threshold(self, X_val, y_val):
        """Find the optimal classification threshold based on validation set."""
        probabilities = self.sigmoid(X_val @ self.theta)
        thresholds = np.linspace(0.1, 0.9, 81)
        f1_scores = []

        for threshold in thresholds:
            y_pred = (probabilities >= threshold).astype(int)
            f1 = self.f1_score(y_val, y_pred)
            f1_scores.append(f1)

        best_idx = np.argmax(f1_scores)
        self.best_threshold = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]

        print(f"Best threshold found: {self.best_threshold}")
        print(f"Best F1 Score on validation set: {best_f1}")

    @staticmethod
    def f1_score(y_true, y_pred):
        """Compute the F1 score manually."""
        TP = np.sum((y_true == 1) & (y_pred == 1))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))

        precision = TP / (TP + FP + 1e-8)
        recall = TP / (TP + FN + 1e-8)

        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        return f1

    def evaluate(self, X, y, data_split='Validation'):
        """Evaluate the model on a given dataset."""
        probabilities = self.sigmoid(X @ self.theta)
        y_pred = (probabilities >= self.best_threshold).astype(int)
        f1 = self.f1_score(y, y_pred)
        print(f"F1 Score on {data_split} set: {f1}")
        return f1

    def predict(self, X):
        """Make predictions on new data."""
        probabilities = self.sigmoid(X @ self.theta)
        return (probabilities >= self.best_threshold).astype(int)

if __name__ == "__main__":
    # Initialize data loader
    data_loader = DataLoader(data_folder='classer-le-text')

    # Load and preprocess data
    data_loader.load_data()
    data_loader.preprocess_data()
    data_loader.split_data(validation_size=0.2)

    # Get data splits
    X_train, y_train, X_val, y_val = data_loader.get_train_validation_data()
    X_test = data_loader.get_test_data()

    # Initialize and train the model
    model = LogisticRegressionModel(class_weights=data_loader.class_weights)
    model.train(X_train, y_train, X_val, y_val)

    # Find the best threshold
    model.find_best_threshold(X_val, y_val)

    # Evaluate the model
    model.evaluate(X_val, y_val, data_split='Validation')
    model.evaluate(X_train, y_train, data_split='Training')

    # Make predictions on the test set
    y_test_pred = model.predict(X_test)

    # Save predictions to CSV
    test_ids = np.arange(X_test.shape[0])
    predictions_df = pd.DataFrame({
        'ID': test_ids,
        'label': y_test_pred
    })
    predictions_df.to_csv('predictions.csv', index=False)

    print("Predictions saved to 'predictions.csv'.")
