import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

class DataLoader:
    def __init__(self, data_folder, stop_words):
        self.data_folder = data_folder
        self.stop_words = stop_words
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.tfidf_transformer = TfidfTransformer(sublinear_tf=True)

    def load_data(self):
        data_train_path = os.path.join(self.data_folder, 'data_train.npy')
        data_test_path = os.path.join(self.data_folder, 'data_test.npy')
        labels_train_npy_path = os.path.join(self.data_folder, 'labels_train.npy')
        labels_train_csv_path = os.path.join(self.data_folder, 'label_train.csv')

        self.X_train = np.load(data_train_path, allow_pickle=True)
        self.X_test = np.load(data_test_path, allow_pickle=True)

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

    def preprocess_data(self):
        self.remove_stop_words()
        self.X_train = self.tfidf_transformer.fit_transform(self.X_train).toarray()
        self.X_test = self.tfidf_transformer.transform(self.X_test).toarray()
        self.feature_selection()
        return self.X_train, self.y_train, self.X_test

    def remove_stop_words(self):
        stop_word_indices = [index for index in range(self.X_train.shape[1]) if str(index) in self.stop_words]
        self.X_train = np.delete(self.X_train, stop_word_indices, axis=1)
        self.X_test = np.delete(self.X_test, stop_word_indices, axis=1)
        print(f"Stop words removed. Remaining features: {self.X_train.shape[1]}")

    def feature_selection(self):
        N = self.X_train.shape[0]
        DF = np.sum(self.X_train > 0, axis=0)
        DF_threshold_low = 5
        DF_threshold_high = N * 0.9

        selected_features = np.where((DF > DF_threshold_low) & (DF < DF_threshold_high))[0]
        self.X_train = self.X_train[:, selected_features]
        self.X_test = self.X_test[:, selected_features]
        print(f"Feature selection complete. Selected features: {self.X_train.shape[1]}")

def build_voting_classifier():
    rf = RandomForestClassifier(class_weight='balanced', random_state=42)
    gb = GradientBoostingClassifier()
    adb = AdaBoostClassifier(algorithm="SAMME")

    voting_clf = VotingClassifier(
        estimators=[('rf', rf), ('gb', gb), ('adb', adb)],
        voting='soft'
    )

    pipeline = Pipeline([
        ('feature_selection', SelectFromModel(RandomForestClassifier(n_estimators=50, random_state=42))),
        ('scaler', StandardScaler()),
        ('classifier', voting_clf)
    ])

    return pipeline

def find_best_threshold(model, X_val, y_val):
    probabilities = model.predict_proba(X_val)[:, 1]
    thresholds = np.linspace(0.25, 0.45, 41)
    f1_scores = []

    for threshold in thresholds:
        y_pred = (probabilities >= threshold).astype(int)
        f1 = f1_score(y_val, y_pred)
        f1_scores.append(f1)

    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    print(f"Best threshold found: {best_threshold}")
    print(f"Best F1 Score on validation set: {f1_scores[best_idx]}")
    return best_threshold

def k_fold_cross_validation(X, y, model, k=10):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    val_scores = []
    thresholds = []

    for train_index, val_index in kf.split(X):
        X_train_k, X_val_k = X[train_index], X[val_index]
        y_train_k, y_val_k = y[train_index], y[val_index]

        model.fit(X_train_k, y_train_k)

        # Find the best threshold on validation data
        best_threshold = find_best_threshold(model, X_val_k, y_val_k)
        thresholds.append(best_threshold)

        # Evaluate on validation data
        probabilities_val = model.predict_proba(X_val_k)[:, 1]
        y_val_pred = (probabilities_val >= best_threshold).astype(int)
        val_f1 = f1_score(y_val_k, y_val_pred)
        val_scores.append(val_f1)

    avg_val_score = np.mean(val_scores)
    avg_threshold = np.mean(thresholds)
    print(f"Average Validation F1 Score: {avg_val_score}")
    print(f"Average Optimal Threshold: {avg_threshold}")
    return avg_threshold

if __name__ == "__main__":
    stop_words = set([
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

    # Load and preprocess data
    data_loader = DataLoader(data_folder='Kaggle', stop_words=stop_words)
    data_loader.load_data()
    X_train, y_train, X_test = data_loader.preprocess_data()

    # Build model pipeline
    model = build_voting_classifier()

    # Parameter grid with optimized options
    param_grid = {
        'classifier__rf__n_estimators': [100, 200],
        'classifier__rf__max_depth': [10, 20],
        'classifier__rf__min_samples_split': [5, 10],
        'classifier__rf__min_samples_leaf': [2, 4],
        'classifier__gb__n_estimators': [50, 100],
        'classifier__gb__learning_rate': [0.1, 0.2],
        'classifier__gb__max_depth': [3, 5],
        'classifier__adb__n_estimators': [50, 100],
        'classifier__adb__learning_rate': [0.1, 0.5]
    }

    # Using RandomizedSearchCV with n_iter=10 and cv=10
    random_search = RandomizedSearchCV(model, param_grid, scoring='f1', cv=10, verbose=2, n_iter=4, random_state=42)
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_

    print("Model training complete with optimized voting classifier.")

    # Perform K-Fold Cross-Validation to find average optimal threshold with k=10
    avg_threshold = k_fold_cross_validation(X_train, y_train, best_model, k=11)

    # Evaluate on full training set with average threshold
    probabilities_train = best_model.predict_proba(X_train)[:, 1]
    y_train_pred = (probabilities_train >= avg_threshold).astype(int)
    train_f1 = f1_score(y_train, y_train_pred)
    print(f"F1 Score on Training set: {train_f1}")

    # Make predictions on the test set
    probabilities_test = best_model.predict_proba(X_test)[:, 1]
    y_test_pred = (probabilities_test >= avg_threshold).astype(int)

    # Save predictions to CSV
    predictions_df = pd.DataFrame({
        'ID': np.arange(X_test.shape[0]),
        'label': y_test_pred
    })
    predictions_df.to_csv('predictions.csv', index=False)
    print("Predictions saved to 'predictions.csv'.")
