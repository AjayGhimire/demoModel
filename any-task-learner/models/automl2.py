import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from skopt import BayesSearchCV


class AutoML2_Supervised(tf.keras.Model):
    def __init__(self, input_dim, output_dim):
        super(AutoML2_Supervised, self).__init__()

        # Preprocessing components
        self.scaler = StandardScaler()

        # Supervised learning components
        self.fc1 = tf.keras.layers.Dense(256, activation='relu')
        self.fc2 = tf.keras.layers.Dense(128, activation='relu')
        self.output_layer = tf.keras.layers.Dense(output_dim, activation='softmax')

        # Auto-tuning with RandomForestClassifier
        self.model = RandomForestClassifier()

    def preprocess(self, X, y):
        # Standardize and split data
        X_scaled = self.scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
        return X_train, X_test, y_train, y_test

    def auto_tune(self, X_train, y_train, param_distributions):
        # Auto-tune the model
        search = BayesSearchCV(self.model, param_distributions, n_iter=50, cv=3)
        search.fit(X_train, y_train)
        self.model = search.best_estimator_

    def call(self, inputs):
        # Preprocess inputs
        X_train, X_test, y_train, y_test = self.preprocess(inputs)

        # Train the model
        self.auto_tune(X_train, y_train, param_distributions={'n_estimators': [50, 100, 200]})

        # Neural network forward pass
        x = self.fc1(inputs)
        x = self.fc2(x)
        return self.output_layer(x)
