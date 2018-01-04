# import distance, iris data, split stuff, accuracy metric
from scipy.spatial import distance
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# define simple knn
class SimpleKNN():
    def fit(self, features_train, labels_train):
        self.features = features_train
        self.labels = labels_train

    def predict(self, features):
        predictions = []
        for row in features:
            label = self.closest(row)
            predictions.append(label)
        return predictions

    def closest(self, row):
        best_dist = distance.euclidean(row, self.features[0])
        best_index = 0
        for i in range(1, len(self.features)):
            dist = distance.euclidean(row, self.features[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i
        return self.labels[best_index]

# split data into training and testing
iris = datasets.load_iris()
features = iris.data
labels = iris.target
features_train, features_test, labels_train, labels_test = \
                    train_test_split(features, labels, test_size = 0.5)

# magic
classifier = SimpleKNN()
classifier.fit(features_train, labels_train)
predictions = classifier.predict(features_test)

print(accuracy_score(labels_test, predictions))
