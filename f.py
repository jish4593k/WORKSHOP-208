

from google.colab import drive
import time
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import tensorflow as tf  # Added Tensorflow import
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd

class KMeansClustering:
    def __init__(self, img_path):
        drive.mount('/content/drive')
        self.img_path = img_path

    def normalize_data(self, data):
        return data / 255.0

    def euclidean_distance(self, a, b):
        return np.linalg.norm(a - b)

    def kmeans(self, data, k, centroids, max_iterations=50):
        data_size, features = data.shape
        cluster_assignments = np.zeros(data_size, dtype=int)
        centroids = np.array(centroids)

        for iteration in range(max_iterations):
            for i in range(data_size):
                distances = [self.euclidean_distance(data[i], centroid) for centroid in centroids]
                cluster_assignments[i] = np.argmin(distances)

            new_centroids = [np.mean(data[cluster_assignments == j], axis=0) for j in range(k)]
            if np.allclose(centroids, new_centroids, rtol=1e-4, atol=1e-4):  # Check for convergence
                break

            centroids = new_centroids

        return cluster_assignments, centroids

    def recolor_clusters(self, img, assignments):
        colors = {
            0: (60, 179, 113),
            1: (0, 191, 255),
            2: (255, 255, 0),
            3: (255, 0, 0),
            4: (0, 0, 0),
            5: (169, 169, 169),
            6: (255, 140, 0),
            7: (128, 0, 128),
            8: (255, 192, 203),
            9: (255, 255, 255)
        }

        colored_img = np.zeros_like(img)
        for i in range(10):
            mask = (assignments == i).reshape(img.shape[:-1])  # Reshape mask to match image dimensions
            colored_img[mask] = colors[i]

        return colored_img

    def calculate_sse(self, data, assignments, centroids):
        sse = np.sum([self.euclidean_distance(data[i], centroids[assignments[i]]) ** 2
                      for i in range(len(assignments))])
        return sse

    def run_kmeans_for_k_values(self, k_values):
        img = io.imread(self.img_path)
        io.imshow(img)
        plt.show()

        normalized_img = self.normalize_data(img.reshape((-1, 3)))

        for k in k_values:
            starting_centroids = [(i / 10.0, i / 10.0, i / 10.0) for i in range(k)]
            assignments, final_centroids = self.kmeans(normalized_img, k, starting_centroids)

            sse = self.calculate_sse(normalized_img, assignments, final_centroids)

            colored_img = self.recolor_clusters(img, assignments)

            io.imsave(f'clustered_image_k{k}.png', colored_img)

            print(f'For k={k}, Final SSE: {sse}')

            io.imshow(colored_img)
            plt.show()

            time.sleep(1)

def tensor_operations_example():
    # Example tensor operations with TensorFlow
    tensor_example = tf.constant([[1, 2], [3, 4]])
    tensor_squared = tf.square(tensor_example)
    print("Tensor Squared:")
    print(tensor_squared)

def turtle_graphics_example():
    # Example turtle graphics
    turtle.forward(100)
    turtle.right(90)
    turtle.forward(100)
    turtle.done()

def gui_example():
    # Example GUI using tkinter
    root = Tk()
    label = Label(root, text="Hello, GUI!")
    button = Button(root, text="Click me")
    label.pack()
    button.pack()
    root.mainloop()

def sklearn_example():
    # Example machine learning with sklearn
    iris = datasets.load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print("Sklearn Model Accuracy:", accuracy)

def keras_example():
    # Example deep learning with Keras
    model = Sequential()
    model.add(Dense(units=64, activation='relu', input_dim=8))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Dummy data for illustration
    X_train = np.random.rand(100, 8)
    y_train = np.random.randint(2, size=(100, 1))

    model.fit(X_train, y_train, epochs=10, batch_size=32)

def data_mining_example():
    # Example data mining with pandas
    data = {'Name': ['John', 'Alice', 'Bob'],
            'Age': [25, 30, 22],
            'Salary': [50000, 60000, 55000]}

    df = pd.DataFrame(data)
    print("Data Mining Example DataFrame:")
    print(df)

def processing_example():
    # Example data processing
    data = np.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]])

    processed_data = np.mean(data, axis=1)
    print("Processed Data:")
    print(processed_data)

def main():
    img_path = '/content/drive/MyDrive/Homework3/image.png'
    k_values = [2, 3, 6, 10]

    kmeans_clustering = KMeansClustering(img_path)
    kmeans_clustering.run_kmeans_for_k_values(k_values)

    # Examples
    tensor_operations_example()
    turtle_graphics_example()
    gui_example()
    sklearn_example()
    keras_example()
    data_mining_example()
    processing_example()

if __name__ == "__main__":
    main
