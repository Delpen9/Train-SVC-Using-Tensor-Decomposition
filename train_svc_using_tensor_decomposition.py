import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorly.tenalg as tg
from sklearn import svm
import scipy.sparse.linalg as sla
from sklearn.metrics import classification_report

def tensor_decomposition(tensor, N, mode = 1):
    np.random.seed(1)
    U = np.random.random(size = (N, 10))
    for iteration in range(50):
        for n in range(2):
            W = tg.multi_mode_dot(tensor, U.T, modes = [mode], skip = [N + 1])
            S = tg.tensordot(W, W.T, modes = mode)
            _, U = sla.eigs(S, 10)
    G = tg.multi_mode_dot(tensor, U.T, modes = [mode], skip = [N + 1])
    return G.T

def main():
    ## ------------------
    ## Part 1:
    ## ------------------
    training_labels = []
    training_tensor = np.zeros((128, 128, 840), dtype = float)
    filename = r'Question3A'
    for i in range(1, 21):
        for j in range(42):
            training_labels.append(i)
            object = fr'obj{i}_{j}.jpg'
            filepath = fr'{filename}\{object}'
            image = cv2.imread(filepath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            training_tensor[:, :, (i - 1) * 42 + j] = image

    test_labels = []
    test_tensor = np.zeros((128, 128, 600), dtype = float)
    filename = r'Question3A'
    for i in range(1, 21):
        for j in range(42, 72):
            test_labels.append(i)
            object = fr'obj{i}_{j}.jpg'
            filepath = fr'{filename}\{object}'
            image = cv2.imread(filepath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            test_tensor[:, :, (i - 1) * 30 + (j - 43)] = image

    concatenated_training_samples = np.abs(tensor_decomposition(training_tensor, 128))
    training_labels = np.expand_dims(np.array(training_labels), axis = 1)

    concatenated_test_samples = np.abs(tensor_decomposition(test_tensor, 128))
    test_labels = np.array(test_labels)

    linear = svm.SVC(kernel = 'linear', C = 1, decision_function_shape = 'ovo').fit(concatenated_training_samples, training_labels)

    predictions = linear.predict(concatenated_test_samples)

    difference = np.subtract(predictions, test_labels)
    accuracy = (len(difference) - np.count_nonzero(difference)) / len(difference) * 100
    print(fr'The SVM achieves {accuracy}% accuracy.')

    print('Classification Report')
    print(classification_report(test_labels, predictions, target_names = [fr'class_{i}' for i in range(20)]))
    ## ------------------
    ## Part 1 End\
    ## ------------------

    ## ------------------
    ## Part 2:
    ## ------------------
    training_labels = []
    training_tensor = np.zeros((112, 112, 280), dtype = float)
    filename = r'Question3B'
    for i in range(1, 41):
        for j in range(7):
            training_labels.append(i)
            object = fr'Subj{i}_{j}.jpg'
            filepath = fr'{filename}\{object}'
            image = cv2.imread(filepath)
            image = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), dsize = (112, 112))
            training_tensor[:, :, (i - 1) * 7 + j] = image

    test_labels = []
    test_tensor = np.zeros((112, 112, 120), dtype = float)
    filename = r'Question3B'
    for i in range(1, 41):
        for j in range(7, 10):
            test_labels.append(i)
            object = fr'Subj{i}_{j}.jpg'
            filepath = fr'{filename}\{object}'
            image = cv2.imread(filepath)
            image = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), dsize = (112, 112))
            test_tensor[:, :, (i - 1) * 3 + (j - 7)] = image

    concatenated_training_samples = np.abs(tensor_decomposition(training_tensor, 112, mode = 1))
    training_labels = np.expand_dims(np.array(training_labels), axis = 1)

    concatenated_test_samples = np.abs(tensor_decomposition(test_tensor, 112, mode = 1))
    test_labels = np.array(test_labels)

    linear = svm.SVC(kernel = 'linear', C = 1, decision_function_shape = 'ovo').fit(concatenated_training_samples, training_labels)

    predictions = linear.predict(concatenated_test_samples)

    difference = np.subtract(predictions, test_labels)
    accuracy = (len(difference) - np.count_nonzero(difference)) / len(difference) * 100
    print(fr'The SVM achieves {accuracy}% accuracy.')

    print('Classification Report')
    print(classification_report(test_labels, predictions, target_names = [fr'class_{i}' for i in range(40)]))
    ## ------------------
    ## Part 2 End\
    ## ------------------

if __name__ == 'main':
    main()