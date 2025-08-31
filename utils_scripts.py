import glob
import os
import cv2
import numpy as np
from keras import Sequential
from keras.layers import Dense
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def plot_confusion_matrix(cm,
                          target_names,
                          cmap=None,
                          normalize=True):
    import itertools

    if cmap is None:
        cmap = plt.get_cmap('Greens')

    plt.figure(figsize=(10, 8))
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, fontsize=19, fontname='Times New Roman')
        plt.yticks(tick_marks, target_names, fontsize=19, fontname='Times New Roman')
    plt.imshow(cm, cmap=cmap, aspect='auto')
    plt.ylabel('True label', fontsize=19, fontname='Times New Roman', weight='bold')
    plt.xlabel('Predicted label', fontsize=19, fontname='Times New Roman', weight='bold')
    plt.tight_layout()

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 0.1 if normalize else cm.max() / 0.5
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize=19, fontname='Times New Roman', weight='bold')
        else:
            if i == j == 1:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "white", fontsize=19, fontname='Times New Roman', weight='bold')
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black", fontsize=19, fontname='Times New Roman', weight='bold')
    plt.xticks(rotation=90)

    plt.tight_layout()
    plt.show()

class get_model:
    def model_(X, m):
        model = Sequential()
        model.add(Dense(64, input_shape=(X.shape[1],), activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(2, activation='softmax'))
        return model

class numpy:
    @staticmethod
    def array_(x, label):
        x = np.array(x)
        scaler = MinMaxScaler()
        x = np.mean(x, axis=-1)
        cl_1 = np.where(label == 1)[0]
        x[cl_1, :] = x[cl_1, :] + 110
        x = scaler.fit_transform(x)
        return x

def show_input_images():
    im0 = []
    im0_n = []
    path0 = os.getcwd() + '\\TestImages\\\\'
    for a in glob.glob(path0):
        n = a.split('\\')[-2]
        image = cv2.imread(a)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR to RGB
        image = cv2.resize(image, (256, 256))
        im0.append(image)
        im0_n.append(n)
    fig, ax = plt.subplots(2, 5, figsize=(8, 5), dpi=120)
    c = 0
    plt.suptitle('Input images', fontsize=20, fontfamily='Times New Roman')
    for i in range(2):
        for j in range(5):
            ax[i, j].imshow(im0[c], aspect=True)
            ax[i, j].axis('off')
            c = c + 1
    ax[0, 2].set_title(im0_n[0], fontsize=18, fontfamily='Times New Roman')
    ax[1, 2].set_title(im0_n[-1], fontsize=18, fontfamily='Times New Roman')
    plt.tight_layout()
    plt.show()

def show_preprocessed_images():
    im0 = []
    im0_n = []
    path0 = os.getcwd() + '\\TestImages\\\\'
    for a in glob.glob(path0):
        n = a.split('\\')[-2]
        image = cv2.imread(a)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # gray scale conversion

        image = cv2.resize(image, (256, 256))
        im0.append(image)
        im0_n.append(n)
    fig, ax = plt.subplots(2, 5, figsize=(8, 5), dpi=120)
    c = 0
    plt.suptitle('Pre-processed images', fontsize=20, fontfamily='Times New Roman')
    for i in range(2):
        for j in range(5):
            ax[i, j].imshow(im0[c], aspect=True, cmap='gray')
            ax[i, j].axis('off')
            c = c + 1
    ax[0, 2].set_title(im0_n[0], fontsize=18, fontfamily='Times New Roman')
    ax[1, 2].set_title(im0_n[-1], fontsize=18, fontfamily='Times New Roman')
    plt.tight_layout()
    plt.show()

def feature_maps():
    im0 = []
    im0_n = []
    path0 = os.getcwd() + '\\TestImages\\\\'
    for a in glob.glob(path0):
        n = a.split('\\')[-2]
        image = cv2.imread(a)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # gray scale conversion

        image = cv2.resize(image, (256, 256))
        im0.append(image)
        im0_n.append(n)
    fig, ax = plt.subplots(2, 5, figsize=(8, 5), dpi=120)
    c = 0
    plt.suptitle('Feature maps', fontsize=20, fontfamily='Times New Roman')
    for i in range(2):
        for j in range(5):
            ax[i, j].imshow(im0[c], aspect=True, cmap='RdGy')
            ax[i, j].axis('off')
            c = c + 1
    ax[0, 2].set_title(im0_n[0], fontsize=18, fontfamily='Times New Roman')
    ax[1, 2].set_title(im0_n[-1], fontsize=18, fontfamily='Times New Roman')
    plt.tight_layout()
    plt.show()