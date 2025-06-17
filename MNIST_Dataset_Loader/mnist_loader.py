import os
import numpy as np
import pandas as pd
import cv2


class MNIST(object):
    def __init__(self, project_root_path):
        self.project_root_path = project_root_path

        self.test_images = []
        self.test_labels = []

        self.train_images = []
        self.train_labels = []

    def _load_local_data(self, data_type):
        base_path = os.path.join(self.project_root_path, 'Dataset')
        if data_type == 'train':
            data_dir = os.path.join(base_path, 'train')
            labels_path = os.path.join(base_path, 'train.csv')
            txt_col_name = 'Train_txt_name'
        else: # data_type == 'test'
            data_dir = os.path.join(base_path, 'test_no_label')
            labels_path = os.path.join(base_path, 'test.csv')
            txt_col_name = 'Test_txt_name'

        images = []
        labels = []
        filenames = []

        label_df = pd.read_csv(labels_path)

        for index, row in label_df.iterrows():
            txt_name = str(row[txt_col_name]) + '.txt'
            img_path = os.path.join(data_dir, txt_name)
            digit = row['Digit']
            
            with open(img_path, 'r') as f:
                lines = f.readlines()
                img_data = np.array([[int(char) for char in line.strip()] for line in lines], dtype=np.float32)
                resized_img = cv2.resize(img_data, (28, 28), interpolation=cv2.INTER_AREA)
                images.append(resized_img)
                labels.append(digit)
                filenames.append(str(row[txt_col_name]))
        
        return np.array(images), np.array(labels), filenames

    def load_testing(self):
        ims, labels, fnames = self._load_local_data('test')
        self.test_images = ims
        self.test_labels = labels
        return ims, labels, fnames

    def load_training(self):
        ims, labels, fnames = self._load_local_data('train')
        self.train_images = ims
        self.train_labels = labels
        return ims, labels, fnames

    @classmethod
    def display(cls, img, width=28, threshold=200):
        render = ''
        for i in range(len(img)):
            if i % width == 0:
                render += '\n'
            if img[i] > threshold:
                render += '@'
            else:
                render += '.'
        return render
