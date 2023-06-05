import cv2
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as lr
from PIL import Image
import PIL.ImageOps

X = np.load('image.npz')['arr_0']
y = pd.read_csv('labels.csv')['labels']

classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L','M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 7500, test_size = 2500, random_state = 9)

X_train = X_train/255.0
X_test = X_test/255.0

clf = lr(solver = 'saga', multi_class= 'multinomial')

clf.fit(X_train, y_train)

def get_prediction(img):
    
    image_PIL = Image.open(img)
    image_bw = image_PIL.convert('L') #cvt to grayscale

    image_bw_resized = image_bw.resize((28,28), Image.ANTIALIAS)

    pixel_filter = 20
        
    min_pixel = np.percentile(image_bw_resized, pixel_filter)
        
    image_bw_resized_scaled = np.clip(image_bw_resized-min_pixel, 0, 255)
        
    max_pixel = np.max(image_bw_resized)
        
    image_bw_resized_scaled = np.asarray(image_bw_resized_scaled)/max_pixel
        
    test_sample = np.array(image_bw_resized_scaled).reshape(1,784)
        
    test_pred = clf.predict(test_sample)

    return test_pred[0]
