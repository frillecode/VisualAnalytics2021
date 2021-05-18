#!/usr/bin/python

import os
import cv2
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Conv2D, 
                                     MaxPooling2D, 
                                     Activation, 
                                     Flatten, 
                                     Dense,
                                     Dropout)
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.applications.vgg16 import (preprocess_input,
                                                 decode_predictions,
                                                 VGG16)


class CNN_books():
    """
    This is a class for creating and fitting a convoloutional neural network model to predict the genre of a book based on images of the covers. The model relies on a pretrained model (VGG-16).
    
    """
    def __init__(self, args):
        self.args = args
        
    def preprocessing(self, folder_path):
        """This function extracts image data and performs preprocessing steps to get it in the correct format to feed to our model
        Input:
            folder_path: path to folder containing subfolders with image data
        """
        # List of the genres I am analysing
        genres = ["Business-Finance-Law", "Childrens-Books", "Crime-Thriller", "Dictionaries-Languages", "Food-Drink", "History-Archaeology", "Medical", "Romance", "Science-Fiction-Fantasy-Horror", "Teen-Young-Adult"]
        
        # Empty lists for extracting data into
        labelNames = []
        data = []
        labels = []
        
        # Looping through the folders
        for genre in genres:
            folder = os.path.join(folder_path, genre) #path to folder of genre
            labelNames.append(genre) #create list of possible labels
            # Take the current subfolder
            for pic in Path(folder).glob("*.jpg"):
                pic_array = cv2.imread(str(pic)) #load image
                compressed = cv2.resize(pic_array, (224, 224), interpolation = cv2.INTER_AREA) #resize image to fit VGG-16
                data.append(compressed) #append image to list
                labels.append(genre) #append label to list
        
        # Turn into numpy arrays (this is the format that we need for the model)
        labels = np.array(labels)
        data = np.array(data)

        # Normalise data
        data = data.astype("float")/255.0

        # Split data and assign to 'self'
        (self.trainX, self.testX, self.trainY, self.testY) = train_test_split(data, 
                                                          labels, 
                                                          test_size=0.2)
        self.labelNames = labelNames 
        
        # Convert labels to one-hot encoding
        lb = LabelBinarizer()
        self.trainY = lb.fit_transform(self.trainY)
        self.testY = lb.fit_transform(self.testY)    
      
        
    def create_model(self):
        """This function initializes and compiles the model for the convolutional neural network using the pretrained VGG-16 model
        """
        
        # Load the pretrained model without classifier layers
        self.model = VGG16(include_top=False, 
                          pooling='avg',
                          input_shape=(224, 224, 3))
        
        # Mark loaded layers as not trainable
        for layer in self.model.layers:
            layer.trainable = False
            
        # Add new classifier layers
        flat1 = Flatten()(self.model.layers[-1].output)
        class1 = Dense(256, activation='relu')(flat1)
        output = Dense(12, activation='softmax')(class1)

        # Define new model
        self.model = Model(inputs=self.model.inputs, 
                      outputs=output)

        # Compile model
        self.model.compile(optimizer="adam", 
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    
        return self.model.summary()
    
    
    def fit_model(self):
        """This function fits the model to the data and returns the classification report
        """

        self.H = self.model.fit(self.trainX, self.trainY, 
                              validation_data=(self.testX, self.testY), 
                              batch_size=self.args['batch_size'],
                              epochs=self.args['epochs'],
                              verbose=1)
    

    def eval_model(self):
        """This function calculates measures to evaluate model performance. It creates and saves a classification report along with a plot of the model as it learns.
        """
        # Get predictions
        self.predictions = self.model.predict(self.testX, batch_size=self.args['batch_size'])
        
        # Comparing predictions to test labels
        cm = classification_report(self.testY.argmax(axis=1),
                            self.predictions.argmax(axis=1),
                            target_names=self.labelNames)
        
        # Save classification report   
        outpath = os.path.join("out", "evaluation_metric.txt")
        with open(outpath, "w", encoding="utf-8") as file:
            file.write(cm)
        
        return cm
    
    def plot_history(self): 
        ''' Plot of the model as it learns 
        '''
        # Visualize performance
        plt.style.use("fivethirtyeight")
        plt.figure()
        plt.plot(np.arange(0, self.args['epochs']), self.H.history["loss"], label="train_loss")
        plt.plot(np.arange(0, self.args['epochs']), self.H.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, self.args['epochs']), self.H.history["accuracy"], label="train_acc")
        plt.plot(np.arange(0, self.args['epochs']), self.H.history["val_accuracy"], label="val_acc")
        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        # Save plot
        plot_path = os.path.join("out", "performance.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")

def main():
    ap = argparse.ArgumentParser(description="[INFO] This script takes image data, creates a convolutional neural network model using a pretrained VGG-16 model, and saves the results of the model")
    # Argument for specifying number of epochs
    ap.add_argument("-e", 
                "--epochs", 
                required=False, 
                type=int, 
                default=10, 
                help="int, number of epochs") 
    # Argument for specifying batch size
    ap.add_argument("-bs", 
                "--batch_size", 
                required=False, 
                type=int, 
                default=128, 
                help="int, batch size")     
    
    args = vars(ap.parse_args())
    
    
    # Run 
    convNN = CNN_books(args)
    print("[INFO] Loading and preprocessing data")
    convNN.preprocessing(os.path.join("..", "..", "data", "final_project", "book-covers"))
    print("[INFO] Creating model")
    summary = convNN.create_model()
    print(summary)
    print("[INFO] Fitting model")
    convNN.fit_model()
    cm = convNN.eval_model()
    print(f"[INFO] Classification report: \n {cm}")
    convNN.plot_history()

# Define behaviour when called from command line
if __name__=="__main__":
    main()
    print("[INFO] DONE! You can find the results of the performance of the model in the output-folder")

