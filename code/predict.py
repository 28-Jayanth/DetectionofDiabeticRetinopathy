#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tensorflow.keras.models import load_model

from PIL import Image
import numpy as np


def load(filename):
    np_image = Image.open(filename)
    np_image = np_image.resize((256,256))
    np_image = np.asarray(np_image, dtype= np.float32)
    np_image = np_image / 255
    np_image = np_image.reshape(-1,256,256,3)
    return np_image


class retinopathy:
    def __init__(self, filename):
        self.filename = filename

    def predictionretinopathy(self):
        # load model
        model = load_model('retina_weights.hdf5')

        # summarize model
        # model.summary()
        imagename = self.filename
        test_image = load(imagename)
        result = model.predict(test_image)
        result = np.argmax(result)

        if result == 0:
            prediction = 'Mild diabetic retinopathy observed 20%'
            return [prediction]
        elif result == 1:
            prediction = 'Moderate diabetic retinopathy observed 35%'
            return [prediction]
        elif result == 2:
            prediction = 'No diabetic retinopathy observed 0%'
            return [prediction]
        elif result == 3:
            prediction = 'Proliferate diabetic retinopathy observed 50%'
            return [prediction]
        else:
            prediction = 'Severe diabetic retinopathy observed 80%'
            return [prediction]
