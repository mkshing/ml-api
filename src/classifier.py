import logging

from keras.preprocessing.image import img_to_array
from keras.models import load_model
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
import numpy as np
import cv2
import cvlib as cv

logger = logging.getLogger(__name__)

MODEL_PATH = "../model/gender_detection.model"
TRAIN_DIR = "../train-data"


class Classifier:
    def __init__(self):

        # load model and tokenizer
        print("Loading model...")
        self.sess = tf.compat.v1.Session()
        self.graph = tf.compat.v1.get_default_graph()

        # IMPORTANT: models have to be loaded AFTER SETTING THE SESSION for keras!
        # Otherwise, their weights will be unavailable in the threads after the session there has been set
        set_session(self.sess)
        # load pre-trained model
        self.model = load_model(MODEL_PATH)
        self.model._make_predict_function()
        logger.info("model: %s", self.model)

    def _process(self, image):
        with self.graph.as_default():
            set_session(self.sess)
            # detect faces in the image
            face, confidence = cv.detect_face(image)
            logger.info("num of faces: %s", len(face))
            classes = ['man', 'woman']

            # loop through detected faces
            for idx, f in enumerate(face):
                # get corner points of face rectangle
                (startX, startY) = f[0], f[1]
                (endX, endY) = f[2], f[3]

                # draw rectangle over face
                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

                # crop the detected face region
                face_crop = np.copy(image[startY:endY, startX:endX])

                # preprocessing for gender detection model
                face_crop = cv2.resize(face_crop, (96, 96))
                face_crop = face_crop.astype("float") / 255.0
                face_crop = img_to_array(face_crop)
                face_crop = np.expand_dims(face_crop, axis=0)

                # apply gender detection on face
                # logger.info("Predicting...")
                conf = self.model.predict(face_crop)[0]
                logger.info("probs: %s", conf)
                # print(classes)

                # get label with max accuracy
                idx = np.argmax(conf)
                label = classes[idx]

                label = "{}: {:.2f}%".format(label, conf[idx] * 100)

                Y = startY - 10 if startY - 10 > 10 else startY + 10

                # write label and confidence above face rectangle
                cv2.putText(image, label, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)
        return image
