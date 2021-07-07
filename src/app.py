import warnings
warnings.simplefilter('ignore', FutureWarning)
import sys
import io
from pathlib import Path
from tempfile import NamedTemporaryFile
import shutil
from fastapi import FastAPI, HTTPException, UploadFile, File
from starlette.responses import StreamingResponse
from flask import jsonify
import logging
import uvicorn
from uvicorn.config import LOGGING_CONFIG

from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.utils import get_file
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
import numpy as np
import cv2
import cvlib as cv

logger = logging.getLogger(__name__)
app = FastAPI()


MODEL_PATH = "../model/gender_detection.model"
TRAIN_DIR = "../train-data"
# load model and tokenizer
print("Loading model...")
sess = tf.Session()
graph = tf.get_default_graph()

# IMPORTANT: models have to be loaded AFTER SETTING THE SESSION for keras!
# Otherwise, their weights will be unavailable in the threads after the session there has been set
set_session(sess)
# load pre-trained model
model = load_model(MODEL_PATH)
model._make_predict_function()
logger.info("model: %s", model)


def _process(image):
    with graph.as_default():
        set_session(sess)
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
            conf = model.predict(face_crop)[0]
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


def save_upload_file_tmp(upload_file: UploadFile) -> Path:
    try:
        suffix = Path(upload_file.filename).suffix
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(upload_file.file, tmp)
            tmp_path = Path(tmp.name)
    finally:
        upload_file.file.close()
    return tmp_path


@app.post("/genpredict/")
def predict(image: UploadFile = File(...)):
    # try:
    logger.info("image: %s", image)
    image_path = str(save_upload_file_tmp(image))
    logger.info("image_path: %s", image_path)
    # read input image
    cv2image = cv2.imread(image_path)
    logger.info("cv2image: %s", cv2image)
    labeled_image = _process(cv2image)
    # except:
    #     raise HTTPException(status_code=500, detail=sys.exc_info()[0])

    res, im_png = cv2.imencode(".png", labeled_image)
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")


@app.get("/")
def root():
    return jsonify({'msg': 'Try POSTing to the /genpredict endpoint'})


if __name__ == "__main__":
    # uvicorn main:app --reload
    LOGGING_CONFIG.get("loggers").setdefault(
        "",
        {
            "level": "DEBUG",
            "handlers": ["default"],
        },
    )
    args = dict(host="0.0.0.0", port=7000, debug=True, root_path="")
    uvicorn.run("app:app", **args)
