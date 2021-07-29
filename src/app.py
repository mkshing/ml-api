import warnings
warnings.simplefilter('ignore', FutureWarning)
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

import cv2
from classifier import Classifier

logger = logging.getLogger(__name__)
app = FastAPI()

model = Classifier()


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
    labeled_image = model._process(cv2image)
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
