"""
Simple app to upload an image via a web form 
and view the inference results on the image in the browser.
"""
import argparse
import io
import os
from PIL import Image

import torch
from flask import Flask, render_template, request, redirect

app = Flask(__name__)
# parser = argparse.ArgumentParser(description="Flask app exposing yolov5 models")
# parser.add_argument("--port", default=5000, type=int, help="port number")
# args = parser.parse_args()
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # force_reload = recache latest code
model.eval()  # debug=True causes Restarting with stat


@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if not file:
            return

        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        results = model([img])

        results.render()  # updates results.imgs with boxes and labels
        results.save(save_dir=".")

        os.replace(".2/image0.jpg", "static/out.jpg")
        os.rmdir(".2")

        return render_template("index.html",img="static/out.jpg")

    return render_template("index.html",img=None)



