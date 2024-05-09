import re
from flask import Flask, Response, jsonify, request, send_file, send_from_directory
from werkzeug.utils import secure_filename
import os
import cv2
from ultralytics import YOLO
from flask_cors import CORS
import base64
import ffmpeg
import glob
import time
from threading import Lock
import numpy as np

streaming = False
outputFrame = None
cap_stream = None
lock = Lock()

# Load a model for Streaming
model = YOLO('model/weights/singleclass/best.pt', task="detect",
             verbose=False)

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'tmp'
app.config['OUTPUT_FOLDER'] = 'output'

# Ensure upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Ensure output folder exists
if not os.path.exists(app.config['OUTPUT_FOLDER']):
    os.makedirs(app.config['OUTPUT_FOLDER'])


@app.route('/process_video', methods=['POST'])
def process_video():
    print("Processing video...")
    try:
        model = YOLO('model/weights/200epochs/best.pt',
                     task="detect", verbose=False)
        print("Model Loaded...")

        # Open the video file
        print('Files in request:', request.files)

        uploaded_file = request.files['file']
        filename = secure_filename(uploaded_file.filename)

        print('Secure filename:', filename)

        uploaded_file_url = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print('File path:', uploaded_file_url)

        uploaded_file.save(uploaded_file_url)
        print("Video Uploaded...")

        cap = cv2.VideoCapture(uploaded_file_url)

        # Get the video's frame rate
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Get the video's width and height
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create a VideoWriter object to write processed frames to a new video
        output_filename = os.path.join(
            app.config['OUTPUT_FOLDER'], 'input.mp4')
        out = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(
            *'mp4v'), fps, (frame_width, frame_height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process the frame
            results = model(frame, verbose=False, conf=0.65)

            # Write each processed frame to the new video
            for result in results:
                processed_frame = result.plot()
                high_confidence = True

                for box in result.boxes:
                    class_id = int(box.cls.item())
                    class_name = result.names[class_id]

                # If all detections have a confidence score greater than 0.6, write the frame to the video

                out.write(processed_frame)
                if not out.isOpened():
                    print("Failed to write frame")

        # Release the video capture and writer
        cap.release()
        out.release()
        if not out.isOpened():
            print("Failed to finalize video file")

        stream = ffmpeg.input(os.path.join(
            app.config['OUTPUT_FOLDER'], "input.mp4"))
        stream = ffmpeg.output(stream, os.path.join(
            app.config['OUTPUT_FOLDER'], "output.mp4"))
        ffmpeg.run(stream, overwrite_output=True)

        print("Video Processed...")

        # Delete all files in the tmp and output folders except for output.mp4
        for folder in [app.config['UPLOAD_FOLDER'], app.config['OUTPUT_FOLDER']]:
            for filename in glob.glob(os.path.join(folder, '*')):
                if filename != os.path.join(app.config['OUTPUT_FOLDER'], 'output.mp4'):
                    os.remove(filename)

        return jsonify(url="http://localhost:5000/get_video")
    except Exception as e:
        return jsonify(error=str(e)), 400


@app.route('/process_image', methods=['POST'])
def process_image():
    print("Processing image...")
    try:
        # Load a model
        model = YOLO('model/weights/singleclass/best.pt',
                     task="detect", verbose=False)  # pretrained YOLOv8n model
        print("Model Loaded...")

        # Read the image file
        uploaded_file = request.files['file']

        # Convert the uploaded file to a numpy array
        filestr = uploaded_file.read()
        npimg = np.fromstring(filestr, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        print("Image Read...")

        # Process the image
        results = model(img, verbose=False, conf=0.4)
        print("Image Processed...")

        # Process the results
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls.item())
                class_name = result.names[class_id]
                confidence = box.conf.item()
                print(f"Confidence: {confidence}")
                print(f"Class Name: {class_name}")

        processed_img = result.plot()
        print("Image Plotted...")

        blob = cv2.dnn.blobFromImage(processed_img)
        print("Blob Created...")

        # Convert the blob back into an image
        img_from_blob = cv2.dnn.imagesFromBlob(blob)[0]
        print("Image from Blob Created...")

        # Encode the processed image in base64
        retval, buffer = cv2.imencode('.jpg', img_from_blob)
        encoded_image = base64.b64encode(buffer).decode('utf-8')
        print("Image Encoded...")

        print("All Steps Completed...")

        # Return the encoded image as JSON
        return jsonify(url=f"data:image/jpeg;base64,{encoded_image}")
    except Exception as e:
        return jsonify(error=str(e)), 400


@app.route('/get_video', methods=['GET'])
def get_video():
    video_path = os.path.join(app.config['OUTPUT_FOLDER'], "output.mp4")
    range_header = request.headers.get('Range', None)
    if not range_header:
        return send_file(video_path, mimetype='video/mp4')

    size = os.path.getsize(video_path)
    byte1, byte2 = 0, None

    # Extract range values from Range header
    m = re.search('(\d+)-(\d*)', range_header)
    g = m.groups()

    if g[0]:
        byte1 = int(g[0])
    if g[1]:
        byte2 = int(g[1])

    length = size - byte1
    if byte2 is not None:
        length = byte2 - byte1

    data = None
    with open(video_path, 'rb') as f:
        f.seek(byte1)
        data = f.read(length)

    rv = Response(data,
                  206,
                  mimetype='video/mp4',
                  content_type='video/mp4',
                  direct_passthrough=True)

    rv.headers.add('Content-Range',
                   'bytes {0}-{1}/{2}'.format(byte1, byte1 + length - 1, size))

    return rv


def detect(frame):
    # Process the frame
    results = model(frame, verbose=False, conf=0.4)

    # Check if any objects were detected
    if not results:
        return frame

    # Process the results
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls.item())
            class_name = result.names[class_id]
            confidence = box.conf.item()

    # Plot the results on the frame
    processed_frame = result.plot()

    print(processed_frame.shape)
    return processed_frame


def generate():
    global outputFrame, lock, streaming

    while True:
        if not streaming:
            break

        with lock:
            if cap_stream is None or not cap_stream.isOpened():
                outputFrame = cv2.imread('error.jpg')
            else:
                ret, frame = cap_stream.read()
                if not ret:
                    outputFrame = cv2.imread('error.jpg')
                else:
                    outputFrame = detect(frame)

            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            if not flag:
                continue

        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')


@app.route('/process_rtsp', methods=['POST'])
def process_rtsp():
    try:
        global source, cap_stream, streaming, outputFrame
        source = request.form['rtspUrl'] + "/video"
        if cap_stream is not None:
            cap_stream.release()
            cap_stream = None
        cap_stream = cv2.VideoCapture(source)
        if not cap_stream.isOpened():
            raise ValueError('Failed to open video capture')
        streaming = True
        outputFrame = None  # Reset the output frame
        return jsonify(url='http://localhost:5000/live_stream')
    except Exception as e:
        return jsonify(error=str(e)), 400


@app.route('/process_webcam', methods=['GET'])
def process_webcam():
    try:
        global cap_stream, streaming, outputFrame
        if cap_stream is not None:
            cap_stream.release()
            cap_stream = None
        cap_stream = cv2.VideoCapture(0)
        if not cap_stream.isOpened():
            raise ValueError('Failed to open video capture')
        streaming = True
        outputFrame = None  # Reset the output frame
        return jsonify(url='http://localhost:5000/live_stream')
    except Exception as e:
        return jsonify(error=str(e)), 400


@app.route('/stop_stream', methods=['GET'])
def stop_stream():
    try:
        global cap_stream, streaming, outputFrame
        if cap_stream is not None:
            cap_stream.release()
        cap_stream = None
        streaming = False
        outputFrame = None
        return jsonify(success=True)
    except Exception as e:
        return jsonify(error=str(e)), 400


@app.route('/live_stream')
def live_stream():
    print("Starting RTSP stream...")
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == '__main__':
    app.run(debug=True)
