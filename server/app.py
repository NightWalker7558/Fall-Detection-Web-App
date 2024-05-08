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

outputFrame = None
cap_rtsp = None
lock = Lock()

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
        filename = secure_filename(uploaded_file.filename)

        # Save the uploaded file to a temporary directory
        uploaded_file_url = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print('File path:', uploaded_file_url)

        uploaded_file.save(uploaded_file_url)
        print("Image Read...")

        img = cv2.imread(uploaded_file_url)

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

        # Save the processed image to an output directory
        output_file_url = os.path.join(
            app.config['OUTPUT_FOLDER'], 'image_from_blob.jpg')
        cv2.imwrite(output_file_url, img_from_blob)
        print("Image Saved...")

        # Encode the processed image in base64
        with open(output_file_url, 'rb') as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
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


# Load the model outside of the detect function
model = YOLO('model/weights/singleclass/best.pt', task="detect",
             verbose=False)  # pretrained YOLOv8n model


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
    # grab global references to the output frame and lock variables
    global outputFrame, lock

    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            # Check if the video capture is opened
            if not cap_rtsp.isOpened():
                # Load an error image
                outputFrame = cv2.imread('error.jpg')

            else:
                # read a frame from cap_rtsp
                ret, frame = cap_rtsp.read()

                # check if the frame was successfully read
                if not ret:
                    # Load an error image
                    outputFrame = cv2.imread('error.jpg')
                else:
                    # apply detection on the frame
                    outputFrame = detect(frame)

            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

            # ensure the frame was successfully encoded
            if not flag:
                continue

        # yield the output frame in the byte format
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')


@app.route('/process_rtsp', methods=['POST'])
def process_rtsp():
    try:
        global source, cap_rtsp
        # get the rtsp link from the form data
        source = request.form['rtspUrl'] + "/video"
        # release the previous video capture
        if cap_rtsp is not None:
            cap_rtsp.release()
        # create a new video capture with the new source
        cap_rtsp = cv2.VideoCapture(source)
        # check if the video capture was successfully opened
        if not cap_rtsp.isOpened():
            raise ValueError('Failed to open video capture')
        # return the url where the live stream can be accessed
        return jsonify(url='http://localhost:5000/rtsp_stream')
    except Exception as e:
        return jsonify(error=str(e)), 400


@app.route('/rtsp_stream')
def rtsp_stream():
    print("Starting RTSP stream...")
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route('/process_webcam', methods=['GET'])
def process_webcam():
    print("Processing webcam...")
    try:
        model = YOLO('model/weights/200epochs/best.pt',
                     task="detect", verbose=False)
        print("Model Loaded...")

        # Open the webcam
        cap = cv2.VideoCapture(0)

        # Set a fixed frame rate
        fps = 30
        # Set the video's width and height
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create a VideoWriter object to write processed frames to a new video
        output_filename = os.path.join(
            app.config['OUTPUT_FOLDER'], 'input.mp4')
        out = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(
            *'mp4v'), fps, (frame_width, frame_height))

        start_time = time.time()  # Start the timer
        max_duration = 25  # Maximum duration of the webcam stream in seconds

        while cap.isOpened():
            if time.time() - start_time > max_duration:
                break
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


if __name__ == '__main__':
    app.run(debug=True)
