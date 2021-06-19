from flask import Flask, render_template, Response
import cv2

from camera import Camera

app = Flask(__name__)


def gen_video(camera):
    while True:
        success, image = camera.get_image()
        if not success:
            break

        ret, frame = cv2.imencode(".jpg", image)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame.tobytes() + b"\r\n"
        )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/stream")
def stream():
    return render_template("stream.html")


@app.route("/video_feed")
def video_feed():
    cam = Camera()
    return Response(
        gen_video(cam), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/kill_camera")
def kill_camera():
    cam = Camera()
    del cam
    return render_template("camera_killed.html")


if __name__ == "__main__":
    app.run(port=8000, debug=False, threaded=True)
