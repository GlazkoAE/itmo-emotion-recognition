import time
from pathlib import Path
from threading import Thread

import cv2
import PySimpleGUI as sg
import torch
import yaml
from moviepy.editor import VideoFileClip

from face.faceProcessing import EmotionDetector


class FPS:
    def __init__(self, interval=30):
        self._interval = interval
        self._limit = interval - 1
        self._start = 0
        self._end = 0
        self._numFrames = 0
        self._fps = 0

    def fps(self):
        if self._numFrames % self._interval == self._limit:
            self._end = time.time()
            self._fps = 30 / (self._end - self._start)
            self._start = time.time()
        self._numFrames += 1
        return self._fps


class WebcamVideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False
        self.w = self.stream.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.h = self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def start(self):
        Thread(target=self.update, args=()).start()
        return self, (self.w, self.h)

    def update(self):
        while True:
            if self.stopped:
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True


class App:
    def __init__(self, config=None):
        self.config = config
        self.window1 = self.make_main_window()
        self.webcam_frames_grabber = None
        self.fps = FPS()
        self.emotion_detector = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # processing frames resolution
        self.w = None
        self.w = None

    def settings_window(self):
        """
        Checkbox Lair
        in another window
        """
        pass

    def create_emotion_detector(self, frame_w: int, frame_h: int, draw_face_bbox=True):
        """
        Create face processor

        """
        self.emotion_detector = EmotionDetector(
            inp_img_w=frame_w, inp_img_h=frame_h, draw_face_bbox=draw_face_bbox
        )

    def make_main_window(self):
        layout = [
            [sg.Image(key="-IMAGE-")],
            [
                sg.Text(
                    "Im status string, im describe something lol",
                    key="-TEXT-",
                    expand_x=True,
                    justification="c",
                )
            ],
            [
                sg.Input("", key="-IN-"),
                sg.FileBrowse("Local file", key="-FS-"),
                sg.Button("Process data", key="-PROCESS-"),
            ],
        ]

        return sg.Window(
            "EmotinRecognitionDemo", layout, location=(800, 600), finalize=True
        )

    def process_frame(self, get_frame, t):
        frame = get_frame(t)
        emotion = self.emotion_detector(frame)
        cv2.putText(
            frame,
            f"Facial emotion: {emotion}",
            (1, frame.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 25, 0),
            1,
        )

        return frame

    def run(self):
        try:
            face_emotion = ""  # temp
            while True:  # Event Loop
                window, event, values = sg.read_all_windows(timeout=1)

                if self.webcam_frames_grabber:
                    frame = self.webcam_frames_grabber.read()

                    face_emotion = self.emotion_detector(frame)

                    imgbytes = cv2.imencode(".png", frame)[1].tobytes()
                    self.window1["-IMAGE-"].update(data=imgbytes)
                    self.window1["-TEXT-"].update(
                        value=f"Fps: {int(self.fps.fps())}, emotion: {face_emotion}"
                    )

                if (
                    (event == "-PROCESS-")
                    and (values["-IN-"] == "")
                    and (self.webcam_frames_grabber == None)
                ):
                    self.webcam_frames_grabber, (self.w, self.h) = WebcamVideoStream(
                        src=0
                    ).start()
                    if self.emotion_detector is None:
                        self.create_emotion_detector(
                            self.w, self.h, draw_face_bbox=True
                        )

                elif (event == "-PROCESS-") and (self.webcam_frames_grabber != None):
                    self.webcam_frames_grabber.stop()
                    self.webcam_frames_grabber = None
                    self.window1["-IMAGE-"].update(data=None)

                elif (event == "-PROCESS-") and (values["-IN-"] != ""):
                    # values['-IN-'] - link valudation
                    if self.webcam_frames_grabber != None:
                        self.webcam_frames_grabber.stop()
                        self.webcam_frames_grabber = None
                        self.window1["-IMAGE-"].update(data=None)

                    videopath = str(values["-IN-"])
                    try:
                        video = VideoFileClip(videopath)
                    except Exception as e:
                        print("video load error")
                        sg.Popup(e, keep_on_top=True)
                        continue

                    if self.emotion_detector is None:
                        self.create_emotion_detector(
                            video.size[0], video.size[1], draw_face_bbox=True
                        )

                    newclip = video.fl(self.process_frame, apply_to="mask")
                    newclip.write_videofile("demo_output.mp4")

                elif event == sg.WIN_CLOSED or event == "Exit":
                    window.close()

                    if window == self.window1:  # if closing win 1, exit program
                        if self.webcam_frames_grabber != None:
                            self.webcam_frames_grabber.stop()
                            self.webcam_frames_grabber = None
                            self.window1["-IMAGE-"].update(data=None)
                        break

        except Exception as e:
            if self.webcam_frames_grabber != None:
                self.webcam_frames_grabber.stop()
            sg.popup_error_with_traceback(f"Somethig went wrong:", e)


if __name__ == "__main__":
    gui_app = App()
    gui_app.run()
