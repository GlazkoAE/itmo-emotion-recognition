import argparse

import cv2
from model.inferense import ArousalModel


def draw_label(img, text, pos, bg_color):
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.5
    color = (0, 0, 0)
    thickness = cv2.FILLED
    margin = 2
    txt_size = cv2.getTextSize(text, font_face, scale, thickness)

    end_x = pos[0] + txt_size[0][0] + margin
    end_y = pos[1] - txt_size[0][1] - margin

    cv2.rectangle(img, pos, (end_x, end_y), bg_color, thickness)
    cv2.putText(img, text, pos, font_face, scale, color, 1, cv2.LINE_AA)


def main(inputs):
    model_path = inputs.weights
    frames = 30

    video = inputs.video

    model = ArousalModel(seq_length=frames, saved_model=model_path)

    vid_capture = cv2.VideoCapture(video)

    if vid_capture.isOpened():
        fps = int(vid_capture.get(5))
        print("Frame Rate : ", fps, "frames per second")
        frame_count = vid_capture.get(7)
        print("Frame count : ", frame_count)
        height = vid_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width = vid_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        # frameSize = (width, height)
    else:
        print("Error opening the video file")

    # out = cv2.VideoWriter(out_video_path, fourcc=cv2.VideoWriter_fourcc(*'mp4v'), fps=fps, frameSize=frameSiz

    while vid_capture.isOpened():

        ret, frame = vid_capture.read()
        if ret:
            # frame.flags.writeable = False
            prediction = model.predict(frame)
            text = "Arousal: " + str(prediction)
            # frame.flags.writeable = True
            draw_label(frame, text, (20, 50), (255, 255, 255))
            cv2.imshow("Demo (press 'q' to exit)", frame)

            # out.write(frame)

            k = cv2.waitKey(20)
            # 113 is ASCII code for q key
            if k == 113:
                break
        else:
            break

    vid_capture.release()
    # out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demo for human arousal prediction model"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="best-lstm.hdf5",
        help="saved weights of trained model",
    )
    parser.add_argument(
        "--video", type=str, default="demo_videos/demo_1.mp4", help="video path"
    )
    args = parser.parse_args()

    main(args)
