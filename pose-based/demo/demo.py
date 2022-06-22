import os

import cv2

from inferense import ArousalModel


def __draw_label(img, text, pos, bg_color):
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    scale = 2
    color = (0, 0, 0)
    thickness = cv2.FILLED
    margin = 2
    txt_size = cv2.getTextSize(text, font_face, scale, thickness)

    end_x = pos[0] + txt_size[0][0] + margin
    end_y = pos[1] - txt_size[0][1] - margin

    cv2.rectangle(img, pos, (end_x, end_y), bg_color, thickness)
    cv2.putText(img, text, pos, font_face, scale, color, 1, cv2.LINE_AA)


def main():
    model_path = "lstm-features.005-0.036.hdf5"
    frames = 30
    model = ArousalModel(seq_length=frames, saved_model=model_path)

    video = "demo_1.mp4"
    out_video = "demo_1_out.mp4"
    video_path = os.path.join("demo_videos", video)
    out_video_path = os.path.join("demo_videos", out_video)

    vid_capture = cv2.VideoCapture(video_path)

    if vid_capture.isOpened():
        fps = int(vid_capture.get(5))
        print("Frame Rate : ", fps, "frames per second")
        frame_count = vid_capture.get(7)
        print("Frame count : ", frame_count)
        height = vid_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width = vid_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        frameSize = (width, height)
    else:
        print("Error opening the video file")

    # out = cv2.VideoWriter(out_video_path, fourcc=cv2.VideoWriter_fourcc(*'mp4v'), fps=fps, frameSize=frameSiz

    while vid_capture.isOpened():

        ret, frame = vid_capture.read()
        if ret:
            prediction = model.predict(frame)
            text = str(prediction)
            __draw_label(frame, text, (20, 50), (255, 255, 255))
            cv2.imshow("Frame", frame)

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
    main()
