import telebot
import os
import cv2
import argparse
from service.image_loader import ImageTransformer
from face.inference import Model as FaceModel
from pose.model.inferense import ArousalModel
from service.drawer import Drawer

if __name__ == "__main__":
    # Get telegram bot token from arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-key", type=str, default="some_token", help="Telegram bot secret token"
    )
    args = parser.parse_args()

    # init all modules
    bot = telebot.TeleBot(args.key, parse_mode=None)
    image_loader = ImageTransformer()
    face_model = FaceModel()
    pose_model = ArousalModel(saved_model='./pose/best-lstm.hdf5')
    drawer = Drawer()


    @bot.message_handler(content_types=["text"])
    def handle_start(message):  # start message reaction
        if message.text == "/start":
            bot.send_message(
                message.from_user.id, "Send me video or photo"
            )
            bot.register_next_step_handler(message, handle_dock_photo)
        else:
            bot.send_message(
                message.from_user.id,
                "Write /start if you want to use bot",
            )


    @bot.message_handler(content_types=["photo"])
    def handle_dock_photo(message):  # image massage reaction
        download_file = []
        try:
            file_info = bot.get_file(message.photo[-1].file_id)
            download_file = bot.download_file(file_info.file_path)

            image = image_loader.image_as_array(download_file)
            prediction, box = face_model.predict(image)

            # Send message
            if prediction is not None:
                text = "This person looks like " + prediction
            else:
                text = "I can't find any face, sorry"
            bot.send_message(message.chat.id, text)

            # Send image
            image = drawer.draw_face_box(image, box)
            image = drawer.draw_face_predict(image, prediction)
            image = image_loader.image_as_bytes(image)
            bot.send_photo(message.chat.id, image)

        except Exception as e:
            bot.reply_to(message, "Can't load image")


    @bot.message_handler(content_types=["video"])
    def handle_dock_video(message):
        try:
            file_info = bot.get_file(message.video.file_id)
            download_file = bot.download_file(file_info.file_path)

            bot.reply_to(message, "Processing")

            input_file_name = 'input_video' + str(message.chat.id) + '.mp4'
            output_file_name = 'output_video' + str(message.chat.id) + '.mp4'

            if os.path.exists(output_file_name):
                os.remove(output_file_name)

            if os.path.exists(input_file_name):
                os.remove(input_file_name)

            with open(input_file_name, "wb") as input_file:
                input_file.write(download_file)

            vid_capture = cv2.VideoCapture(input_file_name)
            w = vid_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
            h = vid_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = vid_capture.get(cv2.CAP_PROP_FPS)
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            writer = cv2.VideoWriter(output_file_name, fourcc, int(fps),
                                     (int(w), int(h)), True)

            for frame_num in range(int(vid_capture.get(cv2.CAP_PROP_FRAME_COUNT))):
                ret, frame = vid_capture.read()
                if ret:
                    pose_prediction = pose_model.predict(frame)
                    face_prediction, box = face_model.predict(frame)
                    frame = drawer.draw_face_box(frame, box)
                    frame = drawer.draw_face_predict(frame, face_prediction)
                    frame = drawer.draw_pose_predict(frame, pose_prediction)
                    writer.write(frame)
                else:
                    bot.reply_to(message, "Can't process video")
                    break

            vid_capture.release()
            writer.release()

            with open(output_file_name, 'rb') as output_file:
                send_file = output_file.read()

            bot.send_video(message.chat.id, send_file)
            os.remove(output_file_name)
            os.remove(input_file_name)

        except Exception as e:
            bot.reply_to(message, "Something went wrong")


    bot.polling(non_stop=True, interval=0)
