import cv2
from PoseTracker import PoseTracker

if __name__ == "__main__":

    cap = cv2.VideoCapture(0)

    tracker = PoseTracker(static_image_mode=False,
                          model_complexity=1,
                          smooth_landmarks=True,
                          enable_segmentation=False,
                          smooth_segmentation=True,
                          min_detection_confidence=0.3,
                          min_tracking_confidence=0.3
                          )

    while True:
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get landmarks
        landmarks = tracker.get_landmarks(image)

        # Draw landmark annotation on the image
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        tracker.draw_pose(image, landmarks)
        tracker.draw_hands(image, landmarks)

        # Flip the image horizontally for a selfie-view display
        cv2.imshow('Pose and face estimation demo', cv2.flip(image, 1))

        # Flip the image horizontally for a selfie-view display
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
