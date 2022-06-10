import typing
from numpy import ndarray
import mediapipe as mp


def _get_face_connection_drawing_spec(spec):
    """
    Set connection drawing spec for face landmarks.

    :param str spec: 'mesh' or 'tesselation'
    :return: connection drawing style
    """
    if spec == 'mesh':
        connection_drawing_spec = mp.solutions.drawing_styles.get_default_face_mesh_contours_style()
    elif spec == 'tesselation':
        connection_drawing_spec = mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()
    else:
        raise ValueError('Incorrect spec value')
    return connection_drawing_spec


class PoseTracker:
    def __init__(self,
                 static_image_mode=False,
                 model_complexity=1,
                 smooth_landmarks=True,
                 enable_segmentation=False,
                 smooth_segmentation=True,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):

        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.tracker = mp.solutions.pose.Pose(static_image_mode=self.static_image_mode,
                                              model_complexity=self.model_complexity,
                                              smooth_landmarks=self.smooth_landmarks,
                                              enable_segmentation=self.enable_segmentation,
                                              smooth_segmentation=self.smooth_segmentation,
                                              min_detection_confidence=self.min_detection_confidence,
                                              min_tracking_confidence=self.min_tracking_confidence)

        self.drawing = mp.solutions.drawing_utils

    def get_landmarks(self, image: ndarray):
        """
        Processes an image and returns the pose landmarks
        on the most prominent person detected.

        :param image: an RGB image as numpy ndarray
        :return: landmarks on the most prominent person detected
        """
        return self.tracker.process(image)

    def draw_pose(self, image, landmarks):
        """
        Draw landmarks and connections of pose.

        :param ndarray image: an BGR image as numpy ndarray
        :param typing.NamedTuple landmarks: all landmarks from get_landmarks function
        """
        self.drawing.draw_landmarks(
            image,
            landmarks,
            mp.solutions.holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style()
        )
