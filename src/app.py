"""Real-time facial emotion detection."""
import os
from typing import Any

import cv2
import numpy as np
from tensorflow.keras.models import load_model

from utils.classes import AgeGroups, Emotions, Products


class App:
    """OpenCV Video App for facial age and emotion recognition.

    Attributes:
        _age_classifier: The age classifier model.
        _emotion_classifier: The emotion classifier model.
    """

    def __init__(self, age_classifier_filepath: str, emotion_classifier_filepath: str) -> None:
        """Initialize the application."""
        self._age_classifier = self._load_model(age_classifier_filepath)
        self._emotion_classifier = self._load_model(emotion_classifier_filepath)

    def run(self) -> None:
        """Runs the application."""
        cv2.ocl.setUseOpenCL(False)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + \
                                             'haarcascade_frontalface_default.xml')

        capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 3)

        while True:
            retval, image = capture.read()

            if not retval: break
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces_detected = face_cascade.detectMultiScale(gray_image, 1.1, 6, minSize=(150, 150))

            if not isinstance(faces_detected, tuple):
                for (x, y, width, height) in faces_detected:
                    cv2.rectangle(img=image,
                                  pt1=(x, y),
                                  pt2=(x + width, y + height),
                                  color=(153, 153, 153),
                                  thickness=2)

                    roi_gray = gray_image[y:y + width, x:x + height]
                    roi_gray = cv2.resize(roi_gray, (48, 48))
                    pixels = image.img_to_array(roi_gray)
                    pixels = np.expand_dims(pixels, axis=0)

                    age_group = self._recognize_age_group(pixels)
                    emotion = self._recognize_emotion(pixels)
                    age_group_and_emotion = f'({age_group}, {emotion})'
                    recommended_product = self._recommend_product(age_group_and_emotion)

                    cv2.putText(img=image,
                                text=recommended_product,
                                org=(int(x), int(y)),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=1,
                                color=(0, 0, 255),
                                thickness=2)

            resized_image = cv2.resize(image, (1000, 700))
            cv2.imshow('Age Groupd and Emotion Recognition', resized_image)
            if cv2.waitKey(10) == ord('q'):
                break

        capture.release()
        cv2.destroyAllWindows()

    def _load_model(self, filepath: str) -> Any:
        """Loads a tensorflow model."""
        model = os.path.join(os.path.dirname(__file__), filepath)
        return load_model(model)

    def _recognize_age_group(self, pixels: np.ndarray) -> str:
        """Recognizes the age group of a person given the pixels of an image."""
        prediction = self._age_classifier.predict(pixels)
        age_group = AgeGroups.TABLE[prediction]
        return age_group

    def _recognize_emotion(self, pixels: np.ndarray) -> str:
        """Recognizes the emotion of a person given the pixels of an image."""
        predictions = self._emotion_classifier.predict(pixels)
        max_index = np.argmax(predictions[0])
        emotion = Emotions.LIST[max_index]
        return emotion

    def _recommend_product(self, age_group_and_emotion: str) -> str:
        """Recommends a product based on the age group and emotion of a person."""
        recommendation = Products.TABLE[age_group_and_emotion]
        return recommendation


# Example usage
if __name__ == '__main__':
    app = App(
        age_classifier_filepath='../model/emotion_classifier.h5',
        emotion_classifier_filepath='../model/age_classifier.h5',
    )
    app.run()
