"""Real-time facial emotion detection."""
import os

import cv2
import numpy as np
from tensorflow.keras.models import load_model


class App:
    """OpenCV Video App for facial age and emotion recognition.

    Attributes:
        _emotion_recognition_model_filepath: The filepath to the tensorflow emotion model.
        _age_recognition_model_filepath: The filepath to the tensorflow age model.
    """
    _EMOTIONS = [
        'neutral',
        'happy',
        'surprised',
        'sad',
        'angry',
        'disgusted',
        'fearful',
    ]

    _AGE_LABELS = {
        0: 'Child',
        1: 'Young Adult',
        2: 'Adult',
        3: 'Senior',
    }

    _PRODUCT_RECOMMENDATIONS = {'(Child, happy)': '...'}

    def __init__(
        self,
        emotion_recognition_model_filepath: str,
        age_recognition_model_filepath: str,
    ) -> None:
        """Initialize the application."""
        self._emotion_recognition_model = load_model(emotion_recognition_model_filepath)
        self._age_recognition_model = load_model(age_recognition_model_filepath)

    def run(self) -> None:
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
                for (x, y, w, h) in faces_detected:
                    cv2.rectangle(image, (x, y), (x + w, y + h), (153, 153, 153), thickness=2)
                    roi_gray = gray_image[y:y + w, x:x + h]
                    roi_gray = cv2.resize(roi_gray, (48, 48))
                    img_pixels = image.img_to_array(roi_gray)
                    img_pixels = np.expand_dims(img_pixels, axis=0)
                    img_pixels /= 255.0

                    predictions = self.__model.predict(img_pixels)
                    confidence = predictions[0].argsort()[-2:][::-1]

                    i = int(confidence[0])
                    j = int(confidence[1])
                    predicted_emotion_max = self._EMOTIONS[i]
                    predicted_emotion_sec = self._EMOTIONS[j]
                    text = [f'{predicted_emotion_max}: {predictions[0][i]*100:.2f} %']

                    if predictions[0][i] < 0.9:
                        text.append(f'{predicted_emotion_sec}: {predictions[0][j]*100:.2f} %')

                    y0, dy = int(y) - 30, 30
                    for i, line in enumerate(text):
                        iy = y0 + i * dy
                        cv2.putText(image, line, (int(x), iy), \
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 171, 240), 2)

                    resized_img = cv2.resize(image, (1000, 700))
            else:
                resized_img = cv2.resize(capture.read()[1], (1000, 700))

            cv2.imshow('Facial Emotion Recognition', resized_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        capture.release()
        cv2.destroyAllWindows()

    def _recognize_age(self, faces):
        pass

    def _recognize_emotion(self, faces):
        pass


# Example usage
if __name__ == '__main__':
    model_path = os.path.join(os.path.dirname(__file__), '../model/model_84_test_acc.h5')
    print(model_path)
    app = App(model_path, model_path)
    app.run()
