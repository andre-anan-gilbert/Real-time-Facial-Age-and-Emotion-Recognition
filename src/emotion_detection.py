"""Real-time facial emotion detection."""
import os

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


class App:

    def __init__(self, model) -> None:
        """Initialize the application."""
        self.__model = model

    def run(self) -> None:
        cv2.ocl.setUseOpenCL(False)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 5)

        while True:
            ret, img = cap.read()
            if not ret:
                break

            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces_detected = face_cascade.detectMultiScale(gray_img,
                                                           1.1,
                                                           6,
                                                           minSize=(150, 150))

            if type(faces_detected) != tuple:
                for (x, y, w, h) in faces_detected:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (153, 153, 153),
                                  thickness=2)
                    roi_gray = gray_img[y:y + w, x:x + h]
                    roi_gray = cv2.resize(roi_gray, (48, 48))
                    img_pixels = image.img_to_array(roi_gray)
                    img_pixels = np.expand_dims(img_pixels, axis=0)
                    img_pixels /= 255.0

                    predictions = self.__model.predict(img_pixels)
                    conf = predictions[0].argsort()[-2:][::-1]

                    i = int(conf[0])
                    j = int(conf[1])

                    emotions = [
                        'neutral', 'happy', 'surprised', 'sad', 'angry',
                        'disgusted', 'fearful'
                    ]
                    predicted_emotion_max = emotions[i]
                    predicted_emotion_sec = emotions[j]

                    text = [
                        f'{predicted_emotion_max}: {predictions[0][i]*100:.2f} %'
                    ]

                    if predictions[0][i] < 0.9:
                        text.append(
                            f'{predicted_emotion_sec}: {predictions[0][j]*100:.2f} %'
                        )

                    y0, dy = int(y) - 30, 30
                    for i, line in enumerate(text):
                        iy = y0 + i * dy
                        cv2.putText(img, line, (int(x), iy),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                    (0, 171, 240), 2)

                    resized_img = cv2.resize(img, (1000, 700))
                    # cv2.imshow('Facial Emotion Recognition', resized_img)
            else:
                resized_img = cv2.resize(cap.read()[1], (1000, 700))
            cv2.imshow('Facial Emotion Recognition', resized_img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


# Example usage
if __name__ == '__main__':
    model_path = os.path.join(os.path.dirname(__file__),
                              '../model/model_84_test_acc.h5')
    print(model_path)
    app = App(model=load_model(model_path))
    app.run()
