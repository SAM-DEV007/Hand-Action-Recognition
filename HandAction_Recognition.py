import mediapipe as mp
import tensorflow as tf
import numpy as np

import cv2
import copy
import itertools
import os
import time


def landmark_list(image, landmarks):
    '''Prepares the landmark points list'''

    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_data(landmark_list):
    '''Pre processes the data'''

    temp_landmark_list = copy.deepcopy(landmark_list)

    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return (n / max_value) or 0.0001

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def pad(list_):
    '''Pads the list for prediction'''

    return tf.keras.utils.pad_sequences(list_, padding="post", dtype='float32', maxlen=1000)


def main():
    '''The main function to start the filter'''

    cam = cv2.VideoCapture(0)

    mphands = mp.solutions.hands
    hands = mphands.Hands()

    model_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__))) + '\\Model\\Model_Data\\', 'Model.h5')
    model = tf.keras.models.load_model(model_path, compile=False)

    handLMs = None

    pred = False
    start = False
    timer = 0
    write_list = list()

    gesture = ['Waving', 'Stickman Walking', 'Palm to Fist', 'Other']
    prediction = None
    
    while True:
        if not start:
            timer = time.time()
            start = True

        _, frame = cam.read()
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        hand_landmarks = result.multi_hand_landmarks

        if hand_landmarks:
            handLMs = hand_landmarks[0]
            if handLMs:
                landmark = landmark_list(frame, handLMs)
                real_landmark = pre_process_data(landmark)

                if timer:
                    if time.time() - timer > 2:
                        pred = True
                        timer = 0

                if start:
                    write_list.extend(real_landmark)

                if pred:
                    pred = False
                    start = False

                    pred_ = np.squeeze(model.predict(pad([write_list]), verbose=0))
                    prediction = np.argmax(pred_)

                    write_list = list()

        if prediction is not None:
            cv2.putText(frame, f'Prediction: {gesture[prediction]} - Probability: {str(np.round(pred_[prediction], 4)*100)[:4]}%', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA, False)
        cv2.imshow('Cam', frame)

        # Closes the window
        # Q button
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Esc button
        if cv2.waitKey(1) == 27:
            break

        # X button on the top of the window
        if cv2.getWindowProperty('Cam', cv2.WND_PROP_VISIBLE) < 1:
            break
    
    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
