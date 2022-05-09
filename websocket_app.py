import copy
import io
import itertools
import pathlib
import ssl
import tensorflow as tf
import cv2
import numpy
import websockets
import asyncio
import base64
import pickle
from PIL import Image
import numpy as np
import mediapipe as mp


ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
localhost_pem = pathlib.Path(__file__).with_name("cert.pem")
ssl_context.load_cert_chain(localhost_pem, pathlib.Path(__file__).with_name("key.pem"))


def keypoints_classifier(inputs_list):
    new_model = tf.keras.models.load_model('Models/model.h5', compile=False)
    inputs = np.asarray(inputs_list)
    inputs = tf.expand_dims(inputs, 0)
    prediction = new_model(inputs, training=False)
    label = int(numpy.argmax(np.squeeze(prediction)))
    with open('Models/labels.csv', 'r') as file:
        labels = file.readlines()
        if label < len(labels):
            return labels[label].rstrip('\n')


async def hello(websocket, path):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands
    # For webcam input:
    with mp_hands.Hands(
            model_complexity=0,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        async for data in websocket:
            st = data
            data = data[data.find('base64,') + len('base64,'):]
            decoded_data = base64.b64decode(data)
            img_arr = np.frombuffer(decoded_data, dtype=np.uint8)
            if len(img_arr) != 0:
                image = cv2.imdecode(img_arr, flags=cv2.IMREAD_COLOR)
                image = cv2.flip(image, 1)
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image)
                # Draw the hand annotations on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if results.multi_hand_landmarks:
                    landmarks_point = []
                    for hand_landmarks in results.multi_hand_landmarks:
                        image_width, image_height = image.shape[1], image.shape[0]
                        landmarks_point = []
                        for _, landmark in enumerate(hand_landmarks.landmark):
                            landmark_x = min(int(landmark.x * image_width), image_width - 1)
                            landmark_y = min(int(landmark.y * image_height), image_height - 1)
                            landmarks_point.append([landmark_x, landmark_y])
                    base_vector = [- landmarks_point[0][0], - landmarks_point[0][1]]
                    res = np.asarray(landmarks_point) + np.asarray(base_vector)
                    res = list(itertools.chain.from_iterable(res))
                    max_value = max(list(map(abs, res)))
                    res = list(map(lambda x: (x / max_value + 1) / 2 * 0.99 + 0.1, res))
                    name = keypoints_classifier(res)
                    cv2.putText(
                        image,
                        name,
                        (10, 50),
                        cv2.FONT_HERSHEY_COMPLEX,
                        1,
                        (0, 0, 255),
                        3)
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())
                buf = io.BytesIO()
                img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                img.save(buf, 'JPEG')
                # print(base64.b64encode(buf.getvalue()).decode('utf-8'))
                await websocket.send(base64.b64encode(buf.getvalue()).decode('utf-8'))

# ssl=ssl_context
start_server = websockets.serve(hello, "192.168.100.5", 8001, ssl=ssl_context)
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
