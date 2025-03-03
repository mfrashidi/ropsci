import json
import base64
import math
import random
import time
import logging
from typing import Dict, Tuple

import numpy as np
from channels.generic.websocket import AsyncWebsocketConsumer
from ultralytics import YOLO
import cv2
import mediapipe as mp

# Constants
ACCEPTANCE_THRESHOLD = 0.6

# Load the latest model
model = YOLO('model.pt')
model.to('mps')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Games
games = {}

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_tracking_confidence=0.7)

def overlay_with_transparency(background, overlay, X, Y):
    bh, bw = background.shape[:2]
    oh, ow = overlay.shape[:2]
    if X >= bw or Y >= bh:
        return background
    if ow + X <= 0 or oh + Y <= 0:
        return background
    if X + ow > bw:
        ow = bw - X
        overlay = overlay[:, :ow]
    if Y + oh > bh:
        oh = bh - Y
        overlay = overlay[:oh]

    if overlay.shape[2] == 3:
        background[Y:Y + oh, X:X + ow] = overlay
    else:
        mask = overlay[..., 3:] / 255.0
        background[Y:Y + oh, X:X + ow] = mask * overlay[..., :3] + (1 - mask) * background[Y:Y + oh, X:X + ow]

    return background

def remove_white_background(image):
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)

    lower_white = np.array([200, 200, 200, 255])
    upper_white = np.array([255, 255, 255, 255])
    mask = cv2.inRange(image, lower_white, upper_white)
    image[mask != 0] = [255, 255, 255, 0]
    return image

def decode_base64_image(base64_string: str) -> np.ndarray:
    """Decode a base64 string to an OpenCV image."""
    try:
        img_data = base64.b64decode(base64_string)
        np_array = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image")
        return img
    except Exception as e:
        logger.error(f"Error decoding base64 image: {e}")
        raise

def preprocess_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Preprocess the image for prediction."""
    image_resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    image_normalized = image_rgb / 255.0  # Normalize pixel values
    image_expanded = np.expand_dims(image_normalized, axis=0)
    return image_expanded

async def predict_image(base64_string: str, filename: str) -> Tuple[str, float]:
    """Predict the class of the ROI from base64 string."""
    try:
        # Decode the base64 string
        image = decode_base64_image(base64_string)

        # Preprocess the image for the model
        cv2.imwrite(filename, image)

        # Make prediction
        result = model.predict(
            source=filename,
            project='.',
            name='detected',
            exist_ok=True,
            save=False,
            show=False,
            show_labels=True,
            show_conf=True,
            conf=0.5
        )
        confidences = {
            i: 0 for i in CLASS_LABELS
        }
        for r in result:
            boxes = r.boxes
            for box in boxes:
                confidence = math.ceil((box.conf[0] * 100)) / 100
                classname = CLASS_LABELS[int(box.cls[0])]
                if confidence > confidences[classname]:
                    confidences[classname] = confidence

        return confidences
    except Exception as e:
        logger.error(f"Error predicting image: {e}")
        raise


crown = cv2.imread('crown.png')
crown = cv2.cvtColor(crown, cv2.COLOR_BGR2RGB)
crown = remove_white_background(crown)
crown_height, crown_width = crown.shape[:2]

mask = cv2.imread('cheater_red_mask.jpg')
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
mask = remove_white_background(mask)
mask_height, mask_width = mask.shape[:2]

def calculate_movement(current_landmarks, previous_landmarks):
    # Calculate the distance between current and previous landmarks
    distance = 0.0
    for current_hand_landmarks, previous_hand_landmarks in zip(current_landmarks, previous_landmarks):
        for current_landmark, previous_landmark in zip(current_hand_landmarks.landmark,
                                                       previous_hand_landmarks.landmark):
            dx = current_landmark.x - previous_landmark.x
            dy = current_landmark.y - previous_landmark.y
            distance += np.sqrt(dx * dx + dy * dy)  # Euclidean distance
    return distance

class GameConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.game_id = self.scope['url_route']['kwargs']['game_id']
        self.room_group_name = f"game_{self.game_id}"

        await self.accept()

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(
            self.room_group_name,
            self.channel_name
        )

    async def receive(self, text_data):
        try:
            if self.game_id not in games:
                games[self.game_id] = {
                    'rounds_suggestions': [],
                    'final_round': None,
                    'rounds': dict(),
                    'movements': dict(),
                    'zoom_level': 1.0,
                    'zoom_target': 1.0,
                    'scores': dict()
                }
            text_data_json = json.loads(text_data)
            message_type = text_data_json.get('type')
            player_name = text_data_json.get('player_name')
            if player_name not in games[self.game_id]['scores']:
                games[self.game_id]['scores'][player_name] = 0

            if message_type is None or player_name is None:
                logger.warning("Missing message type or player name")
                return

            if message_type == 'player_joined':
                await self.handle_player_joined(player_name)
            elif message_type == 'image_update':
                await self.handle_image_update(text_data_json, player_name)
            elif message_type == 'task_complete':
                await self.handle_task_complete(text_data_json, player_name)
            elif message_type == 'predict_image':
                await self.handle_predict_image(text_data_json, player_name)
            elif message_type == 'rounds_set':
                await self.handle_rounds_set(text_data_json, player_name)
            elif message_type == 'round_answer':
                await self.handle_round_answer(text_data_json, player_name)
            elif message_type == 'ready':
                await self.handle_ready(text_data_json, player_name)
            elif message_type == 'cheating':
                for player in games[self.game_id]['scores']:
                    if player != player_name:
                        games[self.game_id]['scores'][player] += 1
                await self.channel_layer.group_send(
                    self.room_group_name,
                    {
                        'type': 'cheating_message',
                        'player_name': player_name,
                        'scores': games[self.game_id]['scores']
                    }
                )
        except Exception as e:
            logger.error(f"Error processing message: {e}")

    async def handle_player_joined(self, player_name: str):
        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                'type': 'player_joined',
                'player_id': self.channel_name,
                'player_name': player_name,
            }
        )

    async def handle_image_update(self, text_data_json: Dict, player_name: str):
        receive_time = time.time()
        frame_data = text_data_json.get('frame')
        should_predict = text_data_json.get('should_predict')
        show_crown = text_data_json.get('show_crown')
        show_mask = text_data_json.get('show_mask')
        should_calculate_movement = text_data_json.get('calculate_movement')
        predictions = None

        image = None
        image_sanitized = None
        if show_crown or calculate_movement or show_mask:
            if frame_data.startswith("data:image"):
                image_sanitized = frame_data.split(",")[1]
            else:
                image_sanitized = frame_data
            image = decode_base64_image(image_sanitized)

        if show_crown:
            games[self.game_id]['zoom_target'] = 1.3
        else:
            games[self.game_id]['zoom_target'] = 1.0
        games[self.game_id]['zoom_level'] += (games[self.game_id]['zoom_target'] - games[self.game_id]['zoom_level']) * 0.05

        if show_crown:
            try:
                grayframe = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                faces = face_cascade.detectMultiScale(
                    grayframe,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(50, 50),
                )
                if len(faces) > 0:
                    x, y, w, h = faces[0]  # Use the first detected face

                    # Compute zoomed crop area
                    center_x, center_y = x + w // 2, y + h // 2
                    crop_w = int(image.shape[1] / games[self.game_id]['zoom_level'])
                    crop_h = int(image.shape[0] / games[self.game_id]['zoom_level'])
                    x1 = max(center_x - crop_w // 2, 0)
                    y1 = max(center_y - crop_h // 2, 0)
                    x2 = min(x1 + crop_w, image.shape[1])
                    y2 = min(y1 + crop_h, image.shape[0])

                    cropped_image = image[y1:y2, x1:x2]
                    image = cv2.resize(cropped_image, (image.shape[1], image.shape[0]))

                    x, y, w, h = faces[0]

                    scale = 1.75 * w / crown_width
                    overlay = cv2.resize(crown, (0, 0), fx=scale, fy=scale)
                    overlay_height, overlay_width = overlay.shape[:2]

                    x4 = x + (w - overlay_width) // 2
                    y4 = y - overlay_height // 1

                    if x4 < 0:
                        x4 = 0
                    if y4 < 0:
                        y4 = 0
                    if x4 + overlay_width > image.shape[1]:
                        overlay_width = image.shape[1] - x4
                        overlay = overlay[:, :overlay_width]
                    if y4 + overlay_height > image.shape[0]:
                        overlay_height = image.shape[0] - y4
                        overlay = overlay[:overlay_height]

                    image = overlay_with_transparency(image, overlay, x4, y4)

                _, buffer = cv2.imencode('.jpg', image)
                frame_data = f'data:image/png;base64,{base64.b64encode(buffer).decode("utf-8")}'

            except Exception as e:
                logger.error(f"Error processing image: {e}")

        if show_mask:
            try:
                start = time.time()
                grayframe = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                faces = face_cascade.detectMultiScale(
                    grayframe,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(50, 50),
                )

                for (x, y, w, h) in faces:
                    scale = 1.75 * w / mask_width
                    overlay = cv2.resize(mask, (0, 0), fx=scale, fy=scale)
                    overlay_height, overlay_width = overlay.shape[:2]

                    x4 = x + (w - overlay_width) // 2
                    y4 = y + (h - overlay_height) // 2

                    if x4 < 0:
                        x4 = 0
                    if y4 < 0:
                        y4 = 0
                    if x4 + overlay_width > image.shape[1]:
                        overlay_width = image.shape[1] - x4
                        overlay = overlay[:, :overlay_width]
                    if y4 + overlay_height > image.shape[0]:
                        overlay_height = image.shape[0] - y4
                        overlay = overlay[:overlay_height]

                    image = overlay_with_transparency(image, overlay, x4, y4)

                _, buffer = cv2.imencode('.jpg', image)
                frame_data = f'data:image/png;base64,{base64.b64encode(buffer).decode("utf-8")}'

                logger.info(f'Mask process time: {time.time() - start}')
            except Exception as e:
                logger.error(f"Error processing image: {e}")

        if should_predict:
            try:
                label, confidence = await predict_image(
                    image_sanitized,
                    f'{self.room_group_name}_{player_name}.jpg'
                )
                if label is not None:
                    predictions = {
                        'label': label,
                        'confidence': confidence
                    }
            except Exception as e:
                logger.error(f"Prediction failed: {e}")

        is_moving = None
        if player_name not in games[self.game_id]['movements']:
            games[self.game_id]['movements'][player_name] = {
                'current_landmarks': None,
                'stationary_count': 0
            }
        if should_calculate_movement:
            start = time.time()
            RGB_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = hands.process(RGB_frame)  # RGB

            stationary_count = games[self.game_id]['movements'][player_name]['stationary_count']
            previous_landmarks = games[self.game_id]['movements'][player_name]['current_landmarks']
            current_landmarks = None
            if result.multi_hand_landmarks:
                current_landmarks = result.multi_hand_landmarks
                if previous_landmarks is not None:
                    # Calculate movement
                    movement_distance = calculate_movement(current_landmarks, previous_landmarks)

                    # Define a threshold for movement detection
                    movement_threshold = 0.05  # Tune this value based on testing

                    if movement_distance > movement_threshold:
                        is_moving = True
                    else:
                        logger.info(f'{player_name} is stationary with hands')
                        logger.info(f'Stationary Count: {stationary_count}')
                        stationary_count += 1

            else:
                stationary_count = 0
                is_moving = False
                logger.info(f'{player_name} is stationary without hands')

            if stationary_count >= 2:
                is_moving = False
                stationary_count = 0
            if is_moving is True:
                stationary_count = 0

            logger.info(f'Movement Process Time: {time.time() - start}')
            logger.info(f'{player_name} is Moving? {is_moving}')
            games[self.game_id]['movements'][player_name] = {
                'current_landmarks': current_landmarks,
                'stationary_count': stationary_count
            }
        else:
            games[self.game_id]['movements'][player_name] = {
                'current_landmarks': None,
                'stationary_count': 0
            }

        await self.channel_layer.group_send(
            self.room_group_name,
            {
                'type': 'frame_message',
                'frame': frame_data,
                'predictions': predictions,
                'player_id': self.channel_name,
                'player_name': player_name,
                'receive_time': receive_time,
                'is_moving': is_moving
            }
        )

    async def handle_predict_image(self, text_data_json: Dict, player_name: str):
        receive_time = time.time()
        frame_data = text_data_json.get('frame')
        predictions = None

        if frame_data.startswith("data:image"):
            image_sanitized = frame_data.split(",")[1]
        else:
            image_sanitized = frame_data

        try:
            predictions = await predict_image(
                image_sanitized,
                f'{self.room_group_name}_{player_name}.jpg'
            )
        except Exception as e:
            logger.error(f"Prediction failed: {e}")

        await self.channel_layer.group_send(
            self.room_group_name,
            {
                'type': 'predict_message',
                'frame': None,
                'predictions': predictions,
                'player_id': self.channel_name,
                'player_name': player_name,
                'receive_time': receive_time,
            }
        )

    async def handle_task_complete(self, text_data_json: Dict, player_name: str):
        task = text_data_json.get('task')
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                'type': 'task_message',
                'task': task,
                'player_id': self.channel_name,
                'player_name': player_name,
            }
        )

    async def handle_rounds_set(self, text_data_json: Dict, player_name: str):
        rounds = text_data_json.get('rounds')
        games[self.game_id]['rounds_suggestions'].append(rounds)
        final_round = random.choice(games[self.game_id]['rounds_suggestions'])
        games[self.game_id]['final_round'] = final_round
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                'type': 'rounds_message',
                'final_round': final_round,
                'player_id': self.channel_name,
                'player_name': player_name,
            }
        )

    async def handle_round_answer(self, text_data_json: Dict, player_name: str):
        answer = text_data_json.get('answer')
        current_round = text_data_json.get('round')
        if self.game_id not in games:
            return

        if current_round not in games[self.game_id]['rounds']:
            games[self.game_id]['rounds'][current_round] = []

        games[self.game_id]['rounds'][current_round].append({
            'player_id': self.channel_name,
            'player_name': player_name,
            'answer': answer,
        })

        if len(games[self.game_id]['rounds'][current_round]) == 2:
            player1 = games[self.game_id]['rounds'][current_round][0]
            player2 = games[self.game_id]['rounds'][current_round][1]

            player1_answer = player1['answer']
            player2_answer = player2['answer']
            if player1_answer in ['cheated', 'none'] and player2_answer not in ['cheated', 'none']:
                games[self.game_id]['scores'][player2['player_name']] += 1
                await self.channel_layer.group_send(
                    self.room_group_name,
                    {
                        'type': 'cheating_message',
                        'player_name': player1['player_name'],
                        'scores': games[self.game_id]['scores']
                    }
                )
                logger.info(f'{player1["player_name"]} Cheated')
                logger.info(f'Round {current_round} Winner: {player2["player_name"]}')
                return
            elif player1_answer not in ['cheated', 'none'] and player2_answer in ['cheated', 'none']:
                games[self.game_id]['scores'][player1['player_name']] += 1
                await self.channel_layer.group_send(
                    self.room_group_name,
                    {
                        'type': 'cheating_message',
                        'player_name': player2['player_name'],
                        'scores': games[self.game_id]['scores']
                    }
                )
                logger.info(f'{player2["player_name"]} Cheated')
                logger.info(f'Round {current_round} Winner: {player1["player_name"]}')
                return

            logger.info(f'Player 1: {player1["answer"]}, Player 2: {player2["answer"]}')
            logger.info(f'Player 1 Answer: {player1["answer"]}, Player 2 Answer: {player2["answer"]}')
            if player1['answer'] == player2['answer']:
                winner = 'draw'
            elif player1['answer'] == 'rock' and player2['answer'] == 'scissors':
                winner = player1['player_name']
            elif player1['answer'] == 'scissors' and player2['answer'] == 'paper':
                winner = player1['player_name']
            elif player1['answer'] == 'paper' and player2['answer'] == 'rock':
                winner = player1['player_name']
            else:
                winner = player2['player_name']

            if winner == 'draw':
                games[self.game_id]['scores'][player1['player_name']] += 1
                games[self.game_id]['scores'][player2['player_name']] += 1
            else:
                games[self.game_id]['scores'][winner] += 1

            logger.info(f'Round {current_round} Winner: {winner}')
            await self.channel_layer.group_send(
                self.room_group_name,
                {
                    'type': 'round_answer_message',
                    'winner': winner,
                    'round': current_round,
                    'moves': [
                        {'player': player1['player_name'], 'move': player1['answer']},
                        {'player': player2['player_name'], 'move': player2['answer']}
                    ],
                    'scores': games[self.game_id]['scores'],
                    'player_id': self.channel_name,
                    'player_name': player_name,
                }
            )

    async def handle_ready(self, text_data_json: Dict, player_name: str):
        if self.game_id not in games:
            return

        current_round = text_data_json.get('round')
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                'type': 'ready_message',
                'player_id': self.channel_name,
                'player_name': player_name,
                'round': current_round
            }
        )


    async def frame_message(self, event):
        message = {
            'type': 'frame',
            'predictions': event['predictions'],
            'player_name': event['player_name'],
            'frame': event['frame'],
            'is_moving': event['is_moving']
        }
        await self.send(text_data=json.dumps(message))

    async def predict_message(self, event):
        logger.info(f'Predict {event["player_name"]} Process Time: {time.time() - event["receive_time"]}')
        message = {
            'type': 'predict',
            'predictions': event['predictions'],
            'player_name': event['player_name'],
            'frame': event['frame']
        }
        await self.send(text_data=json.dumps(message))

    async def player_joined(self, event):
        await self.send(text_data=json.dumps({
            'type': 'player_joined',
            'player_id': event['player_id'],
            'player_name': event['player_name'],
        }))

    async def task_message(self, event):
        await self.send(text_data=json.dumps({
            'type': 'task',
            'task': event['task'],
            'player_name': event['player_name'],
        }))

    async def rounds_message(self, event):
        await self.send(text_data=json.dumps({
            'type': 'rounds',
            'final_round': event['final_round']
        }))

    async def round_answer_message(self, event):
        await self.send(text_data=json.dumps({
            'type': 'round_answer',
            'winner': event['winner'],
            'round': event['round'],
            'moves': event['moves'],
            'scores': event['scores']
        }))

    async def ready_message(self, event):
        await self.send(text_data=json.dumps({
            'type': 'ready',
            'player_name': event['player_name'],
            'round': event['round']
        }))

    async def cheating_message(self, event):
        await self.send(text_data=json.dumps({
            'type': 'cheating',
            'cheater': event['player_name'],
            'scores': event['scores']
        }))