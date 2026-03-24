import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import time
import pygame
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = "face_landmarker.task"
ALARM_PATH = "alarm.wav"

pygame.mixer.init()
alarm_sound = pygame.mixer.Sound(ALARM_PATH)
alarm_sound.set_volume(0.5)

options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

detector = vision.FaceLandmarker.create_from_options(options)

LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

FACE_CONTOURS = [
    (10,338),(338,297),(297,332),(332,284),(284,251),(251,389),(389,356),
    (356,454),(454,323),(323,361),(361,288),(288,397),(397,365),(365,379),
    (379,378),(378,400),(400,377),(377,152),(152,148),(148,176),(176,149),
    (149,150),(150,136),(136,172),(172,58),(58,132),(132,93),(93,234),
    (234,127),(127,162),(162,21),(21,54),(54,103),(103,67),(67,109),(109,10),
    (46,53),(53,52),(52,65),(65,55),(55,70),(70,63),(63,105),(105,66),(66,107),(107,55),
    (276,283),(283,282),(282,295),(295,285),(285,300),(300,293),(293,334),(334,296),(296,336),
    (61,146),(146,91),(91,181),(181,84),(84,17),(17,314),(314,405),(405,321),
    (321,375),(375,291),(291,409),(409,270),(270,269),(269,267),(267,0),
    (0,37),(37,39),(39,40),(40,185),(185,61),
    (78,95),(95,88),(88,178),(178,87),(87,14),(14,317),(317,402),(402,318),
    (318,324),(324,308),(308,415),(415,310),(310,311),(311,312),(312,13),
    (13,82),(82,81),(81,80),(80,191),(191,78),
    (33,246),(246,161),(161,160),(160,159),(159,158),(158,157),(157,173),(173,133),
    (133,155),(155,154),(154,153),(153,145),(145,144),(144,163),(163,7),(7,33),
    (263,466),(466,388),(388,387),(387,386),(386,385),(385,384),(384,398),(398,362),
    (362,382),(382,381),(381,380),(380,374),(374,373),(373,390),(390,249),(249,263),
]

EYE_CONTOURS = [
    # Left eye
    (33,246),(246,161),(161,160),(160,159),(159,158),(158,157),(157,173),
    (173,133),(133,155),(155,154),(154,153),(153,145),(145,144),(144,163),
    (163,7),(7,33),
    # Right eye
    (263,466),(466,388),(388,387),(387,386),(386,385),(385,384),(384,398),
    (398,362),(362,382),(382,381),(381,380),(380,374),(374,373),(373,390),
    (390,249),(249,263),
]

NOSE_TIP  = 1
FOREHEAD  = 10
CHIN      = 152

EAR_THRESHOLD  = 0.24
DROWSY_TIME    = 2.0
HEAD_THRESHOLD = 0.15
BLINK_TIME     = 0.3

GREEN  = (0, 255, 0)
RED    = (0, 0, 255)
ORANGE = (0, 165, 255)
WHITE  = (255, 255, 255)
BLACK  = (0, 0, 0)
YELLOW = (0, 255, 255)

def get_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def get_ear(eye_landmarks, face_landmarks, width, height):
    points = []
    for idx in eye_landmarks:
        landmark = face_landmarks[idx]
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        points.append((x, y))

    p1, p2, p3, p4, p5, p6 = points

    vertical1  = get_distance(p2, p6)
    vertical2  = get_distance(p3, p5)
    horizontal = get_distance(p1, p4)
    epsilon = 1e-6
    ear = (vertical1 + vertical2) / (2.0 * max(horizontal, epsilon))
    return ear

def get_head_pitch(face_landmarks):
    nose     = face_landmarks[NOSE_TIP]
    forehead = face_landmarks[FOREHEAD]
    chin     = face_landmarks[CHIN]

    forehead_to_chin = abs(chin.y - forehead.y)

    if forehead_to_chin == 0:
        return 0

    nose_position = (nose.y - forehead.y) / forehead_to_chin
    return nose_position

def draw_panel(frame, x, y, w, h, alpha=0.6):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), BLACK, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_ui(frame, avg_ear, pitch_diff, is_drowsy, eyes_closing,
            head_drooping, drowsy_count, session_seconds, alarm_playing,
            eyes_closed_duration=0):

    height, width, _ = frame.shape

    if is_drowsy:
        flash = int(time.time() * 4) % 2 == 0
        if flash:
            cv2.rectangle(frame, (0, 0), (width, height), RED, 12)

    draw_panel(frame, 0, 0, width, 50)

    if is_drowsy:
        status_text  = "SLEEPY ALERT"
        status_color = RED
    elif head_drooping:
        status_text  = "WARNING! HEAD DROOPING"
        status_color = ORANGE
    elif eyes_closing and eyes_closed_duration > BLINK_TIME:
        status_text  = "WARNING! EYES CLOSING"
        status_color = ORANGE
    else:
        status_text  = "BE ALERT"
        status_color = GREEN

    cv2.putText(frame, status_text, (10, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

    mins = int(session_seconds // 60)
    secs = int(session_seconds % 60)
    cv2.putText(frame, f"{mins:02d}:{secs:02d}", (width - 90, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)

    panel_x = 10
    panel_y = 60
    draw_panel(frame, panel_x, panel_y, 220, 180)

    ear_color = RED if eyes_closing and eyes_closed_duration > BLINK_TIME else GREEN
    cv2.putText(frame, "EYE OPENNESS", (panel_x + 10, panel_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
    ear_pct = min(100, int((avg_ear / 0.4) * 100))
    cv2.putText(frame, f"{ear_pct}%", (panel_x + 10, panel_y + 55),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, ear_color, 2)

    pitch_color = RED if head_drooping else GREEN
    cv2.putText(frame, "HEAD POSITION", (panel_x + 10, panel_y + 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
    tilt_angle = round(pitch_diff * 100, 1)
    cv2.putText(frame, f"{tilt_angle}deg", (panel_x + 10, panel_y + 130),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, pitch_color, 2)
    
    draw_panel(frame, 0, height - 50, width, 50)

    cv2.putText(frame, f"SLEEPY EVENTS: {drowsy_count}", (10, height - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 2)

    alarm_text  = "ALARM: ON " if alarm_playing else "ALARM: OFF"
    alarm_color = RED if alarm_playing else WHITE
    cv2.putText(frame, alarm_text, (width - 160, height - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, alarm_color, 2)

    if is_drowsy:
        text      = "DONT SLEEP"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 4)[0]
        text_x    = (width - text_size[0]) // 2
        text_y    = (height + text_size[1]) // 2
        cv2.putText(frame, text, (text_x + 2, text_y + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, BLACK, 6)
        cv2.putText(frame, text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, RED, 4)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("could not open webcam!")
    exit()

print("✅ DontSleep running! Press Q to quit.")

eyes_closed_start  = None
head_drooped_start = None
is_drowsy          = False
alarm_playing      = False
baseline_pitch     = None
drowsy_count       = 0
session_start      = time.time()
calibrating        = True
calibration_start  = time.time()
calibration_values = []
CALIBRATION_TIME   = 3.0

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    results = detector.detect(mp_image)

    session_seconds = time.time() - session_start

    eyes_closing  = False
    head_drooping = False
    avg_ear       = 0.0
    pitch_diff    = 0.0

    if results.face_landmarks:

        face_landmarks = results.face_landmarks[0]

        left_ear  = get_ear(LEFT_EYE,  face_landmarks, width, height)
        right_ear = get_ear(RIGHT_EYE, face_landmarks, width, height)
        avg_ear   = (left_ear + right_ear) / 2.0

        head_pitch = get_head_pitch(face_landmarks)

        if calibrating:
            elapsed = time.time() - calibration_start
            remaining = int(CALIBRATION_TIME - elapsed) + 1

            calibration_values.append(head_pitch)

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (width, height), BLACK, -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            cv2.putText(frame, "CALIBRATING", (width//2 - 160, height//2 - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, GREEN, 3)

            if elapsed >= CALIBRATION_TIME:
                baseline_pitch = sum(calibration_values) / len(calibration_values)
                calibrating = False

                done_frame = frame.copy()
                cv2.imshow("dontSleep", done_frame)
                cv2.waitKey(1000)

            cv2.imshow("dontSleep", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        if baseline_pitch is None:
            baseline_pitch = head_pitch

        pitch_diff    = head_pitch - baseline_pitch

        eyes_closing  = avg_ear < EAR_THRESHOLD
        head_drooping = abs(pitch_diff) > HEAD_THRESHOLD
        eye_color  = RED if eyes_closing  else GREEN
        head_color = RED if head_drooping else GREEN

        for connection in FACE_CONTOURS:
            p1 = face_landmarks[connection[0]]
            p2 = face_landmarks[connection[1]]
            x1 = int(p1.x * width)
            y1 = int(p1.y * height)
            x2 = int(p2.x * width)
            y2 = int(p2.y * height)
            cv2.line(frame, (x1, y1), (x2, y2), GREEN, 1)

        for idx in range(468):
            landmark = face_landmarks[idx]
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            cv2.circle(frame, (x, y), 1, GREEN, -1)

        for connection in EYE_CONTOURS:
            p1 = face_landmarks[connection[0]]
            p2 = face_landmarks[connection[1]]
            x1 = int(p1.x * width)
            y1 = int(p1.y * height)
            x2 = int(p2.x * width)
            y2 = int(p2.y * height)
            cv2.line(frame, (x1, y1), (x2, y2), eye_color, 1)

        for idx in LEFT_EYE + RIGHT_EYE:
            landmark = face_landmarks[idx]
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            cv2.circle(frame, (x, y), 2, eye_color, -1)

        fx = int(face_landmarks[FOREHEAD].x * width)
        fy = int(face_landmarks[FOREHEAD].y * height)
        nx = int(face_landmarks[NOSE_TIP].x * width)
        ny = int(face_landmarks[NOSE_TIP].y * height)
        cx = int(face_landmarks[CHIN].x * width)
        cy = int(face_landmarks[CHIN].y * height)

        cv2.circle(frame, (fx, fy), 5, head_color, -1)
        cv2.circle(frame, (nx, ny), 5, head_color, -1)
        cv2.circle(frame, (cx, cy), 5, head_color, -1)
        cv2.line(frame, (fx, fy), (nx, ny), head_color, 1)
        cv2.line(frame, (nx, ny), (cx, cy), head_color, 1)

        if eyes_closing:
            if eyes_closed_start is None:
                eyes_closed_start = time.time()
            closed_duration = time.time() - eyes_closed_start
            if closed_duration >= DROWSY_TIME:
                if not is_drowsy:
                    drowsy_count += 1
                is_drowsy = True
        else:
            eyes_closed_start = None

        if head_drooping:
            if head_drooped_start is None:
                head_drooped_start = time.time()
            if time.time() - head_drooped_start >= DROWSY_TIME:
                if not is_drowsy:
                    drowsy_count += 1
                is_drowsy = True
        else:
            head_drooped_start = None

        if not eyes_closing and not head_drooping:
            is_drowsy = False

    else:
        if alarm_playing:
            alarm_sound.stop()
            alarm_playing = False

    if is_drowsy:
        if not alarm_playing:
            alarm_sound.play(-1)
            alarm_playing = True
    else:
        if alarm_playing:
            alarm_sound.stop()
            alarm_playing = False

    eyes_closed_duration = (time.time() - eyes_closed_start) if eyes_closed_start else 0

    draw_ui(frame, avg_ear, pitch_diff, is_drowsy, eyes_closing,
            head_drooping, drowsy_count, session_seconds, alarm_playing,
            eyes_closed_duration)

    cv2.imshow("dontSleep", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

alarm_sound.stop()
cap.release()
cv2.destroyAllWindows()
print(f"""session ended \ntotal drowsy events: {drowsy_count}""")