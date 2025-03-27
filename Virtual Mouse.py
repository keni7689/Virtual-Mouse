import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import math
import time
import os
import pycaw.pycaw as pycaw
import comtypes

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
screen_width, screen_height = pyautogui.size()

# Screenshot variables
screenshot_open_hand_state = False
screenshot_cooldown = 2.0
last_screenshot_time = 0

# Volume control variables
def get_volume_control():
    try:
        # Get the default audio endpoint (speaker)
        devices = pycaw.AudioUtilities.GetSpeakers()
        interface = devices.Activate(
            pycaw.IAudioEndpointVolume._iid_, 
            pycaw.CLSCTX_ALL, 
            None
        )
        return interface.QueryInterface(pycaw.IAudioEndpointVolume)
    except Exception as e:
        print(f"Could not get volume control: {e}")
        return None

volume_control = get_volume_control()
volume_cooldown = 1.0
last_volume_change_time = 0

CLICK_THRESHOLD = 40
click_cooldown = 1.0
last_click_time = 0

double_click_threshold = 0.5
last_left_click_time = 0

# Improved smoothing parameters
smoothing = 0.7  # Increased smoothing for more stable cursor
frame_smoothing = 0.3
prev_x, prev_y = None, None
mouse_speed_factor = 1.5  # Adjust cursor sensitivity

threshold_x_std = 20
scroll_threshold = 5
scroll_factor = 10
prev_avg_y = None

cv2.namedWindow("Virtual Mouse", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Virtual Mouse", 1000, 800)

def is_hand_open(hand_landmarks, h, w):
    """Check if hand is fully open by checking finger tip and base distances"""
    # Get landmarks for finger tips and base
    landmarks_to_check = [
        (8, 5),   # Index finger
        (12, 9),  # Middle finger
        (16, 13), # Ring finger
        (20, 17)  # Pinky
    ]
    
    open_fingers = 0
    for tip_idx, base_idx in landmarks_to_check:
        tip = hand_landmarks.landmark[tip_idx]
        base = hand_landmarks.landmark[base_idx]
        
        # Convert to pixel coordinates
        tip_x, tip_y = int(tip.x * w), int(tip.y * h)
        base_x, base_y = int(base.x * w), int(base.y * h)
        
        # Check if tip is significantly above base
        if tip_y < base_y - 30:
            open_fingers += 1
    
    return open_fingers >= 4  # Most fingers should be open

def is_hand_closed(hand_landmarks, h, w):
    """Check if hand is closed into a fist"""
    landmarks_to_check = [8, 12, 16, 20]  # Finger tip landmarks
    
    closed_fingers = 0
    base_y = hand_landmarks.landmark[0].y * h  # Wrist y coordinate
    
    for tip_idx in landmarks_to_check:
        tip = hand_landmarks.landmark[tip_idx]
        tip_y = tip.y * h
        
        # Check if tip is below or near base
        if tip_y > base_y:
            closed_fingers += 1
    
    return closed_fingers >= 4  # Most fingers should be closed

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror view.
    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    scroll_mode = False
    current_time = time.time()

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            thumb = hand_landmarks.landmark[4]
            index_finger = hand_landmarks.landmark[8]
            middle_finger = hand_landmarks.landmark[12]

            index_base = hand_landmarks.landmark[5]
            middle_base = hand_landmarks.landmark[9]
            ring_base = hand_landmarks.landmark[13]
            pinky_base = hand_landmarks.landmark[17]

            thumb_x, thumb_y = int(thumb.x * w), int(thumb.y * h)
            index_x, index_y = int(index_finger.x * w), int(index_finger.y * h)
            middle_x, middle_y = int(middle_finger.x * w), int(middle_finger.y * h)

            index_base_x, index_base_y = int(index_base.x * w), int(index_base.y * h)
            middle_base_x, middle_base_y = int(middle_base.x * w), int(middle_base.y * h)
            ring_base_x, ring_base_y = int(ring_base.x * w), int(ring_base.y * h)
            pinky_base_x, pinky_base_y = int(pinky_base.x * w), int(pinky_base.y * h)

            dist_index_thumb = math.hypot(index_x - thumb_x, index_y - thumb_y)
            dist_middle_thumb = math.hypot(middle_x - thumb_x, middle_y - thumb_y)

            # Screenshot Feature
            if is_hand_open(hand_landmarks, h, w):
                screenshot_open_hand_state = True
            
            if (screenshot_open_hand_state and 
                is_hand_closed(hand_landmarks, h, w) and 
                (current_time - last_screenshot_time) > screenshot_cooldown):
                # Create screenshots directory if it doesn't exist
                os.makedirs('screenshots', exist_ok=True)
                screenshot_path = f'screenshots/screenshot_{int(current_time)}.png'
                pyautogui.screenshot(screenshot_path)
                cv2.putText(frame, "Screenshot Taken!", (50, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                last_screenshot_time = current_time
                screenshot_open_hand_state = False

            # Volume Control Feature
            # Thumbs Up - Increase Volume
            if (thumb.y < index_base.y and 
                thumb.x > index_base.x and 
                (current_time - last_volume_change_time) > volume_cooldown):
                if volume_control:
                    current_volume = volume_control.GetMasterVolumeLevelScalar()
                    new_volume = min(1.0, current_volume + 0.4)
                    volume_control.SetMasterVolumeLevelScalar(new_volume, None)
                    cv2.putText(frame, f"Volume Up: {int(new_volume*100)}%", (50, 150), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    last_volume_change_time = current_time

            # Thumbs Down - Decrease Volume
            elif (thumb.y > index_base.y and 
                  thumb.x > index_base.x and 
                  (current_time - last_volume_change_time) > volume_cooldown):
                if volume_control:
                    current_volume = volume_control.GetMasterVolumeLevelScalar()
                    new_volume = max(0.0, current_volume - 0.4)
                    volume_control.SetMasterVolumeLevelScalar(new_volume, None)
                    cv2.putText(frame, f"Volume Down: {int(new_volume*100)}%", (50, 150), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    last_volume_change_time = current_time

            # Scroll mode detection
            fingers_x = np.array([index_base_x, middle_base_x, ring_base_x, pinky_base_x])
            if np.std(fingers_x) < threshold_x_std:
                scroll_mode = True
                avg_y = np.mean([index_base_y, middle_base_y, ring_base_y, pinky_base_y])
                if prev_avg_y is not None:
                    diff = avg_y - prev_avg_y
                    if abs(diff) > scroll_threshold:
                        pyautogui.scroll(-int(diff * scroll_factor))
                prev_avg_y = avg_y
            else:
                scroll_mode = False
                prev_avg_y = None

            # Improved Mouse Movement
            if not scroll_mode:
                # More advanced interpolation with frame smoothing
                screen_x = np.interp(index_x, [0, w], [0, screen_width])
                screen_y = np.interp(index_y, [0, h], [0, screen_height])

                # Enhanced smoothing technique
                if prev_x is None:
                    prev_x, prev_y = screen_x, screen_y
                else:
                    # Apply more advanced smoothing with frame-based interpolation
                    smoothed_x = prev_x + frame_smoothing * (screen_x - prev_x) * mouse_speed_factor
                    smoothed_y = prev_y + frame_smoothing * (screen_y - prev_y) * mouse_speed_factor
                    
                    screen_x, screen_y = smoothed_x, smoothed_y
                    prev_x, prev_y = screen_x, screen_y

                pyautogui.moveTo(screen_x, screen_y, duration=0.01)

                # Left Click
                if dist_index_thumb < CLICK_THRESHOLD and (current_time - last_click_time) > click_cooldown:
                    # Check for double click
                    if (current_time - last_left_click_time) < double_click_threshold:
                        pyautogui.doubleClick()
                        cv2.putText(frame, "Double Left Click!", (index_x, index_y - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        last_left_click_time = 0
                    else:
                        pyautogui.click()
                        cv2.putText(frame, "Left Click!", (index_x, index_y - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        last_left_click_time = current_time
                    last_click_time = current_time

                # Right Click
                elif dist_middle_thumb < CLICK_THRESHOLD and (current_time - last_click_time) > click_cooldown:
                    pyautogui.rightClick()
                    last_click_time = current_time
                    cv2.putText(frame, "Right Click!", (middle_x, middle_y - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Virtual Mouse", frame)
    if cv2.getWindowProperty("Virtual Mouse", cv2.WND_PROP_VISIBLE) < 1:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()