# 🖐️ Virtual Mouse - Gesture-Controlled Computer Interaction

## Overview
Virtual Mouse is an innovative Python application that allows you to control your computer using hand gestures through webcam input. This project leverages computer vision and machine learning to transform your hand movements into mouse interactions.

## 🚀 Features
- **Cursor Movement**: Move mouse cursor by pointing your index finger
- **Left Click**: Pinch index finger and thumb together
- **Right Click**: Pinch middle finger and thumb together
- **Double Click**: Two quick index finger-thumb pinches
- **Scrolling**: Align finger bases horizontally to activate scroll mode
- **Screenshot**: Open hand gesture followed by closing into a fist
- **Volume Control**: 
  - Thumbs up: Increase volume
  - Thumbs down: Decrease volume

## 🛠️ Prerequisites
- Python 3.7+
- Windows Operating System (for volume control)
- Webcam

## 📦 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/keni7689/virtual-mouse.git
cd virtual-mouse
```

### 2. Create Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
# Activate the virtual environment
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### Create requirements.txt
Create a `requirements.txt` file in your project directory with these contents:
```
opencv-python
mediapipe
pyautogui
numpy
pycaw
comtypes
```

## 🎮 How to Use

### Cursor Control
- Point your index finger to move the mouse cursor
- The cursor movement is smooth and follows your hand's position

### Clicking
- **Left Click**: Bring your index finger and thumb close together
- **Right Click**: Bring your middle finger and thumb close together
- **Double Click**: Two quick pinches of index finger and thumb

### Scrolling
- Align the bases of your fingers horizontally
- Move your hand up or down to scroll

### Screenshot
1. Open your hand fully (all fingers spread)
2. Close your hand into a fist
3. Screenshot will be saved in the `screenshots` folder

### Volume Control
- **Increase Volume**: Thumbs up gesture
- **Decrease Volume**: Thumbs down gesture

### Exiting the Application
- Press 'q' key to quit the Virtual Mouse application

## 🔧 Troubleshooting
- Ensure good lighting conditions
- Position your hand clearly in front of the webcam
- Make sure your webcam is working correctly
- Close other applications using the webcam



## 📝 License
Distributed under the MIT License. See `LICENSE` for more information.

## 📞 Contact
Yogesh Shejul - yogeshshejul54@gmail.com

Project Link: https://github.com/keni7689/Virtual-Mouse/tree/main

## 🙏 Acknowledgments
- [MediaPipe](https://mediapipe.dev/)
- [OpenCV](https://opencv.org/)
- [PyAutoGUI](https://pyautogui.readthedocs.io/)
