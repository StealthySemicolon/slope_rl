import cv2
import keyboard
import numpy as np

screen = cv2.VideoCapture(0)

class SlopeGame:
    def __init__(self):
        self.screen = cv2.VideoCapture(0)
        self.LEFT = 'left'
        self.RIGHT = 'right'
        self.NONE = 'up'
        self.pressed = self.NONE
    def conv_action_to_input(self, action):
        max_idx = np.argmax(action)
        if max_idx == 0:
            return self.LEFT
        if max_idx == 1:
            return self.NONE
        if max_idx == 2:
            return self.RIGHT
    def act(self, cvrtd_action):
        keyboard.release(self.pressed)
        keyboard.press(cvrtd_action)
        self.pressed = cvrtd_action
        