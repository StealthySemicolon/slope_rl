import cv2
import keyboard
import numpy as np
import mss
import time

class SlopeGame:
    observation_space = [200, 150, 3]
    action_space = [-1, 3]
    def __init__(self):
        self.screen = mss.mss()
        self.monitor = {"top": 111, "left": 1, "width": 800, "height": 600}
        self.LEFT = 'left'
        self.RIGHT = 'right'
        self.NONE = 'up'
        self.reset_key = 'enter'
        self.pressed = self.NONE
    def get_screen(self):
        img = cv2.cvtColor(np.array(self.screen.grab(self.monitor)), cv2.COLOR_RGBA2RGB)
        return img
    def input_action(self, action):
        max_idx = action
        if max_idx == 1:
            cvrtd_action = self.LEFT
        if max_idx == 0:
            cvrtd_action = self.NONE
        if max_idx == 2:
            cvrtd_action = self.RIGHT
        keyboard.release(self.pressed)
        keyboard.press(cvrtd_action)
        self.pressed = cvrtd_action
    def reset(self):
        time.sleep(0.2)
        keyboard.release(self.pressed)
        self.pressed = self.NONE
        keyboard.press(self.pressed)
        keyboard.press_and_release(self.reset_key)
        return cv2.resize(self.get_screen(), (200, 150)) / 255
    def render(self):
        cv2.imshow("Frame", self.get_screen())
        cv2.waitKey(1)
    def check_done(self):
        img = self.get_screen()
        if img[300][275][0] == 255 and img[300][275][1] == 255 and img[300][275][2] == 255:
            return True
        else:
            return False
    def step(self, action):
        self.input_action(action)
        next_state = cv2.resize(self.get_screen(), (200, 150)) / 255
        done = self.check_done()
        reward = -100 if done else -1
        return next_state, reward, done, "hi"
