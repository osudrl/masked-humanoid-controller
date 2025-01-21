from inputs import get_gamepad
import math
import threading
import numpy as np
import time
import copy

class XboxController(object):
    MAX_TRIG_VAL = math.pow(2, 10)
    MAX_JOY_VAL = math.pow(2, 15)

    def __init__(self):

        self.LeftJoystickY = 0
        self.LeftJoystickX = 0
        self.RightJoystickY = 0
        self.RightJoystickX = 0
        self.LeftTrigger = 0
        self.RightTrigger = 0
        self.LeftBumper = 0
        self.RightBumper = 0
        self.A = 0
        self.X = 0
        self.Y = 0
        self.B = 0
        self.LeftThumb = 0
        self.RightThumb = 0
        self.Back = 0
        self.Start = 0
        self.DPadX = 0
        self.DPadY = 0

        self.A_pressed = False
        self.B_pressed = False
        self.X_pressed = False
        self.Y_pressed = False
        self.Back_pressed = False
        self.Start_pressed = False
        self.DPadX_pressed = False
        self.DPadY_pressed = False
        self.LeftBumper_pressed = False
        self.RightBumper_pressed = False

        self.dead_zone = 0.1

        self._monitor_thread = threading.Thread(target=self._monitor_controller, args=())
        self._monitor_thread.daemon = True
        self._monitor_thread.start()

    def apply_deadzone(self, value):
        return 0 if abs(value) < self.dead_zone else value

    def _monitor_controller(self):
        while True:
            events = get_gamepad()
            for event in events:
                if event.code == 'ABS_Y':
                    self.LeftJoystickY = self.apply_deadzone(-event.state / XboxController.MAX_JOY_VAL) # normalize between -1 and 1
                elif event.code == 'ABS_X':
                    self.LeftJoystickX = self.apply_deadzone(event.state / XboxController.MAX_JOY_VAL) # normalize between -1 and 1
                elif event.code == 'ABS_RY':
                    self.RightJoystickY = self.apply_deadzone(-event.state / XboxController.MAX_JOY_VAL) # normalize between -1 and 1
                elif event.code == 'ABS_RX':
                    self.RightJoystickX = self.apply_deadzone(event.state / XboxController.MAX_JOY_VAL) # normalize between -1 and 1
                elif event.code == 'ABS_Z':
                    self.LeftTrigger = self.apply_deadzone(event.state / XboxController.MAX_TRIG_VAL) # normalize between 0 and 1
                elif event.code == 'ABS_RZ':
                    self.RightTrigger = event.state / XboxController.MAX_TRIG_VAL # normalize between 0 and 1
                elif event.code == 'BTN_TL':
                    self.LeftBumper = event.state
                elif event.code == 'BTN_TR':
                    self.RightBumper = event.state
                elif event.code == 'BTN_SOUTH':
                    self.A = event.state
                elif event.code == 'BTN_NORTH':
                    self.X = event.state #previously switched with X
                elif event.code == 'BTN_WEST':
                    self.Y = event.state #previously switched with Y
                elif event.code == 'BTN_EAST':
                    self.B = event.state
                elif event.code == 'BTN_THUMBL':
                    self.LeftThumb = event.state
                elif event.code == 'BTN_THUMBR':
                    self.RightThumb = event.state
                elif event.code == 'BTN_SELECT':
                    self.Back = event.state
                elif event.code == 'BTN_START':
                    self.Start = event.state
                elif event.code == 'ABS_HAT0X':
                    self.DPadX = event.state
                elif event.code == 'ABS_HAT0Y':
                    self.DPadY = -event.state
            time.sleep(1/250)


if __name__ == '__main__':
    joy = XboxController()
    while True:
        time.sleep(1/50)
        print(joy.LeftJoystickX, joy.LeftJoystickY, joy.RightJoystickX, joy.RightJoystickY,
              joy.A, joy.B, joy.Y, joy.X, joy.LeftBumper, joy.LeftTrigger, joy.RightBumper, joy.RightTrigger, joy.Start, joy.Back)
