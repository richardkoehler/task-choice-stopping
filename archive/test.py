#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time

import numpy as np
from psychopy import core, logging, visual
from psychopy.hardware import keyboard

# core.checkPygletDuringWait = False
logging.console.setLevel(logging.WARNING)


blue = np.array([0, 0, 1])
grey = np.array([0.00392157, 0.00392157, 0.00392157])


def change_color(win, log=False):
    win.color = blue if np.array_equal(win.color, grey) else grey
    if log:
        print("Changed color to %s" % win.color)


kb = keyboard.KeyboardDevice(muteOutsidePsychopy=False)

win = visual.Window(color="gray")
text = visual.TextStim(win, text="Press C to change color,\n CTRL + Q to quit.")


def response(message: keyboard.KeyPress):
    if message.duration is None:
        return False
    return message.name == "c"


kb.registerCallback(
    response=response,  # "c",
    func=change_color,
    args=[win],
    kwargs={"log": True},
)

kb.registerCallback(response="q", func=core.quit)

# event.globalKeys.add(
#     key="c",
#     func=change_color,
#     func_args=[win],
#     func_kwargs=dict(log=True),
# )

# # Global event key to change window background color.
# event.globalKeys.add(
#     key="c",
#     func=change_color,
#     func_args=[win],
#     func_kwargs=dict(log=True),
#     name="change window color",
# )

# # Global event key (with modifier) to quit the experiment ("shutdown key").
# event.globalKeys.add(key="q", modifiers=["ctrl"], func=psychopy.core.quit)


while True:
    keys = kb.getKeys()
    for key in keys:
        print(key.name, key.rt, key.duration)
    text.draw()
    win.flip()
    logging.flush()
    time.sleep(0.1)
