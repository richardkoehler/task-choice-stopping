#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import time

from psychopy import core, logging
from psychopy.hardware import keyboard

# core.checkPygletDuringWait = False
logging.console.setLevel(logging.DEBUG)


def change_color(win, log=False):
    print(win.color)
    win.color = "blue" if win.color == "gray" else "gray"
    if log:
        print("Changed color to %s" % win.color)


kb = keyboard.KeyboardDevice(muteOutsidePsychopy=False)

kb.registerCallback(response="q", func=core.quit)

start_time = time.time()
while time.time() - start_time < 10:
    keys = kb.getKeys(waitRelease=True)
    for key in keys:
        print(key.name, key.rt, key.duration)
    time.sleep(1.0)

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
