# ########################################################
# Stop-Signal choice trajectory task
# Pilot task
# ########################################################

# QUESTIONS TASK
# How good should it be? Is reaction and movement time enough? Or should it be more reactive?
# How to signal the stop?
# Make the trial stop based on the movement
# Instructions
# __________________________________________________________________
# Prepare environment

# Import packages and functions
from __future__ import absolute_import, division

import numpy as np
from psychopy import core, data, event, gui, visual

# __________________________________________________________________
# Get input from participant and prepare data file to store behavioural results
expName = "SSCTT"
expInfo = {"participant": "Use the same UNIQUE ID everywhere"}
dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
if dlg.OK == False:
    core.quit()
expInfo["date"] = data.getDateStr()  # add a simple timestamp
expInfo["expName"] = expName
p_num = expInfo["participant"]
dataFile = open(f"{p_num}.csv", "w")
dataFile.write("Trial,Targets,Condition,Stop,Time,Position_X,Position_Y\n")

# __________________________________________________________________
# Set up window, clock and seed
win = visual.Window(
    size=(1920, 1200),
    fullscr=False,
    screen=2,
    winType="pyglet",
    allowGUI=False,
    allowStencil=False,
    monitor="testMonitor",
    color="black",
    colorSpace="rgb",
    blendMode="avg",
    useFBO=True,
    units="height",
)
clock = core.Clock()
np.random.seed(41)  # Change seed in screening session

# __________________________________________________________________
# Define help functions

skip_with_space = visual.TextStim(
    win=win,
    text="Leertaste drücken um zu starten",
    pos=(0.0, -0.25),
    height=0.03,
    color="white",
)


def quit_and_store():
    dataFile.close()
    core.quit()


event.globalKeys.add(key="escape", func=quit_and_store)


def draw_and_wait(message, win):
    message.draw()
    skip_with_space.draw(win)
    win.flip()
    event.waitKeys()
    event.clearEvents()


def draw_objects(objects):
    for object in objects:
        object.draw()


def return_and_delete_rand_idx(array, max_int):
    idx = np.random.randint(max_int)
    x = array[idx]
    del array[idx]
    return x


# __________________________________________________________________
# Define objects and parameters
mouse = event.Mouse(visible=True)
# stop_text = 'STOP'
# stop_message = visual.TextStim(win=win, text=stop_text, color='white', height=0.1)

# Define experimental design parameters
n_trials = 180
objects = [
    visual.Rect(win=win, width=0.15, height=0.15, fillColor="aqua"),
    visual.Rect(win=win, width=0.15, height=0.15, fillColor="yellow"),
    visual.Rect(win=win, width=0.15, height=0.15, fillColor="fuchsia"),
    visual.TextStim(
        win=win, text="Keine \nBewegung", color="lime", height=0.05, bold=True
    ),
]

# Fixation cross
cross_pos = -0.4
cross = visual.ShapeStim(
    win=win,
    lineColor="gray",
    vertices=(
        (0, cross_pos - 0.04),
        (0, cross_pos + 0.04),
        (0, cross_pos),
        (-0.04, cross_pos),
        (0.04, cross_pos),
    ),
    closeShape=False,
)

# Positions
# Generate a list of equidistant target positions (from cross)
n_pos = 20
positions = []
for i in range(n_pos):
    p_x = 0.7 * np.cos((2 * i * np.pi) / n_pos)
    p_y = 0.7 * np.sin((2 * i * np.pi) / n_pos) - 0.4
    if p_y > -0.35:
        positions.append((p_x, p_y))
n_pos = len(positions)

# Generate the order of the conditions and the stop trials
conditions = np.repeat([0, 1, 2, 3], 45)
# Give a stop signal on 20 % of the trials
stop_trials = np.array([[1] * 9 + [0] * 36] * 4).flatten()
trials = np.arange(n_trials)
# Shuffle the trials
np.random.shuffle(trials)
conditions = conditions[trials]
stop_trials = stop_trials[trials]

# __________________________________________________________________
# Start experiment

# Show welcome screen
welcome_text = "Willkommen zu unserer Untersuchung!"
message = visual.TextStim(win=win, text=welcome_text, color="white", height=0.07)
draw_and_wait(message, win)

# __________________________________________________________________
# Loop over triala

for trial in range(n_trials):
    # Target objects, make sure that the position varies from one trial to the next
    condition = conditions[trial]
    positions_tmp = positions.copy()

    if condition < 2:  # 2 & 3 choice
        objects_trial = objects[:3]
        # Choose a random position for every object
        for object in objects_trial:
            object.pos = return_and_delete_rand_idx(positions_tmp, len(positions_tmp))
        # Delete random object if only 2 should be displayed
        if condition == 0:
            return_and_delete_rand_idx(objects_trial, 3)

    elif condition > 1:  # no movement option
        objects_trial = objects[:4]
        # Drop one rectangle
        return_and_delete_rand_idx(objects_trial, 3)
        # Choose a random position for every object
        for object in objects_trial:
            object.pos = return_and_delete_rand_idx(positions_tmp, len(positions_tmp))
        # If only 1 object, drop one
        if condition == 3:
            return_and_delete_rand_idx(objects_trial, 3)

    # Reset variables
    cross.lineColor = "white"
    stopped = 0

    # Pen has to get close to fixation cross
    while not np.sum(np.abs(mouse.getPos() - (0, cross_pos))) < 0.05:
        cross.draw()
        win.flip()

    # Change color of cross if pen is close enough
    cross.lineColor = "red"

    trial_time = core.Clock()
    while trial_time.getTime() < 2.5:
        cross.draw()
        # Save mouse pos
        mouse_pos = mouse.getPos()
        dataFile.write(
            "%i,%i, %i, %i, %.5f, %.5f, %.5f\n"
            % (
                trial,
                0,
                condition,
                stopped,
                trial_time.getTime(),
                mouse_pos[0],
                mouse_pos[1],
            )
        )
        win.flip()

    trial_time.reset()
    stopped = False
    while trial_time.getTime() < 2.3:
        # Draw objects and cross
        draw_objects(objects_trial)

        # Get mouse position
        mouse_pos = mouse.getPos()

        # Save
        dataFile.write(
            "%i, %i, %i, %i, %.5f, %.5f, %.5f\n"
            % (
                trial,
                1,
                condition,
                stopped,
                trial_time.getTime(),
                mouse_pos[0],
                mouse_pos[1],
            )
        )

        # Check whether movement has started to potentially trigger the stop signal
        if trial_time.getTime() > 0.55 and stop_trials[trial] and not stopped:
            stopped = True
            win.color = "red"

        win.flip()
    win.color = "black"

    print(trial)

# Show welcome screen
end_text = "Ende - Vielen Dank!"
message = visual.TextStim(win=win, text=end_text, color="white", height=0.07)
message.draw()
win.flip()

event.waitKeys()
event.clearEvents()

dataFile.close()

core.quit()
