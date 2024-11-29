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
from __future__ import absolute_import, annotations, division

import pathlib
from io import TextIOWrapper

import numpy as np
import psychopy
from packaging.version import Version
from psychopy import core, data, event, gui, logging, visual
from psychopy.constants import FINISHED
from psychopy.hardware import keyboard

logging.console.setLevel(logging.INFO)
NEW_PSYCHOPY = Version(psychopy.__version__) >= Version("2024.3")


def setupExperiment() -> dict[str, str]:
    # __________________________________________________________________
    # Get input from participant and prepare data file to store behavioural results
    expName = "SSCTT"
    expInfo = {"participant": "Use the same UNIQUE ID everywhere"}
    dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
    if not dlg.OK:
        core.quit()
    expInfo["date"] = data.getDateStr(
        format="%Y-%m-%d %Hh%M.%S.%f %z", fractionalSecondDigits=6
    )
    expInfo["expName"] = expName
    return expInfo


def setupLogging(filename: pathlib.Path):
    """
    Setup a log file and tell it what level to log at.

    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.

    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # this outputs to the screen, not a file
    logging.console.setLevel(logging.ERROR)
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename.with_suffix(".log"), level=logging.DEBUG)

    return logFile


def setupWindow(expInfo: dict[str, str]) -> visual.Window:
    """
    Setup the window for the experiment.

    Parameters
    ==========
    expInfo : dict
        Dictionary of experiment information.

    Returns
    ==========
    psychopy.visual.Window
        Window to display stimuli in.
    """
    win = visual.Window(
        size=(1920, 1200),
        fullscr=False,
        screen=3,
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
    # store frame rate of monitor if we can measure it
    expInfo["frameRate"] = win.getActualFrameRate()
    return win


def endExperiment(thisExp: data.ExperimentHandler, win: visual.Window | None = None):
    """
    End this experiment, performing final shut down operations.

    This function does NOT close the window or end the Python process - use `quit` for this.

    Parameters
    ==========
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip()
        # and win.timeOnFlip() tasks get executed
        win.flip()
        # mark experiment handler as finished
    thisExp.status = FINISHED
    logging.flush()


def quit(thisExp: data.ExperimentHandler, win: visual.Window | None = None):
    """
    Fully quit, closing the window and ending the Python process.

    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip()
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    core.quit()


def draw_and_wait(
    skip_with_space: visual.TextStim, message: visual.TextStim, win: visual.Window
):
    message.draw()
    skip_with_space.draw(win)
    win.flip()
    event.waitKeys(keyList=["space"])
    event.clearEvents()


def draw_objects(objects):
    for object in objects:
        object.draw()


def return_and_delete_rand_idx(array, max_int):
    idx = np.random.randint(max_int)
    x = array[idx]
    del array[idx]
    return x


def store_and_quit(
    thisExp: data.ExperimentHandler, win: visual.Window, dataFile: TextIOWrapper
):
    dataFile.close()
    endExperiment(thisExp=thisExp, win=win)
    quit(thisExp=thisExp, win=win)


def run(
    thisExp: data.ExperimentHandler, win: visual.Window, dataFile: TextIOWrapper
) -> None:
    np.random.seed(41)  # Change seed in screening session

    func_kwargs = {"win": win, "thisExp": thisExp, "dataFile": dataFile}
    if NEW_PSYCHOPY:
        print("Using new keyboard class")
        kb = keyboard.KeyboardDevice(muteOutsidePsychopy=False)
        kb.registerCallback(
            response="escape",
            func=store_and_quit,
            kwargs=func_kwargs,
        )
    else:
        print("Using old keyboard class")
        event.globalKeys.add(key="escape", func=store_and_quit, func_kwargs=func_kwargs)

    skip_with_space = visual.TextStim(
        win=win,
        text="Leertaste drücken um zu starten",
        pos=(0.0, -0.25),
        height=0.03,
        color="white",
    )

    # Define objects and parameters
    mouse = event.Mouse(visible=True)
    # stop_text = 'STOP'
    # stop_message = visual.TextStim(win=win, text=stop_text, color='white', height=0.1)

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

    # Define experimental design parameters
    trials_per_block = 45
    n_blocks = 4
    n_trials = trials_per_block * n_blocks  # 180 trials in total

    trials = np.arange(n_trials)
    # Shuffle the trials
    np.random.shuffle(trials)
    # Generate the order of the conditions and the stop trials
    conditions = np.repeat(np.arange(n_blocks), trials_per_block)
    # Shuffle the order of the conditions
    conditions = conditions[trials]

    # Give a stop signal on 20 % of the trials
    n_stop_trials = trials_per_block / 5
    if not n_stop_trials.is_integer():
        raise ValueError("Number of trials per block must be divisible by 5.")
    n_stop_trials = int(n_stop_trials)
    n_go_trials = trials_per_block - n_stop_trials
    stop_trials = np.array(
        [[1] * n_stop_trials + [0] * n_go_trials] * n_blocks
    ).flatten()
    stop_trials = stop_trials[trials]

    # __________________________________________________________________
    # Start experiment

    # Show welcome screen
    welcome_text = "Willkommen zu unserer Untersuchung!"
    message = visual.TextStim(win=win, text=welcome_text, color="white", height=0.07)
    draw_and_wait(skip_with_space, message, win)

    dataFile.write("Trial,Targets,Condition,Stop,Time,Position_X,Position_Y\n")

    # __________________________________________________________________
    # Loop over trials

    for trial, condition in zip(range(n_trials), conditions):
        # Target objects, make sure that the position varies from one trial to the next
        positions_tmp = positions.copy()

        if condition <= 1:  # 2 & 3 choice
            objects_trial = objects[:3]
            # Choose a random position for every object
            for object in objects_trial:
                object.pos = return_and_delete_rand_idx(
                    positions_tmp, len(positions_tmp)
                )
            # Delete random object if only 2 should be displayed
            if condition == 0:
                return_and_delete_rand_idx(objects_trial, 3)

        elif condition > 1:  # no movement option
            objects_trial = objects[:4]
            # Drop one rectangle
            return_and_delete_rand_idx(objects_trial, 3)
            # Choose a random position for every object
            for object in objects_trial:
                object.pos = return_and_delete_rand_idx(
                    positions_tmp, len(positions_tmp)
                )
            # If only 1 object, drop one
            if condition == 3:
                return_and_delete_rand_idx(objects_trial, 3)

        # Reset variables
        cross.lineColor = "white"
        stopped = 0

        # Pen has to get close to fixation cross
        while not np.sum(np.abs(mouse.getPos() - (0, cross_pos))) < 0.05:
            if NEW_PSYCHOPY:
                kb.getKeys()
            cross.draw()
            win.flip()

        # Change color of cross if pen is close enough
        cross.lineColor = "red"

        trial_time = core.Clock()
        while trial_time.getTime() < 1.2:
            if NEW_PSYCHOPY:
                kb.getKeys()
            cross.draw()
            # Save mouse pos
            mouse_pos = mouse.getPos()
            dataFile.write(
                "%i,%i,%i,%i,%.5f %.5f,%.5f\n"
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
        while trial_time.getTime() < 2.3:
            if NEW_PSYCHOPY:
                kb.getKeys()
            # Draw objects and cross
            draw_objects(objects_trial)

            # Get mouse position
            mouse_pos = mouse.getPos()

            # Save
            dataFile.write(
                "%i,%i %i,%i,%.5f,%.5f,%.5f\n"
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
            if stop_trials[trial] and trial_time.getTime() > 0.55 and not stopped:
                stopped = 1
                win.color = "red"
            win.flip()

        win.color = "black"

        print(f"Trial no.: {trial}")

    # Show welcome screen
    end_text = "Ende - Vielen Dank!"
    message = visual.TextStim(win=win, text=end_text, color="white", height=0.07)
    message.draw()
    win.flip()

    dataFile.close()
    print("Ending experiment normally.")
    endExperiment(thisExp=thisExp, win=win)


if __name__ == "__main__":
    expInfo = setupExperiment()
    p_num = expInfo["participant"]
    sourcedata = pathlib.Path("data", "sourcedata")
    sourcedata.mkdir(exist_ok=True, parents=True)
    filename = sourcedata / f"{p_num}_beh-{expInfo['date']}"
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expInfo["expName"],
        version="",
        extraInfo=expInfo,
        runtimeInfo=None,
        originPath="C:\\Users\\richa\\GitHub\\task_motor_stopping\\experiment\\motor_stopping.py",
        savePickle=True,
        saveWideText=True,
        dataFileName=(sourcedata / filename).as_posix(),
        sortColumns="time",
    )
    logFile = setupLogging(filename=filename)
    win = setupWindow(expInfo=expInfo)
    # with open(filename.with_suffix(".csv"), "w") as dataFile:
    #     run(win=win, dataFile=dataFile)
    dataFile = open(filename.with_suffix(".csv"), "w")
    run(thisExp=thisExp, win=win, dataFile=dataFile)
    quit(thisExp=thisExp, win=win)
