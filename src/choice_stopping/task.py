# ########################################################
# Stop-Signal choice trajectory task
# Pilot task
# ########################################################

from __future__ import absolute_import, annotations, division

import pathlib
import time
from io import TextIOWrapper

import numpy as np
import psychopy
from packaging.version import Version
from psychopy import core, data, event, gui, logging, visual
from psychopy.hardware import keyboard

HAS_HB_CALLBACK = Version(psychopy.__version__) >= Version("2024.3")

# Global variable to prevent multiple exiting
is_exiting = False


def setupExperiment() -> dict[str, str]:
    # __________________________________________________________________
    # Get input from participant and prepare data file to store behavioural results
    expName = "choice-stopping"
    expInfo = {"participant": "Use the same UNIQUE ID everywhere"}
    dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
    if not dlg.OK:
        core.quit()
    expInfo["date"] = data.getDateStr(format="%Y-%m-%dT%Hh%M.%S.%f")
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
    logging.console.setLevel(logging.WARNING)
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
        size=(1920, 1080),
        fullscr=True,
        screen=0,  # 1,
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


def draw_and_wait(
    skip_with_space: visual.TextStim,
    message: visual.TextStim,
    win: visual.Window,
):
    message.draw()
    skip_with_space.draw(win)
    win.flip()
    while True:
        keys = event.getKeys()
        if "space" in keys:
            break
        elif "escape" in keys:
            store_and_quit(win=win)
        time.sleep(0.1)


def return_and_delete_rand_idx(array: list, high: int):
    idx = np.random.randint(high)
    x = array[idx]
    del array[idx]
    return x


def store_and_quit(win: visual.Window):
    global is_exiting
    if is_exiting:
        return
    is_exiting = True
    message = visual.TextStim(
        win=win, text="Ende - Vielen Dank!", color="white", height=0.07
    )
    message.draw()
    win.flip()
    time.sleep(1)
    win.clearAutoDraw()
    win.flip()
    win.close()
    logging.flush()
    core.quit()


def run(win: visual.Window, dataFile: TextIOWrapper) -> None:
    np.random.seed(41)  # Change seed in screening session

    func_kwargs = {"win": win}
    if HAS_HB_CALLBACK:
        kb = keyboard.KeyboardDevice(muteOutsidePsychopy=False)
        kb.start()
        kb.clearEvents()
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
    colors = ["aqua", "yellow", "fuchsia"]
    rectangles = [
        visual.Rect(win=win, width=0.15, height=0.15, fillColor=color)
        for color in colors
    ]
    rng_rect = np.random.default_rng(seed=42)
    no_mov_text = visual.TextStim(
        win=win, text="Keine \nBewegung", color="lime", height=0.05, bold=True
    )

    threshold = 0.1

    # Fixation cross
    cross_pos = -0.4
    cross = visual.ShapeStim(
        win=win,
        lineColor="gray",
        lineWidth=4,
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
    n_pos = 15
    positions = []
    for i in range(n_pos):
        p_x = 0.7 * np.cos((2 * i * np.pi) / n_pos)
        p_y = 0.7 * np.sin((2 * i * np.pi) / n_pos) - 0.4
        if p_y > -0.35:
            positions.append((p_x, p_y))
    n_pos = len(positions)  # Get final number of positions
    rng_pos = np.random.default_rng(seed=42)

    # Define experimental design parameters
    trials_per_block = 40
    blocks = np.array([0, 1, 2, 3])  # 4 different experimental conditions
    n_blocks = len(blocks)
    n_trials = trials_per_block * n_blocks  # 180 trials in total

    trials = np.arange(n_trials)
    # Shuffle the trials
    np.random.shuffle(trials)
    # Generate the order of the conditions and the stop trials
    conditions = np.repeat(blocks, trials_per_block)
    # Shuffle the order of the conditions
    conditions = conditions[trials]

    # Give a stop signal on 20 % of the trials
    n_stop_trials = trials_per_block * 0.2
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

    dataFile.write(
        "Trial,Condition,StopTrial,TargetsDisplayed,Stopped,RectVisited,TrialTime,Position_X,Position_Y\n"
    )
    # __________________________________________________________________
    # Loop over trials
    trial_time = core.Clock()
    for trial, condition in zip(np.arange(n_trials), conditions):
        if condition > 3:
            raise ValueError(f"Condition must be between 0 and 3. Got: {condition}.")

        print(f"Trial no.: {trial}")
        # Target objects, make sure that the position varies from one trial to the next
        if condition % 2:  # if condition == 1 or condition == 3
            ids_rect = rng_rect.choice(3, 2, replace=False)
        else:
            ids_rect = rng_rect.choice(3, 1, replace=False)
        objects_trial = [rectangles[i] for i in ids_rect]
        if condition > 1:  # 2 or 3 squares with no-movement option
            objects_trial.append(no_mov_text)

        # Choose a random position for every object
        ids_pos = rng_pos.choice(n_pos, len(objects_trial), replace=False)
        for object, id in zip(objects_trial, ids_pos, strict=True):
            object.pos = positions[id]

        # Reset variables
        cross.lineColor = "white"
        stop_trial = stop_trials[trial]
        stopped = 0
        obj_visited = 0
        end_trial = 0

        # Pen has to get close to fixation cross
        while not np.sum(np.abs(mouse.getPos() - (0, cross_pos))) < 0.05:
            if HAS_HB_CALLBACK:
                kb.getKeys()
            cross.draw()
            win.flip()

        # Change color of cross if pen is close enough
        cross.lineColor = "red"

        trial_time.reset()
        while trial_time.getTime() < 1.2:
            if HAS_HB_CALLBACK:
                kb.getKeys()
            cross.draw()
            # Save mouse pos
            mouse_pos = mouse.getPos()
            dataFile.write(  # Trial,Targets,Condition,Stop,Time,Position_X,Position_Y
                "%i,%i,%i,%i,%i,%i,%.5f,%.5f,%.5f\n"
                % (
                    trial,
                    condition,
                    stop_trial,
                    0,
                    stopped,
                    obj_visited,
                    trial_time.getTime(),
                    mouse_pos[0],
                    mouse_pos[1],
                )
            )
            win.flip()

        trial_time.reset()
        timer_on_end = core.CountdownTimer(1.0)  # 300 ms
        while not end_trial or timer_on_end.getTime() > 0:
            time_elapsed = trial_time.getTime()
            if condition > 1 and time_elapsed > 2.3:
                break

            if HAS_HB_CALLBACK:
                kb.getKeys()

            # Draw rectangles and cross
            for obj in objects_trial:
                obj.draw()

            # Get mouse position
            mouse_pos = mouse.getPos()

            # Save
            dataFile.write(  # Trial,Targets,Condition,Stop,Time,Position_X,Position_Y
                "%i,%i,%i,%i,%i,%i,%.5f,%.5f,%.5f\n"
                % (
                    trial,
                    condition,
                    stop_trial,
                    1,
                    stopped,
                    obj_visited,
                    time_elapsed,
                    mouse_pos[0],
                    mouse_pos[1],
                )
            )

            # Check whether movement has started to potentially trigger the stop signal
            if (
                stop_trial
                and not end_trial
                and np.sum(np.abs(mouse_pos - (0, cross_pos))) > 0.05
            ):
                win.color = "red"
                stopped = 1
                timer_on_end.reset()

            # Für jedes Viereck den Abstand zum Mauszeiger prüfen
            if not end_trial:
                for obj in objects_trial:
                    if isinstance(obj, visual.Rect):
                        dist = np.sum(np.abs(mouse.getPos() - obj.pos))
                        if dist < threshold:
                            obj.fillColor = "red"
                            obj_visited = 1
                            timer_on_end.reset()
            end_trial = stopped or obj_visited
            win.flip()
        if stopped:  # reset window color
            win.color = "black"
        elif obj_visited:  # reset original rectangle colors
            for color, rect in zip(colors, rectangles):
                rect.fillColor = color

    logging.log("Ending experiment normally.", level=logging.INFO)


if __name__ == "__main__":
    expInfo = setupExperiment()
    p_num = expInfo["participant"]
    sourcedata = pathlib.Path("data", "sourcedata")
    sourcedata.mkdir(exist_ok=True, parents=True)
    filename = sourcedata / f"{p_num}_beh-{expInfo['date']}"
    logFile = setupLogging(filename=filename)
    win = setupWindow(expInfo=expInfo)
    logging.flush()
    with open(filename.with_suffix(".csv"), "w") as dataFile:
        TESTING = True
        if TESTING:
            run(win=win, dataFile=dataFile)
            store_and_quit(win=win)
        else:
            try:
                run(win=win, dataFile=dataFile)
            except Exception as e:
                logging.error(f"An unexpected error occurred: {e}")
            finally:
                store_and_quit(win=win)
