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

HAS_KB_CALLBACK = Version(psychopy.__version__) >= Version("2024.3")

# Global variable to prevent multiple exiting
is_exiting = False


def setupExperiment() -> dict[str, str]:
    # __________________________________________________________________
    # Get input from participant and prepare data file to store behavioural results
    expName = "ChoiceStopping"
    expInfo = {
        "Subject": "Use the same UNIQUE ID everywhere",
        "Medication": "Off/On",
        "SessionNumber": "01",
        "Hand": "R/L",
        "Stimulation": "Off",
        "Run": "1",
    }
    dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
    if not dlg.OK:
        core.quit()
    expInfo["Medication"] = expInfo["Medication"].replace("/", "")
    expInfo["Hand"] = expInfo["Hand"].replace("/", "")
    expInfo["date"] = data.getDateStr(format="%Y%m%dT%H%M%S.%f")
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


def setupWindow() -> visual.Window:
    """
    Setup the window for the experiment.

    Returns
    ==========
    psychopy.visual.Window
        Window to display stimuli in.
    """
    win = visual.Window(
        size=(1920, 1080),
        fullscr=True,
        screen=1,
        winType="pyglet",
        allowGUI=False,
        allowStencil=False,
        # monitor="testMonitor",
        color="black",
        colorSpace="rgb",
        blendMode="avg",
        useFBO=True,
        units="height",
    )
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
    if HAS_KB_CALLBACK:
        kb = keyboard.KeyboardDevice(muteOutsidePsychopy=False)
        kb.start()
        kb.clearEvents()
        kb.registerCallback(
            response="escape",
            func=store_and_quit,
            kwargs={"win": win},
        )
    else:
        print("Using old keyboard class")
        event.globalKeys.add(
            key="escape", func=store_and_quit, func_kwargs={"win": win}
        )

    start_with_space = visual.TextStim(
        win=win,
        text="Leertaste drücken um zu starten",
        pos=(0.0, -0.25),
        height=0.03,
        color="white",
    )
    continue_with_space = visual.TextStim(
        win=win,
        text="Leertaste drücken um weiterzumachen",
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
    no_mov_text = visual.TextStim(
        win=win, text="Keine \nBewegung", color="lime", height=0.05, bold=True
    )

    threshold_mov = 0.1

    # Fixation cross
    cross_pos = -0.4
    cross_size = 0.1
    cross = visual.ShapeStim(
        win=win,
        lineColor="gray",
        lineWidth=8,
        vertices=(
            (0, cross_pos - cross_size),
            (0, cross_pos + cross_size),
            (0, cross_pos),
            (-cross_size, cross_pos),
            (cross_size, cross_pos),
        ),
        closeShape=False,
    )

    # Positions
    # Generate a list of equidistant target positions (from cross)
    n_pos = 15
    positions = []
    for i in range(n_pos):
        p_x = 0.75 * np.cos((2 * i * np.pi) / n_pos)
        p_y = 0.8 * np.sin((2 * i * np.pi) / n_pos) - 0.4
        if p_y > -0.35:
            positions.append((p_x, p_y))
    n_pos = len(positions)  # Get final number of positions

    # Define experimental design parameters
    trials_per_block = 45
    blocks = np.array([0, 1, 2, 3])  # 4 different experimental conditions
    n_blocks = len(blocks)
    n_trials = trials_per_block * n_blocks  # 157 trials in total

    trials = np.arange(n_trials)
    # Shuffle the trials
    rng = np.random.default_rng(seed=30)
    rng.shuffle(trials)
    # Generate the order of the conditions and the stop trials
    conditions = np.repeat(blocks, trials_per_block)
    # Shuffle the order of the conditions
    conditions = conditions[trials]

    # Give a stop signal on 33 % of the trials
    n_stop_trials = trials_per_block * 1 / 3
    if not n_stop_trials.is_integer():
        raise ValueError("Number of trials per block must be divisible by 5.")
    n_stop_trials = int(n_stop_trials)
    n_go_trials = trials_per_block - n_stop_trials
    stop_trials = np.array(
        [[1] * n_stop_trials + [0] * n_go_trials] * n_blocks
    ).flatten()
    stop_trials = stop_trials[trials]
    print(f"Number of stop trials = {stop_trials.sum()}")

    # __________________________________________________________________
    # Start experiment

    # Show welcome screen
    welcome_message = visual.TextStim(
        win=win, text="Willkommen zu unserer Untersuchung!", color="white", height=0.07
    )
    draw_and_wait(start_with_space, welcome_message, win)

    dataFile.write(
        "Trial,Condition,StopTrial,TargetsDisplayed,Stopped,RectVisited,TrialTime,Position_X,Position_Y\n"
    )
    # __________________________________________________________________
    # Loop over trials
    break_message = visual.TextStim(win=win, text="Pause", color="white", height=0.07)
    trial_time = core.Clock()
    for trial, condition in zip(np.arange(n_trials), conditions):
        if trial > 1 and not trial % trials_per_block:
            draw_and_wait(continue_with_space, break_message, win)

        print(f"Trial no.: {trial+1}/{n_trials}")
        # Target objects, make sure that the position varies from one trial to the next
        ids = rng.choice(3, 3, replace=False)
        objects_trial = [rectangles[ids[0]]]
        if condition == 0:
            objects_trial.append(rectangles[ids[1]])
        elif condition == 1:
            objects_trial.extend([rectangles[ids[1]], rectangles[ids[2]]])
        elif condition == 2:
            objects_trial.append(no_mov_text)
        elif condition == 3:
            objects_trial.extend([rectangles[ids[1]], no_mov_text])
        else:
            raise ValueError(f"Condition must be between 0 and 3. Got: {condition}.")

        # Choose a random position for every object
        ids_pos = rng.choice(n_pos, len(objects_trial), replace=False)
        for object, id in zip(objects_trial, ids_pos):
            object.pos = positions[id]

        # Reset variables
        cross.lineColor = "white"
        stop_trial = stop_trials[trial]
        stopped = 0
        obj_visited = 0
        end_trial = 0

        # Pen has to get close to fixation cross
        while not np.sum(np.abs(mouse.getPos() - (0, cross_pos))) < threshold_mov:
            if HAS_KB_CALLBACK:
                kb.getKeys()
            cross.draw()
            win.flip()

        # Change color of cross if pen is close enough
        cross.lineColor = "red"

        trial_time.reset()
        while trial_time.getTime() < 1.2:
            if HAS_KB_CALLBACK:
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

        is_moving = False
        trial_time.reset()
        timer_on_end = core.CountdownTimer(1.0)  # 300 ms
        while not end_trial or timer_on_end.getTime() > 0:
            # Draw rectangles and cross
            for obj in objects_trial:
                obj.draw()
            win.flip()
            time_elapsed = trial_time.getTime()
            mouse_pos = mouse.getPos()

            # Check whether movement has started to potentially trigger the stop signal
            is_moving = np.sum(np.abs(mouse_pos - (0, cross_pos))) > threshold_mov
            if stop_trial and not end_trial and is_moving:
                win.color = "red"
                stopped = 1
                timer_on_end.reset()

            # Für jedes Viereck den Abstand zum Mauszeiger prüfen
            if not end_trial:
                for obj in objects_trial:
                    if isinstance(obj, visual.Rect):
                        dist = np.sum(np.abs(mouse_pos - obj.pos))
                        if dist < threshold_mov:
                            obj.fillColor = "red"
                            obj_visited = 1
                            timer_on_end.reset()
            end_trial = stopped or obj_visited

            # Save data
            dataFile.write(  # Trial,Targets,Condition,Stop,Time,Position_X,Position_Y
                "%i,%i,%i,%i,%i,%i,%i,%.5f,%.5f,%.5f\n"
                % (
                    trial,
                    condition,
                    stop_trial,
                    1,
                    stopped,
                    obj_visited,
                    is_moving,
                    time_elapsed,
                    mouse_pos[0],
                    mouse_pos[1],
                )
            )
            if HAS_KB_CALLBACK:
                kb.getKeys()
            if condition > 1 and not end_trial and not is_moving and time_elapsed > 2.3:
                break
        win.flip()
        if stopped:  # reset window color
            win.color = "black"
        elif obj_visited:  # reset original rectangle colors
            for color, rect in zip(colors, rectangles):
                rect.fillColor = color

    logging.log("Ending experiment normally.", level=logging.INFO)


if __name__ == "__main__":
    expInfo = setupExperiment()
    root_dir = pathlib.Path(__file__).parents[2]
    sourcedata = pathlib.Path(root_dir, "data", "sourcedata")
    sourcedata.mkdir(exist_ok=True, parents=True)
    basename = "_".join(
        (
            f"sub-{expInfo['Subject']}",
            f"ses-Med{expInfo['Medication']}{expInfo['SessionNumber']}",
            f"task-{expInfo['expName']}{expInfo['Hand']}",
            f"acq-Stim{expInfo['Stimulation']}",
            f"run-{expInfo['Run']}",
            f"beh-{expInfo['date']}",
        )
    )
    filename = sourcedata / basename
    logFile = setupLogging(filename=filename)
    win = setupWindow()
    with open(filename.with_suffix(".csv"), "w") as dataFile:
        try:
            run(win=win, dataFile=dataFile)
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            raise e
        finally:
            store_and_quit(win=win)
