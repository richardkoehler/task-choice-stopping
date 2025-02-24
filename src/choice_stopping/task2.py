# ########################################################
# Stop-Signal choice trajectory task
# Pilot task
# ########################################################

from __future__ import absolute_import, annotations, division

import pathlib
import shutil
import time
from io import BufferedWriter, TextIOWrapper

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
    expInfo["date"] = data.getDateStr(format="%Y%m%dT%H%M%S")
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
    """Zeigt eine Nachricht und wartet auf SPACE oder ESC."""
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
    """Sichere Daten und beende das Experiment."""
    global is_exiting
    if is_exiting:
        return
    is_exiting = True
    win.color = "black"  # Changing window color takes 2 flips
    win.flip()
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


def run(win: visual.Window, dataFile: TextIOWrapper, objFile: BufferedWriter) -> None:
    """Führt den Haupt-Durchlauf des Experiments aus."""
    # ESC-Key abfangen
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
    stop_text = visual.TextStim(
        win=win, text="STOP", height=0.14, pos=(0.0, 0.15), color="white", bold=True
    )
    stop_color_win = (0.8 * 2 - 1, -1, -1)  # "red"

    mouse = event.Mouse(visible=True)

    # Vier mögliche Farben (vier Rechtecke)
    colors = ["aqua", "yellow", "fuchsia", "orange"]
    rect_size = 0.15
    all_rects = [
        visual.Rect(
            win=win, width=rect_size, height=rect_size, fillColor=color, lineColor=None
        )
        for color in colors
    ]

    # "Keine Bewegung": Grünes Oval + Weißer Text
    no_mov_text = visual.TextStim(
        win=win,
        text="Keine Bewegung",
        color="white",
        height=0.04,
        bold=True,
    )
    no_mov_oval = visual.ShapeStim(
        win=win,
        vertices="circle",
        size=(0.4, 0.2),  # Oval genug, damit Text nicht überlappt
        fillColor="lime",
        lineColor="lime",
        lineWidth=2,
    )

    threshold_mov = 0.1

    # Fixationskreuz
    cross_pos = (0, -0.4)
    cross_size = 0.1
    cross = visual.ShapeStim(
        win=win,
        lineColor="gray",
        lineWidth=8,
        vertices=(
            (cross_pos[0], cross_pos[1] - cross_size),
            (cross_pos[0], cross_pos[1] + cross_size),
            (cross_pos[0], cross_pos[1]),
            (-cross_size, cross_pos[1]),
            (cross_size, cross_pos[1]),
        ),
        closeShape=False,
    )

    # Positionen generieren (oberhalb von y>-0.35)
    n_pos = 15
    positions = []
    for i in range(n_pos):
        p_x = 0.75 * np.cos((2 * i * np.pi) / n_pos)
        p_y = 0.8 * np.sin((2 * i * np.pi) / n_pos) - 0.4
        if p_y > -0.35:
            positions.append((p_x, p_y))
    positions = np.array(positions)
    n_pos = len(positions)

    # Experimental Design
    trials_per_block = 45
    blocks = np.array([0, 1, 2, 3])  # 4 conditions
    n_blocks = len(blocks)
    n_trials = trials_per_block * n_blocks

    rng = np.random.default_rng(seed=30)
    trial_indices = np.arange(n_trials)
    rng.shuffle(trial_indices)

    conditions = np.repeat(blocks, trials_per_block)
    conditions = conditions[trial_indices]

    jitter = rng.uniform(low=-0.25, high=0.25, size=n_trials)
    cross_durations = 1.2 + jitter
    assert n_trials == len(conditions) == len(cross_durations)

    # 33% Stop Trials
    n_stop_trials = trials_per_block * 1 / 3
    if not n_stop_trials.is_integer():
        raise ValueError("Number of trials per block must be divisible by 5.")
    n_stop_trials = int(n_stop_trials)
    n_go_trials = trials_per_block - n_stop_trials
    stop_trials = np.array([[1]*n_stop_trials + [0]*n_go_trials]*n_blocks).flatten()
    stop_trials = stop_trials[trial_indices]
    print(f"Number of stop trials = {stop_trials.sum()}")

    def pick_rects(num: int) -> list[visual.Rect]:
        """Zieht num verschiedene Rechtecke ohne Wiederholung."""
        idxs = rng.choice(len(all_rects), size=num, replace=False)
        return [all_rects[i] for i in idxs]

    # Daten-Header
    header_items = {
        "Trial": "%i",
        "Condition": "%i",
        "StopTrial": "%i",
        "TargetsDisplayed": "%i",
        "IsMoving": "%i",
        "Stopped": "%i",
        "RectVisited": "%i",
        "EndTrial": "%i",
        "TrialTime": "%f",
        "TotalTime": "%f",
        "Position_X": "%f",
        "Position_Y\n": "%f\n",
    }
    header = ",".join(header_items.keys())
    header_len = len(header_items)
    format_str = ",".join(header_items.values())
    dataFile.write(header)

    break_message = visual.TextStim(win=win, text="Pause", color="white", height=0.07)
    total_time = core.Clock()
    trial_time = core.Clock()

    # Haupt-Schleife
    for trial_idx, condition, cross_duration in zip(
        range(n_trials), conditions, cross_durations
    ):
        if trial_idx > 1 and trial_idx % 30 == 0:
            draw_and_wait(continue_with_space, break_message, win)

        print(f"Trial no.: {trial_idx+1}/{n_trials}")

        # ------------------------------------------------------------
        # Condition 0 => 2 Vierecke
        # Condition 1 => 4 Vierecke
        # Condition 2 => 2 Vierecke + "Keine Bewegung" (Oval+Text)
        # Condition 3 => 1 Viereck  + "Keine Bewegung" (Oval+Text)
        #
        # => Wir überschreiten so nie 4 sichtbare Objekte
        # ------------------------------------------------------------
        objects_trial = []

        if condition == 0:
            # 2 Vierecke
            squares = pick_rects(2)
            pos_ids = rng.choice(n_pos, size=2, replace=False)
            for i, sq in enumerate(squares):
                sq.pos = positions[pos_ids[i]]
            objects_trial = squares

        elif condition == 1:
            # 4 Vierecke
            squares = pick_rects(4)
            pos_ids = rng.choice(n_pos, size=4, replace=False)
            for i, sq in enumerate(squares):
                sq.pos = positions[pos_ids[i]]
            objects_trial = squares

        elif condition == 2:
            # 2 Vierecke + Keine Bewegung => 4 Objekte gesamt
            squares = pick_rects(2)
            pos_ids = rng.choice(n_pos, size=3, replace=False)  
            # => 2 Positionen für die Vierecke + 1 für Oval+Text
            for i, sq in enumerate(squares):
                sq.pos = positions[pos_ids[i]]
            no_mov_oval.pos = positions[pos_ids[-1]]
            no_mov_text.pos = positions[pos_ids[-1]]
            objects_trial = squares + [no_mov_oval, no_mov_text]

        elif condition == 3:
            # 1 Viereck + Keine Bewegung => 3 Objekte
            squares = pick_rects(1)
            pos_ids = rng.choice(n_pos, size=2, replace=False)  
            # => 1 Position für das Viereck + 1 für Oval+Text
            squares[0].pos = positions[pos_ids[0]]
            no_mov_oval.pos = positions[pos_ids[1]]
            no_mov_text.pos = positions[pos_ids[1]]
            objects_trial = squares + [no_mov_oval, no_mov_text]

        else:
            raise ValueError(f"Condition must be between 0 and 3. Got: {condition}.")

        # Positionen in .npy speichern (optional)
        obj_positions = [obj.pos for obj in objects_trial]
        np.save(objFile, obj_positions)

        # Trial-Variablen
        cross.lineColor = "white"
        stop_trial = stop_trials[trial_idx]
        stopped = 0
        obj_visited = 0
        end_trial = 0

        # Warte, bis Maus in Nähe Fixationskreuz ist
        while not np.sum(np.abs(mouse.getPos() - cross_pos)) < threshold_mov:
            if HAS_KB_CALLBACK:
                kb.getKeys()
            cross.draw()
            win.flip()

        # Kreuz wird rot
        cross.lineColor = "red"

        # Warte cross_duration
        trial_time.reset()
        targets_displayed = 0
        while (time_elapsed := trial_time.getTime()) < cross_duration:
            if HAS_KB_CALLBACK:
                kb.getKeys()
            cross.draw()
            mouse_pos = mouse.getPos()
            is_moving = np.sum(np.abs(mouse_pos - cross_pos)) > threshold_mov
            data_to_write = (
                trial_idx,
                condition,
                stop_trial,
                targets_displayed,
                is_moving,
                stopped,
                obj_visited,
                end_trial,
                time_elapsed,
                total_time.getTime(),
                mouse_pos[0],
                mouse_pos[1],
            )
            if len(data_to_write) != header_len:
                raise ValueError("Data to write has incorrect number of columns.")
            dataFile.write(format_str % data_to_write)
            win.flip()

        # Targets einblenden
        trial_time.reset()
        timer_on_end = core.CountdownTimer(1.0)
        targets_displayed = 1

        while (not end_trial) or (timer_on_end.getTime() > 0):
            if not stopped:
                # Alle Objekte zeichnen
                i = 0
                while i < len(objects_trial):
                    obj = objects_trial[i]
                    # Oval+Text zusammen
                    if obj == no_mov_oval and i+1 < len(objects_trial) and objects_trial[i+1] == no_mov_text:
                        obj.draw()  # Oval
                        objects_trial[i+1].draw()  # Text
                        i += 2
                        continue
                    else:
                        obj.draw()
                    i += 1
            else:
                stop_text.draw()

            win.flip()
            time_elapsed = trial_time.getTime()
            mouse_pos = mouse.getPos()

            # Prüfen, ob ein Rechteck "besucht" wurde
            if not obj_visited:
                for obj in objects_trial:
                    if isinstance(obj, visual.Rect):
                        dist = np.sum(np.abs(mouse_pos - obj.pos))
                        if dist < rect_size:
                            obj_visited = 1
                            if not end_trial:
                                obj.fillColor = "red"
                                timer_on_end.reset()

            # Bewegung -> ggf. STOP-Signal
            is_moving = np.sum(np.abs(mouse_pos - cross_pos)) > threshold_mov

            data_to_write = (
                trial_idx,
                condition,
                stop_trial,
                targets_displayed,
                is_moving,
                stopped,
                obj_visited,
                end_trial,
                time_elapsed,
                total_time.getTime(),
                mouse_pos[0],
                mouse_pos[1],
            )
            if len(data_to_write) != header_len:
                raise ValueError("Data to write has incorrect number of columns.")
            dataFile.write(format_str % data_to_write)

            if stop_trial and (not end_trial) and is_moving:
                win.color = stop_color_win
                win.flip()  # Changing window color takes 2 flips
                stopped = 1
                timer_on_end.reset()

            end_trial = stopped or obj_visited

            if HAS_KB_CALLBACK:
                kb.getKeys()

            # Falls Condition > 1 und keine Bewegung => Trial-Abbruch nach 2.3s
            if condition > 1 and not end_trial and not is_moving and time_elapsed > 2.3:
                break

        # Cleanup nach Trial
        if stopped:
            win.color = "black"
            win.flip()
        elif obj_visited:
            for c, rect in zip(colors, all_rects):
                rect.fillColor = c
        win.flip()

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
    sub_dir = sourcedata / basename
    sub_dir.mkdir(exist_ok=True)
    shutil.copy2(__file__, sub_dir)  # Save the state of this script
    logFile = setupLogging(filename=sub_dir / "psychopy.log")
    win = setupWindow()
    with open(sub_dir / "data.csv", "w") as dataFile, open(
        sub_dir / "targets.npy", "wb"
    ) as objFile:
        try:
            run(win=win, dataFile=dataFile, objFile=objFile)
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            raise
        finally:
            store_and_quit(win=win)



