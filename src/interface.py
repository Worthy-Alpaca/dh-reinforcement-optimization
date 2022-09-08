import sys
import os
import json
import configparser
import requests
import threading
import torch
import random
import logging
import numpy as np
import tkinter as tk
from tkinter import *
from types import FunctionType
from tkinter import Grid, filedialog, PhotoImage, ttk, messagebox
from tkcalendar import Calendar
from os.path import exists
from logging import error, info, warning

from misc.dataloader import KappaLoader
from validate import Validate


PACKAGE_PARENT = "../"
SCRIPT_DIR = os.path.dirname(
    os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__)))
)
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from main import RunModel
from helper import TextRedirector

try:
    from src.modules.gui.parent.canvas import MyCanvas
    from src.modules.gui.controller.controller import Controller
except:
    from modules.gui.parent.canvas import MyCanvas
    from modules.gui.controller.controller import Controller


class Interface:
    def __init__(self) -> None:
        """Creates the interface and it's child modules."""
        self.masterframe = tk.Tk()
        self.style = ttk.Style(self.masterframe)
        self.style.theme_use("alt")

        self.masterframe.protocol("WM_DELETE_WINDOW", self.__onClose)

        self.masterframe.title("SMD Produktion")
        self.masterframe.geometry(self.__center_window(self.masterframe, 900, 570))
        self.masterframe.minsize(width=1500, height=600)
        # self.masterframe.maxsize(width=1200, height=600)

        self.mainframe = tk.Frame(
            self.masterframe,
            bd=2,
            relief=tk.RAISED,
        )
        self.mainframe.pack(side="left", fill="both", expand=1, anchor=CENTER)

        self.sideframe = tk.Frame(self.masterframe, bd=2, relief=tk.RIDGE)
        self.sideframe.pack(side="right", fill="both", expand=1)

        # bind keyboard controlls
        self.mainframe.bind("<Control-x>", self.__onClose)

        if not exists(
            os.path.expanduser(os.path.normpath("~/Documents/D+H optimizer/settings"))
        ):
            os.makedirs(
                os.path.expanduser(
                    os.path.normpath("~/Documents/D+H optimizer/settings/logs")
                )
            )
        self.basePath = os.path.expanduser(
            os.path.normpath("~/Documents/D+H optimizer")
        )

        try:
            photo = PhotoImage(file=self.resource_path("bin/assets/logo.png"))
        except:
            photo = PhotoImage(
                file=os.getcwd() + os.path.normpath("/src/assets/logo.png")
            )
        self.masterframe.iconphoto(True, photo)
        self.calDate = {}
        self.machines = {}
        self.OptionList = []
        self.config = configparser.ConfigParser()
        self.__configInit()

        self.optimizerData = None

        # Create interface elements
        menubar = tk.Menu(self.masterframe)
        fileMenu = {
            "New": self.__new,
            "Load": self.__openNew,
            "Save": self.__saveAs,
            "seperator": "",
            "Exit Strg+x": self.__onClose,
        }
        self.__createMenu(menubar, "File", fileMenu)
        self.__createOptionsMenu(menubar)
        self.masterframe.config(menu=menubar)

        # Canvas(self.mainframe)
        self.text = tk.Text(self.sideframe, wrap="word")
        self.text.tag_configure("stderr", foreground="#b22222")
        self.text.pack(
            side="left", fill="both", expand=1
        )  # grid(row=3, column=0, columnspan=20, rowspan=10)

        sys.stdout = TextRedirector(self.text, "stdout")
        sys.stderr = TextRedirector(self.text, "stderr")
        logging.basicConfig(
            level=logging.INFO,
            handlers=[
                logging.FileHandler(
                    os.path.normpath(self.basePath + "/settings/logs/logs.log")
                ),
                logging.StreamHandler(stream=sys.stdout),
            ],
            format="%(asctime)s :: %(levelname)s :: %(message)s",
        )

        self.__createForms()

        self.controller = Controller(self.mainframe)

        if exists(self.config.get("optimizer_backend", "dbpath")):
            self.dbpath = self.config.get("optimizer_backend", "dbpath")
        else:
            self.__findDBPath()

    def resource_path(self, relative_path):
        try:
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath(".")

        return os.path.join(base_path, relative_path)

    def __center_window(self, master, width=300, height=200):
        # get screen width and height
        screen_width = master.winfo_screenwidth()
        screen_height = master.winfo_screenheight()

        # calculate position x and y coordinates
        x = (screen_width / 2) - (width / 2)
        y = (screen_height / 2) - (height / 2)
        return "%dx%d+%d+%d" % (width, height, x, y)

    def __createToplevel(self, height=200, width=300) -> tk.Toplevel:
        top = tk.Toplevel(self.mainframe)
        top.geometry(self.__center_window(self.masterframe, height=height, width=width))
        top.grab_set()
        return top

    def __findDBPath(self, basepath=False):
        top = self.__createToplevel(150, 300)
        top.title("Options")

        label = tk.Label(
            master=top, text="Please navigate to the product database file."
        )
        label.pack()

        basepath = basepath if basepath != False else self.basePath

        def getData():
            top.withdraw()
            top.grab_release()
            datapath = self.__openNew(
                startPath=basepath, filetypes=[("Database File", ".db")]
            )
            if datapath == None:
                return self.__findDBPath()
            self.dbpath = os.path.normpath(datapath)
            self.config.set("optimizer_backend", "dbpath", datapath)
            info("Successfully updated Database file.")

        button = tk.Button(master=top, text="OK", command=getData)
        button.pack()

    def __configInit(self) -> configparser.ConfigParser:
        """Initiate the config variables.

        Returns:
            ConfigParser: Module that contains config settings.
        """
        path = os.path.normpath(self.basePath + "/settings/settings.ini")
        if exists(path):
            return self.config.read(path)

        self.config.add_section("default")
        self.config.set("default", "calcgroups", "false")
        self.config.set("default", "useCache", "true")
        self.config.set("default", "numCarts", "3")
        self.config.set("default", "progressBar", "false")
        self.config.set("default", "trainingSamples", "13")
        self.config.add_section("optimizer_backend")
        self.config.set("optimizer_backend", "dbpath", "")
        self.config.set("optimizer_backend", "overlapThreshhold", "0.5")

    def __call__(self, *args: any, **kwds: any) -> None:
        """Call this to initate the window."""
        self.mainframe.mainloop()

    def run(self) -> None:
        """Call this to initate the window."""
        self.mainframe.mainloop()

    def __onClose(self, *args: any, **kwargs: any) -> None:
        """Closing Operation. Saves config variables to file."""
        if messagebox.askokcancel(
            "Quit",
            "Do you want to quit?",
        ):
            path = self.basePath

            with open(
                os.path.normpath(path + "/settings/settings.ini"), "w"
            ) as configfile:
                self.config.write(configfile)
            self.mainframe.destroy()
            sys.exit()

    def __startThread(self, function: FunctionType, *args):
        """Start a new thread with a given function.

        Args:
            function (FunctionType): The current function.
        """
        threading.Thread(target=function).start()

    def __dummy(self, text="") -> None:
        """Dummy function. Used for testing.

        Args:
            text (str, optional): Optional Textstring. Defaults to "".
        """
        print("this is a test")

    def __optimize(self):
        self.text.config(state=NORMAL)
        self.text.delete(1.0, tk.END)

        try:
            startDate = self.calDate["start"]
            endDate = self.calDate["end"]
        except:
            self.controller.error("Please set a date range.")
            return warning("Please set a date range.")

        if self.optimizerData == None:
            datapath = self.__openNew(
                startPath=os.path.expanduser(os.path.normpath("~/Downloads")),
                filetypes=[("excel spreadsheet", ".xlsx")],
            )
            if datapath == None:
                return
        else:
            return

        self.controller.wait(message="Generating Dataset")

        runmodel = RunModel(
            dbpath=self.config.get("optimizer_backend", "dbpath"),
            numSamples=self.config.getint("default", "trainingSamples"),
            caching=self.config.getboolean("default", "useCache"),
            disableProgress=self.config.getboolean("default", "progressBar"),
            overwriteDevice="cpu",
        )
        loader = KappaLoader(
            os.path.normpath(datapath),
            self.config.get("optimizer_backend", "dbpath"),
            startDate,
            endDate,
        )
        samples, sampleReqs = loader.getData()
        best_value, best_solution = runmodel.getBestOder(
            sampleReqs=sampleReqs,
            samples=samples,
            plot=True,
            numCarts=self.config.getint("default", "numCarts"),
        )
        self.controller(
            best_value,
            best_solution,
            dbpath=self.config.get("optimizer_backend", "dbpath"),
            calcGroups=self.config.getboolean("default", "calcgroups"),
            overlapThreshhold=0.5,
        )
        # try:
        #     validate.plotSoltions()
        # except Exception as e:
        #     print(e)
        # controller.error(best_value)
        return

    def __trainModel(self):
        if messagebox.askyesno(
            "Start Training",
            "Do you really want to start a new training?\nThis can take up to several hours.",
        ):
            random.seed(1000)
            np.random.seed(1000)
            torch.manual_seed(1000)
            torch.multiprocessing.set_start_method("spawn")
            runmodel = RunModel(
                dbpath=self.config.get("optimizer_backend", "dbpath"),
                numSamples=self.config.getint("default", "trainingSamples"),
                caching=self.config.getboolean("default", "useCache"),
            )
            EMBEDDING_DIMENSIONS = 10
            EMBEDDING_ITERATIONS_T = 2
            Q_Function, QNet, Adam, ExponentialLR = runmodel.init_model(
                # fname=os.path.join(runmodel.folder_name, shortest_fname),
                EMBEDDING_DIMENSIONS=EMBEDDING_DIMENSIONS,
                EMBEDDING_ITERATIONS_T=EMBEDDING_ITERATIONS_T,
                OPTIMIZER=torch.optim.Adam,
            )
            runmodel.fit(
                Q_func=Q_Function,
                Q_net=QNet,
                optimizer=Adam,
                lr_scheduler=ExponentialLR,
                NR_EPISODES=501,
                MIN_EPSILON=0.7,
                EPSILON_DECAY_RATE=6e-4,
                N_STEP_QL=4,
                BATCH_SIZE=16,
                GAMMA=0.7,
            )
        else:
            return

    def __createMenu(self, menubar: tk.Menu, label: str, data: dict) -> None:
        """Create a menu in the menubar.

        Args:
            menubar (tk.Menu): The current MenuBar instance.
            label (str): The Label of the menu.
            data (dict): Options in the menu.
        """
        filemenu = tk.Menu(menubar, tearoff=0)
        for key in data:
            if key == "seperator":
                filemenu.add_separator()
            else:
                filemenu.add_command(label=key, command=data[key])
        menubar.add_cascade(label=label, menu=filemenu)

    def __createOptionsMenu(self, menubar: tk.Menu) -> tk.Menu:
        """Creates the Options Menu.

        Args:
            menubar (tk.Menu): The current MenuBar instance.

        Returns:
            tk.Menu: The created menu.
        """
        filemenu = tk.Menu(menubar, tearoff=0)
        self.useIdealState = tk.BooleanVar()
        self.useIdealState.set(self.config.getboolean("default", "calcgroups"))

        def updateConfig(*data):
            self.config.set(*data)
            info(f"Successfully updated {data[1]} option")

        filemenu.add_checkbutton(
            label="Calculate Groups",
            var=self.useIdealState,
            command=lambda: updateConfig(
                "default", "calcgroups", str(self.useIdealState.get())
            ),
        )
        self.progressBar = tk.BooleanVar()
        self.progressBar.set(self.config.getboolean("default", "progressBar"))
        filemenu.add_checkbutton(
            label="Disable Progressbars",
            var=self.progressBar,
            command=lambda: updateConfig(
                "default", "progressBar", str(self.progressBar.get())
            ),
        )

        self.useCache = tk.BooleanVar()
        self.useCache.set(self.config.getboolean("default", "useCache"))
        filemenu.add_checkbutton(
            label="Use Cached Data",
            var=self.useCache,
            command=lambda: updateConfig(
                "default", "useCache", str(self.useCache.get())
            ),
        )

        filemenu.add_separator()
        filemenu.add_command(
            label="Change Database File",
            command=lambda: self.__findDBPath(
                self.config.get("optimizer_backend", "dbpath")
            ),
        )
        filemenu.add_command(label="Options", command=self.__setOptions)
        menubar.add_cascade(label="Options", menu=filemenu)

    def __createButton(
        self,
        posX: int,
        posY: int,
        text: str,
        master: tk.Frame,
        function: FunctionType,
        margin: int = None,
    ) -> tk.Button:
        """Creates a Button at the given position.

        Args:
            posX (int): The X Grid Position.
            posY (int): The Y Grid Position.
            text (str): The display text.
            function (FunctionType): The function to be called on button press.
            margin (int, optional): Margin to next element in Grid. Defaults to None.

        Returns:
            tk.Button: The created Button.
        """
        if margin == None:
            margin = 30
        button = tk.Button(
            master=master,
            height=1,
            width=10,
            text=text,
            command=lambda: self.__startThread(function),
        )
        button.grid(in_=master, column=posX, row=posY, padx=(margin, 0), sticky="nsew")

    def __createLabel(self, posX: int, posY: int, text: str) -> tk.Label:
        """Creates a Label at the given position.

        Args:
            posX (int): The X Grid Position.
            posY (int): The Y Grid Position.
            text (str): The display text.

        Returns:
            tk.Label: The created Label.
        """
        label = tk.Label(master=self.formbar, text=text)
        label.grid(column=posX, row=posY, sticky="nsew")

    def __createForms(self) -> None:
        """Creates the Inputs."""
        self.formbar = tk.Frame(self.mainframe, height=100)
        self.formbar.pack(side="top", fill="both", pady=5)
        tk.Label(self.formbar, text="Start Date:").grid(row=0, column=4, sticky="nsew")

        tk.Label(self.formbar, text="End Date:").grid(
            row=0, column=6, sticky="nsew", padx=5
        )

        self.__createButton(
            5,
            0,
            "Select",
            self.formbar,
            function=lambda: self.__showCal("start", 5, 1),
        )
        self.__createButton(
            7, 0, "Select", self.formbar, function=lambda: self.__showCal("end", 7, 1)
        )

        self.__createButton(
            8,
            0,
            text="Optimize",
            master=self.formbar,
            function=self.__optimize,
            margin=50,
        )
        self.__createButton(
            8,
            1,
            text="Train Model",
            master=self.formbar,
            function=self.__trainModel,
            margin=50,
        )

    def __showCal(self, i: str, posX: int, posY: int) -> tk.Toplevel:
        """Creates the popup Calendar.

        Args:
            i (str): Current time point.
            posX (int): The X Grid Position
            posY (int): The Y Grid Position

        Returns:
            tk.Toplevel: The created Calendar Toplevel
        """
        top = self.__createToplevel(300, 350)
        cal = Calendar(top, font="Arial 14", selectmode="day")

        def getDate(cal: Calendar):
            top.withdraw()
            top.grab_release()
            self.calDate[i] = cal.selection_get()
            self.__createLabel(posX, posY, self.calDate[i])
            info(f"Successfully set {cal.selection_get()} as {i} Date")

        cal.pack(fill="both", expand=True)
        ttk.Button(top, text="ok", command=lambda: getDate(cal)).pack()

    def __setOptions(self) -> tk.Toplevel:
        """Creates window for option management.

        Returns:
            tk.Toplevel: The created window.
        """
        top = self.__createToplevel()
        top.title("Options")

        def callback():
            numCart = numCarts.get()
            trainingSamp = trainingSamples.get()
            address = overlapThreshhold.get()
            self.config.set("default", "numCarts", str(numCart))
            self.config.set("default", "trainingSamples", str(trainingSamp))
            self.config.set("optimizer_backend", "overlapThreshhold", str(address))
            info(f"Successfully set trainingSamples to {str(trainingSamp)} ")
            info(f"Successfully set numCarts to {str(numCart)} ")
            info(f"Successfully set overlapThreshhold to {str(address)} ")
            top.withdraw()
            top.grab_release()
            return True

        tk.Label(top, text="Number of Samples to use in training").pack()
        trainingSamples = tk.IntVar()
        trainingSamples.set(self.config.getint("default", "trainingSamples"))
        tk.Entry(top, textvariable=trainingSamples).pack()

        tk.Label(top, text="Number of Carts available").pack()
        numCarts = tk.IntVar()
        numCarts.set(self.config.getint("default", "numCarts"))
        tk.Entry(top, textvariable=numCarts).pack()

        tk.Label(top, text="Overlap Threshhold").pack()
        overlapThreshhold = tk.StringVar()
        overlapThreshhold.set(self.config.get("optimizer_backend", "overlapThreshhold"))
        tk.Entry(top, textvariable=overlapThreshhold).pack()

        ttk.Button(top, text="OK", command=callback).pack()

    def __new(self) -> None:
        """Clears all entries."""
        self.calDate = {}
        self.__createLabel(5, 1, "")
        self.__createLabel(7, 1, "")

    def __saveAs(self, data: dict) -> None:
        """Save the privided Data to `.json`.

        Args:
            data (dict): The data to save.
        """
        file_opt = options = {}
        options["filetypes"] = [("JSON files", ".json"), ("all files", ".*")]
        options["initialdir"] = os.getcwd() + os.path.normpath("/data/presets")

        filename = filedialog.asksaveasfile(defaultextension=".json", **file_opt)
        if filename is None:
            return

        json.dump(data, filename)

    def __openNew(
        self,
        startPath: str = "",
        filetypes: list = [("database file", ".db"), ("excel spreadsheet", ".xlsx")],
    ) -> (None | str):
        """Opens an interface for file loading.

        Returns:
            str: The path to the selected file.
            None: If no file is selected.
        """
        file_opt = options = {}
        options["filetypes"] = filetypes
        options["initialdir"] = startPath
        filename = filedialog.askopenfilename(**file_opt)
        if filename == None or filename == "":
            return

        return filename


if __name__ == "__main__":
    Interface()()
