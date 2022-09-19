from pathlib import Path
import sys
import os
import json
import configparser
import threading
import torch
import random
import logging
import numpy as np
import ctypes as ct
import tkinter as tk
from tkinter import *
from types import FunctionType
from tkinter import Grid, filedialog, PhotoImage, ttk
from tkcalendar import Calendar
from os.path import exists
from logging import info, warning

from misc.dataloader import KappaLoader
from modules.extra.dataextender import DataExtender


PACKAGE_PARENT = "../"
SCRIPT_DIR = os.path.dirname(
    os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__)))
)
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from main import RunModel
from helper import TextRedirector

try:
    from src.modules.gui.controller.controller import Controller
    from src.modules.gui.parent.menubar import Titlebar, Menubar, MenuCustom
except:
    from modules.gui.parent.canvas import MyCanvas
    from modules.gui.controller.controller import Controller
    from modules.gui.parent.menubar import Titlebar, Menubar, MenuCustom


class Interface:
    def __init__(self) -> None:
        """Creates the interface and it's child modules."""
        self.masterframe = tk.Tk()
        self.masterframe.protocol("WM_DELETE_WINDOW", self.__onClose)
        self.masterframe.minsize(width=1500, height=600)
        self.masterframe.overrideredirect(True)
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

        if exists(os.getcwd() + os.path.normpath("/src/assets/theme")):
            path = os.getcwd() + os.path.normpath("/src/assets")
        else:
            path = self.resource_path("bin/assets")
        self.style = ttk.Style(self.masterframe)
        self.masterframe.tk.call("source", path + "/azure.tcl")
        self.config = configparser.ConfigParser()
        self.__configInit()

        if self.config.getboolean("default", "darkmode"):
            self.masterframe.tk.call("set_theme", "dark")
        else:
            self.masterframe.tk.call("set_theme", "light")

        big_frame = ttk.Frame(self.masterframe)

        try:
            self.photo = PhotoImage(file=self.resource_path("bin/assets/logo.gif"))
        except:
            self.photo = PhotoImage(
                file=os.getcwd() + os.path.normpath("/src/assets/logo.gif")
            )
        self.masterframe.iconphoto(True, self.photo)
        self.masterframe.title("SMD Produktions Optimierung")
        Titlebar(
            self.masterframe,
            big_frame,
            self.photo,
            "SMD Produktions Optimierung",
            False,
            False,
            True,
            1500,
            600,
            self.__onClose,
        )

        self.calDate = {}
        self.machines = {}
        self.OptionList = []

        self.optimizerData = None

        # Create interface elements
        menubar = Menubar(self.masterframe)
        menu = MenuCustom(menubar, "Datei")
        menu.add_command("Neu", self.__new)
        menu.add_command("Laden", self.__openNew)
        menu.add_command("Speichern", self.__saveAs)
        menu.add_separator()
        menu.add_command("Beenden", self.__onClose)
        self.__createOptionsMenu(menubar)

        self.mainframe = ttk.Frame(
            self.masterframe,
        )
        self.mainframe.pack(side="left", fill="both", expand=1, anchor=CENTER)
        self.sideframe = ttk.Frame(self.masterframe)
        self.sideframe.pack(side="right", fill="both", expand=1)

        # bind keyboard controlls
        self.mainframe.bind("<Control-x>", self.__onClose)

        self.text = tk.Text(self.sideframe, wrap="word")
        self.text.tag_configure("stderr", foreground="#b22222")
        self.text.pack(side="left", fill="both", expand=1)

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
        if self.config.getboolean("default", "darkmode"):
            self.controller.figure.patch.set_facecolor("#333333")
            self.controller.dark = True
        else:
            self.controller.figure.patch.set_facecolor("#ffffff")

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

    def __createToplevel(self, height=200, width=300, title="") -> tk.Toplevel:
        top = tk.Toplevel(self.mainframe)
        top.geometry(self.__center_window(self.masterframe, height=height, width=width))
        top.minsize(width=width, height=height)
        top.overrideredirect(True)
        big_frame = ttk.Frame(top)
        Titlebar(
            top,
            big_frame,
            self.photo,
            title,
            False,
            False,
            True,
            width,
            height,
            top.destroy,
            isToplevel=True,
        )
        top.grab_set()
        return top

    def __findDBPath(self, basepath=False):
        top = self.__createToplevel(150, 300, title="Optionen")

        label = ttk.Label(master=top, text="Bitte navigieren Sie zur Datenbank Datei.")
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

        button = ttk.Button(master=top, text="OK", command=getData)
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
        self.config.set("default", "progressBar", "True")
        self.config.set("default", "trainingSamples", "13")
        self.config.set("default", "darkmode", "true")
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
        if self.__askQuestion(
            "",
            "Möchten Sie wirklich beenden?",
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
            self.controller.error("Bitte setzten Sie eine Datumsspanne.")
            return warning("Bitte setzten Sie eine Datumsspanne.")

        if self.optimizerData == None:
            datapath = self.__openNew(
                startPath=os.path.expanduser(os.path.normpath("~/Downloads")),
                filetypes=[("excel spreadsheet", ".xlsx")],
            )
            if datapath == None:
                return
        else:
            return

        self.controller.wait(message="Datenset generierung...")

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
        if exists(self.basePath + os.path.normpath("/models")):
            pathDir = self.basePath + os.path.normpath("/models")
        else:
            pathDir = self.resource_path("/bin/assets/models")
        best_time = -float("inf")
        best_value = any
        best_solution = any
        for x in range(10):
            running_value, running_solution = runmodel.getBestOder(
                sampleReqs=sampleReqs,
                samples=samples,
                plot=True,
                numCarts=self.config.getint("default", "numCarts"),
                modelFolder=pathDir,
            )
            running_time = runmodel.helper.calc_total_time(running_solution["solution"])
            if running_time[0] > best_time:
                best_time = running_time[0]
                best_value = running_value
                best_solution = running_solution
        self.controller(
            best_value,
            best_solution,
            dbpath=self.config.get("optimizer_backend", "dbpath"),
            calcGroups=self.config.getboolean("default", "calcgroups"),
            overlapThreshhold=0.5,
        )
        return

    def __trainModel(self):
        if self.__askQuestion(
            "Training starten?",
            "Möchten Sie wirklich ein neues Training starten?\nDies kann mehrere Stunden dauern.",
            height=200,
            width=400,
        ):
            random.seed(1000)
            np.random.seed(1000)
            torch.manual_seed(1000)
            torch.multiprocessing.set_start_method("spawn")
            info("Starting training iteration.")
            runmodel = RunModel(
                dbpath=self.config.get("optimizer_backend", "dbpath"),
                numSamples=self.config.getint("default", "trainingSamples"),
                caching=self.config.getboolean("default", "useCache"),
            )
            EMBEDDING_DIMENSIONS = 10
            EMBEDDING_ITERATIONS_T = 2
            Q_Function, QNet, Adam, ExponentialLR = runmodel.init_model(
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

    def __askQuestion(self, title, message, height=200, width=300) -> None:
        """Create a menu in the menubar.

        Args:
            menubar (tk.Menu): The current MenuBar instance.
            label (str): The Label of the menu.
            data (dict): Options in the menu.
        """
        messageList = message.split("\n")
        maxlen = max([len(x) for x in messageList]) * 12
        width = maxlen if maxlen > width else width
        top = self.__createToplevel(height=height, width=width, title=title)
        topFrame = ttk.Frame(top)
        topFrame.pack(side="top", expand=1, fill="both")
        bottomFrame = ttk.Frame(top)
        bottomFrame.pack(side="bottom", expand=1, fill="both")
        for x in messageList:
            ttk.Label(
                topFrame,
                text=x,
                justify="center",
                anchor=CENTER,
                font=("Copperplate Gothic Bold", 13),
            ).pack(pady=10)

        buttonValue = tk.BooleanVar()

        buttonYes = ttk.Button(
            bottomFrame, text="Ja", command=lambda: buttonValue.set(True)
        )
        buttonYes.pack(side="left", padx=6)

        buttonNo = ttk.Button(
            bottomFrame, text="Nein", command=lambda: buttonValue.set(False)
        )
        buttonNo.pack(side="right", padx=6)

        buttonNo.wait_variable(buttonValue)

        top.withdraw()
        top.grab_release()
        return buttonValue.get()

    def __createOptionsMenu(self, menubar: tk.Menu) -> tk.Menu:
        """Creates the Options Menu.

        Args:
            menubar (tk.Menu): The current MenuBar instance.

        Returns:
            tk.Menu: The created menu.
        """
        filemenu = MenuCustom(menubar, "Optionen")
        self.useIdealState = tk.BooleanVar()
        self.useIdealState.set(self.config.getboolean("default", "calcgroups"))

        def change_theme():
            if self.masterframe.tk.call("ttk::style", "theme", "use") == "azure-dark":
                self.masterframe.tk.call("set_theme", "light")
                self.controller.figure.patch.set_facecolor("#ffffff")
                self.controller.dark = False
            else:
                self.masterframe.tk.call("set_theme", "dark")
                self.controller.figure.patch.set_facecolor("#333333")
                self.controller.dark = True

        def updateConfig(*data):
            self.config.set(*data)
            info(f"Successfully updated {data[1]} option")
            if data[1] == "darkmode":
                change_theme()

        filemenu.add_checkbutton(
            label="Gruppenkalkulation",
            var=self.useIdealState,
            command=lambda: updateConfig(
                "default", "calcgroups", str(self.useIdealState.get())
            ),
        )
        self.progressBar = tk.BooleanVar()
        self.progressBar.set(self.config.getboolean("default", "progressBar"))
        filemenu.add_checkbutton(
            label="Fortschrittbalken ausblenden",
            var=self.progressBar,
            command=lambda: updateConfig(
                "default", "progressBar", str(self.progressBar.get())
            ),
        )

        self.useCache = tk.BooleanVar()
        self.useCache.set(self.config.getboolean("default", "useCache"))
        filemenu.add_checkbutton(
            label="Cache Daten verwenden",
            var=self.useCache,
            command=lambda: updateConfig(
                "default", "useCache", str(self.useCache.get())
            ),
        )

        self.useDarkmode = tk.BooleanVar()
        self.useDarkmode.set(self.config.getboolean("default", "darkmode"))
        filemenu.add_checkbutton(
            label="Darkmode",
            var=self.useDarkmode,
            command=lambda: updateConfig(
                "default", "darkmode", str(self.useDarkmode.get())
            ),
        )

        filemenu.add_separator()
        filemenu.add_command(
            label="Datenbankdatei austauschen",
            command=lambda: self.__findDBPath(
                self.config.get("optimizer_backend", "dbpath")
            ),
        )
        filemenu.add_command(label="Neues Produkt einfügen", command=self.__addNewData)
        filemenu.add_separator()
        filemenu.add_command(label="Optionen", command=self.__setOptions)

    def __addNewData(self):
        datapath = self.__openNew(
            startPath=self.basePath,
            filetypes=[
                (
                    "Excel Datei",
                    ".xlsm .xls",
                ),
                ("Alle Dateien", "."),
            ],
        )

        dataextender = DataExtender(self.config.get("optimizer_backend", "dbpath"))
        success, productName = dataextender.checkDir(Path(datapath))
        if success:
            info("Product added successfully to database")
            return self.controller.wait("Produkt erfolgreich in Datenbank aufgenommen.")
        elif success == None:
            top = self.__createToplevel(height=250, width=350)
            bigframe = ttk.Frame(top)
            bigframe.pack(side="top", fill="both", expand=1)
            ttk.Label(
                bigframe,
                text=f"Bitte geben Sie die Anzahl der Abschaltungen\n pro Nutzen für das Produkt {productName} ein.",
            ).pack(side="top", pady=5)
            offsetEntryVar = tk.IntVar()
            errorFrame = ttk.Frame(bigframe)
            errorFrame.pack()
            ttk.Entry(bigframe, textvariable=offsetEntryVar).pack(pady=5)

            def getData():
                numOffsets = offsetEntryVar.get()
                if numOffsets == 0:
                    ttk.Label(
                        errorFrame,
                        text="Bitte geben Sie eine Zahl größer als 0 ein.",
                        foreground="red",
                    ).pack()
                    return
                with dataextender.engine.begin() as con:
                    con.execute(
                        f"INSERT INTO 'products' (product, numOffsets) VALUES ('{productName}', {int(numOffsets)} )"
                    )
                    info("Product added successfully to database")

            ttk.Button(bigframe, text="OK", command=getData).pack(pady=5)

    def __createButton(
        self,
        posX: int,
        posY: int,
        text: str,
        master: tk.Frame,
        function: FunctionType,
        margin: int = None,
        marginy: int = 0,
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
        button = ttk.Button(
            master=master,
            text=text,
            command=lambda: self.__startThread(function),
        )
        button.grid(
            in_=master,
            column=posX,
            row=posY,
            padx=(margin, 0),
            pady=marginy,
            sticky="nsew",
        )

    def __createLabel(self, posX: int, posY: int, text: str) -> tk.Label:
        """Creates a Label at the given position.

        Args:
            posX (int): The X Grid Position.
            posY (int): The Y Grid Position.
            text (str): The display text.

        Returns:
            tk.Label: The created Label.
        """
        label = ttk.Label(master=self.formbar, text=text)
        label.grid(column=posX, row=posY, sticky="nsew")

    def __createForms(self) -> None:
        """Creates the Inputs."""
        self.formbar = ttk.Frame(self.mainframe, height=100)
        self.formbar.pack(side="top", fill="both", pady=5)
        ttk.Label(self.formbar, text="Start Datum:").grid(
            row=0, column=4, sticky="nsew", padx=5
        )

        ttk.Label(self.formbar, text="End Datum:").grid(
            row=0, column=6, sticky="nsew", padx=5
        )

        self.__createButton(
            5,
            0,
            "Auswählen",
            self.formbar,
            function=lambda: self.__showCal("start", 5, 1),
        )
        self.__createButton(
            7,
            0,
            "Auswählen",
            self.formbar,
            function=lambda: self.__showCal("end", 7, 1),
        )

        self.__createButton(
            8,
            0,
            text="Optimieren",
            master=self.formbar,
            function=self.__optimize,
            margin=50,
        )
        self.__createButton(
            8,
            1,
            text="Model trainieren",
            master=self.formbar,
            function=self.__trainModel,
            margin=50,
            marginy=5,
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
        top = self.__createToplevel(400, 350)
        cal = Calendar(top, font="Arial 14", selectmode="day")

        def getDate(cal: Calendar):
            top.withdraw()
            top.grab_release()
            self.calDate[i] = cal.selection_get()
            self.__createLabel(posX, posY, self.calDate[i])
            info(f"Successfully set {cal.selection_get()} as {i} Date")

        cal.pack(fill="both", expand=True)
        ttk.Button(top, text="OK", command=lambda: getDate(cal)).pack()

    def __setOptions(self) -> tk.Toplevel:
        """Creates window for option management.

        Returns:
            tk.Toplevel: The created window.
        """
        toplevel = self.__createToplevel(height=400, title="Optionen")
        top = ttk.Frame(toplevel)
        top.pack(side="top", expand=1, fill="both")

        def callback():
            numCart = numCarts.get()
            trainingSamp = trainingSamples.get()
            address = overlapThreshhold.get()
            address = address.replace(",", ".")
            self.config.set("default", "numCarts", str(numCart))
            self.config.set("default", "trainingSamples", str(trainingSamp))
            self.config.set("optimizer_backend", "overlapThreshhold", str(address))
            info(f"Successfully set trainingSamples to {str(trainingSamp)} ")
            info(f"Successfully set numCarts to {str(numCart)} ")
            info(f"Successfully set overlapThreshhold to {str(float(address) * 100)}% ")
            toplevel.withdraw()
            toplevel.grab_release()
            return True

        ttk.Label(top, text="Anzahl an Samples für Training").pack(pady=5)
        ttk.Label(top, text="Muss eine Ganzzahl sein.", font=("", 10, "italic")).pack()
        trainingSamples = tk.IntVar()
        trainingSamples.set(self.config.getint("default", "trainingSamples"))
        ttk.Entry(top, textvariable=trainingSamples).pack(pady=5)
        ttk.Label(top, text="Verfügbare Anzahl an Rüstwagen").pack(pady=5)
        ttk.Label(top, text="Muss eine Ganzzahl sein.", font=("", 10, "italic")).pack()
        numCarts = tk.IntVar()
        numCarts.set(self.config.getint("default", "numCarts"))
        ttk.Entry(top, textvariable=numCarts).pack(pady=5)

        ttk.Label(top, text="Überschneidungsschwelle").pack(pady=5)
        ttk.Label(
            top, text="Muss eine Dezimalzahl unter 1 sein.", font=("", 10, "italic")
        ).pack()
        overlapThreshhold = tk.StringVar()
        overlapThreshhold.set(self.config.get("optimizer_backend", "overlapThreshhold"))
        ttk.Entry(top, textvariable=overlapThreshhold).pack(pady=5)

        ttk.Button(top, text="OK", command=callback).pack(pady=5)

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
