import numpy as np
import tkinter as tk
from helper import UtilFunctions
from logging import info
from matplotlib.pyplot import style
from sqlalchemy import engine


try:
    from src.modules.gui.parent.canvas import MyCanvas
    from src.misc.dataloader import DataBaseLoader
except:
    from modules.gui.parent.canvas import MyCanvas
    from misc.dataloader import DataBaseLoader


class Controller(MyCanvas):
    """Draws PCB on canvas"""

    def __init__(self, frame: tk.Tk) -> None:
        super().__init__(frame)
        self.dark = False

    def __call__(
        self,
        best_value,
        best_solution,
        samples,
        dbpath: str,
        refEngine: engine,
        calcGroups: bool = False,
        overlapThreshhold: float = 0.5,
        textSeparator: str = "",
    ) -> None:
        """Creates the summary with the provided data.

        Args:
            coords (dict): PCB Coordinates for plotting.
            mTime (dict): Time from assembly calculations.
            sTime (dict): Time from setup calculations.
            numParts (int): Number of created PCBs.
            randomInterupt (tuple, optional): The random interrupt values. (Min, Max). Defaults to (0, 0).
            prodName (str, optional): The Product name. Defaults to "".
        """
        self.figure.clear()
        self.textSeparator = textSeparator
        self.best_value = best_value
        self.overlapThreshhold = overlapThreshhold
        self.solution = best_solution["solution"]
        self.coords = best_solution["coords"]
        self.components = best_solution["components"]
        self.numCarts = best_solution["numCarts"]
        self.products = best_solution["products"]
        self.dbpath = dbpath
        self.refEngine = refEngine
        if calcGroups:
            self.validate = False
        else:
            self.validate = True
        self.helper = UtilFunctions(self.components)

        self.__plot_solution(
            self.coords,
            self.solution,
            samples,
            validate=self.validate,
        )
        self.canvas.draw()
        return

    def __plot_solution(
        self, coords: np.ndarray, solution: list, shortText: np.ndarray, validate=False
    ):
        """Method to plot the given coordinates according to the give solution.

        Args:
            coords (np.ndarray): The current coordinate set.
            solution (list): The calculated solution.
        """

        self.figure.clear()

        labels = coords[:, 3:4]
        labels = labels[:, 0].tolist()

        solutionListOG = []
        textShortStr = "\n"
        for x in solution:
            solutionListOG.append(
                f"{coords[x][3:4][0]} {self.textSeparator} {shortText[x, 1]}"
            )

        t = 0
        l = len(solutionListOG)
        if not validate:
            solutionList = self.__calcGroups(solutionListOG)
            t = l - len(solutionList)
            t = t * 20
        else:
            solutionList = []
        textstr = ""
        if not validate:
            textShortStr = (
                f"{len(solutionList)} Gruppen\n{t} Minuten durch Gruppierung gespart\n"
            )
            for x in solutionList:
                textShortStr += f"{x}\n"

        for x in solutionListOG:
            textstr += f"{x}\n"

        coords = coords[:, :3].astype(np.float32)
        plot = self.figure.add_subplot(121)
        ax = self.figure.add_subplot(122)
        ax.axis("off")
        plot.axis("off")

        plot.text(
            -0.2,
            0.95,
            textstr,
            transform=plot.transAxes,
            fontsize=12,
            verticalalignment="top",
            wrap=True,
            color="#ffffff" if self.dark else "#333333",
        )
        plot.set_title(
            "model / overlap = %.5f s"
            % (
                self.helper.calc_total_time(
                    self.solution,
                )[0],
                # + groupTimings
            ),
            color="#ffffff" if self.dark else "#333333",
        )

        ax.text(
            0.05,
            0.95,
            textShortStr,
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            wrap=True,
            color="#ffffff" if self.dark else "#333333",
        )

        info(f"Calculated the best solution:\n{textstr}")

        # for idx in range(n - 1):
        #     i, next_i = solution[idx], solution[idx + 1]
        #     plot.plot(
        #         [coords[i, 0], coords[next_i, 0]],
        #         [coords[i, 1], coords[next_i, 1]],
        #         [coords[i, 2], coords[next_i, 2]],
        #         "k",
        #         lw=2,
        #         alpha=0.8,
        #     )

        # i, next_i = solution[-1], solution[0]
        # plot.plot(
        #     [coords[i, 0], coords[next_i, 0]],
        #     [coords[i, 1], coords[next_i, 1]],
        #     [coords[i, 2], coords[next_i, 2]],
        #     "k",
        #     lw=2,
        #     alpha=0.8,
        # )
        # plot.set(
        #     xlabel="Number of placements",
        #     ylabel="Simulated production time",
        #     zlabel="Cumulative Component Score",
        # )
        # # plot.xlabel("Number of placements")
        # # plot.ylabel("Number of Components")
        # plot.plot(
        #     coords[solution[0], 0],
        #     coords[solution[0], 1],
        #     coords[solution[0], 2],
        #     "x",
        #     markersize=10,
        # )
        # first = solutionList[0]
        # if type(first) == list:
        #     first = first[0]

        # plot.text(
        #     coords[solution[0], 0],
        #     coords[solution[0], 1],
        #     coords[solution[0], 2],
        #     "%s" % (str(first)),
        #     size=10,
        #     zorder=1,
        # )
        # last = solutionList[-1]
        # if type(last) == list:
        #     last = last[-1]

        # plot.text(
        #     coords[solution[-1], 0],
        #     coords[solution[-1], 1],
        #     coords[solution[-1], 2],
        #     "%s" % (str(last)),
        #     size=10,
        #     zorder=1,
        # )

        return

    def wait(self, message: str = "") -> any:
        """Function that displays a loading screen."""
        self.figure.clear()
        waitPlot = self.figure.add_subplot(111)

        style.use("ggplot")
        waitPlot.axis("off")
        waitPlot.text(
            0.5,
            0.5,
            f"Loading...\n{message}",
            fontsize=14,
            verticalalignment="center",
            horizontalalignment="center",
            color="green",
            wrap=True,
        )
        self.canvas.draw()

    def error(self, error: str) -> None:
        """Function that displays an error message.

        Args:
            error (str): The current error message.
        """
        self.figure.clear()
        errorPlot = self.figure.add_subplot(111)

        style.use("ggplot")
        errorPlot.axis("off")
        errorPlot.text(
            0.5,
            0.5,
            error,
            fontsize=14,
            verticalalignment="center",
            horizontalalignment="center",
            color="red",
            wrap=True,
        )
        self.canvas.draw()

    def __calcGroups(self, solutionList):
        """Method to calculate batch queues.

        Args:
            solutionList (list): The current products according to predicted solution.

        Returns:
            list: A list containing the batch lists.
        """
        maxSlots = 36 * self.numCarts * 8
        runningSlots = 0
        solutionListRunning = []
        solutionListReturn = []
        dataloader = DataBaseLoader(self.dbpath)
        for i in range(len(solutionList)):
            product = solutionList[i][:9]
            Components = dataloader.getProductData(product[:9])
            try:
                ComponentsNext = dataloader.getProductData(solutionList[i + 1][:9])
            except:
                ComponentsNext = []
            overlapComponents = list(set(Components) & set(ComponentsNext))
            if len(overlapComponents) / len(Components) < self.overlapThreshhold:
                solutionListRunning.append(product)
                solutionListReturn.append(solutionListRunning.copy())
                solutionListRunning.clear()
                continue
            slotSize = self.__calcSlotSize(Components)
            slotSizeOverlap = self.__calcSlotSize(overlapComponents)
            numComponents = slotSize - slotSizeOverlap
            if numComponents + runningSlots < maxSlots:
                runningSlots += numComponents
                solutionListRunning.append(product)
            else:
                solutionListReturn.append(solutionListRunning.copy())
                solutionListRunning.clear()
                solutionListRunning.append(product)
                runningSlots = slotSize
        solutionListReturn.append(solutionListRunning)
        b = []
        for lng in range(len(solutionListReturn)):
            if len(solutionListReturn[lng]) >= 1:
                b.append(solutionListReturn[lng])
        solutionListReturn = b
        return solutionListReturn

    def __calcSlotSize(self, components) -> int:
        """Method to calculate the slot size for the given component list.

        Args:
            components (list): The current component list.

        Returns:
            int: The cumulative size of the component feeders.
        """
        numComponents = 0
        for i in components:
            with self.refEngine.begin() as connection:
                result = connection.execute(
                    f"SELECT * FROM 'ReferenceComponents' WHERE Component = '{i}'"
                ).first()
                if result == None:
                    result = {"Feedersize": 8}
                else:
                    result = result._asdict()
                size = result["Feedersize"]
            if i == "Kreis 1.5mm Bildver" or i == "ATOM":
                numComponents += 0
            elif size == "Barcode":
                numComponents += 40
            elif size == None:
                numComponents += 8
            elif size == "MSF16" or int(size) == 12:
                numComponents += 16
            else:
                numComponents += int(size)
        return numComponents
