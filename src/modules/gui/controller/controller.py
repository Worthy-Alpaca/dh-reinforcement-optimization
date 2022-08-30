import random
import tkinter as tk
from matplotlib.pyplot import style

try:
    from src.modules.gui.parent.canvas import MyCanvas
except:
    from modules.gui.parent.canvas import MyCanvas


class Controller(MyCanvas):
    """Draws PCB on canvas"""

    def __init__(self, frame: tk.Tk) -> None:
        super().__init__(frame)

    def __call__(
        self,
        coords: dict,
        mTime: dict,
        sTime: dict,
        numParts: int,
        randomInterupt: tuple = (0, 0),
        prodName: str = "",
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
        plot = self.figure.add_subplot(121)
        ax = self.figure.add_subplot(122)
        runtime = sum(list(mTime.values()))
        # setupTime = sum(list(sTime.values()))
        ax.axis("off")
        sumtime = []
        for i in range(numParts):
            sumtime.append(
                runtime + random.randint(randomInterupt[0], randomInterupt[1])
            )

        textstr = "\n".join(
            (
                f"Product: {prodName} ",
                "",
                f"Overall time needed: {round(sum(sumtime), 2)} Seconds",
                f"Average Time: {round(sum(sumtime) / numParts, 2)} Seconds ",
                f"Highest time: {round(max(sumtime), 2)} Seconds",
                "",
                "Machines",
            )
        )
        substr = ""
        for key in mTime:
            substr = (
                substr
                + f"\n{key} Ideal: {round(mTime[key], 2)} Seconds \n{key} Setup Time: {round(sTime[key], 2)} Seconds\n"
            )
        textstr = textstr + substr
        props = dict(boxstyle="round", alpha=0.5)

        ax.text(
            0.05,
            0.95,
            textstr,
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment="top",
            wrap=True,
        )
        for key in coords:
            plot.scatter(coords[key]["X"], coords[key]["Y"])
        plot.legend(tuple(coords.keys()), loc="upper left")
        self.canvas.draw()

    def wait(self, message: str = "") -> any:
        """Function that displays a loading screen."""
        self.figure.clear()
        waitPlot = self.figure.add_subplot(111)

        style.use("ggplot")
        waitPlot.axis("off")
        # waitPlot.set_title("Loading...", color="green")
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
        # errorPlot.set_title(f"An error occured: {error} ", color="red")
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
