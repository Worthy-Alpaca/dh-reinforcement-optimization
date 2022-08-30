import tkinter as tk
from tkinter import *
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


class MyCanvas(tk.Canvas):
    def __init__(self, master, **kwargs) -> None:
        super().__init__(master, **kwargs)
        """Creates the canvas on which to draw"""
        self.mainframe = master
        # self.figure = Figure(figsize=(11.6, 6.5), dpi=100)
        self.figure = Figure(dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.mainframe)
        self.bind("<Configure>", self.__on_resize)
        self.height = self.winfo_reqheight()
        self.width = self.winfo_reqwidth()
        self.canvas.draw()
        self.__tools()

    def __on_resize(self, event):
        # determine the ratio of old width/height to new width/height
        wscale = float(event.width) / self.width
        hscale = float(event.height) / self.height
        self.width = event.width
        self.height = event.height
        # resize the canvas
        self.config(width=self.width, height=self.height)
        # rescale all the objects tagged with the "all" tag
        self.scale("all", 0, 0, wscale, hscale)

    def __tools(self):
        self.canvas.get_tk_widget().grid(
            row=3, column=0, columnspan=10, rowspan=10, padx=(20, 20), sticky="nsew"
        )
        self.toolbar = NavigationToolbar2Tk(
            self.canvas, self.mainframe, pack_toolbar=False
        )
        self.toolbar.update()
        self.toolbar.grid(
            row=13, column=0, columnspan=10, rowspan=10, padx=(20, 20), sticky="nsew"
        )


if __name__ == "__main__":
    root = Tk()
    myframe = Frame(root)
    myframe.pack(fill=BOTH, expand=YES)

    mycanvas = MyCanvas(myframe, highlightthickness=0)

    mycanvas.pack(fill=BOTH, expand=YES)
    mycanvas.create_line(0, 0, 200, 100)
    mycanvas.create_line(0, 100, 200, 0, fill="red", dash=(4, 4))
    mycanvas.create_rectangle(50, 25, 150, 75, fill="blue")
    mycanvas.addtag_all("all")
    root.mainloop()
