import itertools
import os
from scipy.spatial import distance_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from src.misc.dataloader import DataLoader
from src.misc.deploy import DeployModel
from collections import deque
import ctypes
import time


def get_graph_mat(n=10, size=1):
    """Throws n nodes uniformly at random on a square, and build a (fully connected) graph.
    Returns the (N, 2) coordinates matrix, and the (N, N) matrix containing pairwise euclidean distances.
    """
    coords = size * np.random.uniform(size=(n, 2))
    dist_mat = distance_matrix(coords, coords)
    return coords, dist_mat


def plot_graph(coords, mat):
    """Utility function to plot the fully connected graph"""
    n = len(coords)

    plt.scatter(coords[:, 0], coords[:, 1], s=[50 for _ in range(n)])
    for i in range(n):
        for j in range(n):
            if j < i:
                plt.plot(
                    [coords[i, 0], coords[j, 0]],
                    [coords[i, 1], coords[j, 1]],
                    "b",
                    alpha=0.7,
                )


model = DeployModel(Path(os.getcwd() + os.path.normpath("/FINAL MODEL")))

path = Path(
    r"C:\Users\stephan.schumacher\Documents\repos\dh-reinforcement-optimization\programms"
)
products = deque()
test = float("inf")

from sqlalchemy import create_engine
import sqlalchemy

engine = create_engine("sqlite:///products.db", echo=False)
if type(engine) == sqlalchemy.engine.base.Engine:
    print()


with engine.begin() as connection:
    for i in os.listdir(path):
        for m in ["m10", "m20"]:
            data_path = path / i / m

            dataloader = DataLoader(data_path)
            data, components, offsets = dataloader()
            tableName = f"{i}_{m}_data"
            data.to_sql(tableName, con=connection, if_exists="replace")

            tableName = f"{i}_{m}_components"
            components.to_sql(tableName, con=connection, if_exists="replace")

            tableName = f"{i}_{m}_offsets"
            offsets = pd.DataFrame(offsets, columns=["x", "y"])
            offsets.to_sql(tableName, con=connection, if_exists="replace")
    productNames = os.listdir(path)
    productNames = pd.DataFrame(productNames, columns=["Product"])
    productNames.to_sql("products", con=connection, if_exists="replace")

test = engine.execute("SELECT * FROM '2095525_m10_data'").fetchall()
print(test)
test = pd.read_sql_table("2095525_m10_data", "sqlite:///products.db")
print(test)


""" Klassischer TSP ansatz """


def get_stats(i):
    product = i
    overallLen = 0
    allComponents = []
    overallTime = 0
    for m in ["m10", "m20"]:
        data_path = path / i / m

        dataloader = DataLoader(data_path)
        data, components, offsets = dataloader()
        Ymax = data["Y"].max() + max(offsets[1])
        Xmax = data["X"].max() + max(offsets[0])
        predArray = np.array(
            [len(data) * len(offsets), 0 if m == "m10" else 1, Xmax, Ymax]
        )
        time = model.predict(predArray)
        overallTime += time
        overallLen += len(data) * len(offsets)
        allComponents.append(components["Component Code"].unique())

    return (
        overallLen,
        np.array(list(itertools.chain.from_iterable(allComponents))),
        overallTime,
    )


def get_overlap(p1, p2):
    l2 = len(list(set(p1) & set(p2)))
    l1 = len(p1)
    return l2 / l1


for i in os.listdir(path):
    prodLen, prodComps, prodTime = get_stats(i)
    products.append(
        {"name": i, "prodLen": prodLen, "prodComps": prodComps, "prodTime": prodTime}
    )


def calcNearest(current, others):
    overlaps = {}
    for i in others:
        overlaps[i["name"]] = get_overlap(current["prodComps"], i["prodComps"])
    overlaps.pop(current["name"])
    shortest = max(overlaps, key=overlaps.get)
    return shortest, overlaps[shortest]


others = products.copy()
for i in products:
    others.popleft()
    closest, overlap = calcNearest(i, products)
    print(i["name"], "most overlap with:", closest, overlap)


exit()
for i in os.listdir(path):
    product = i
    overallLen = 0
    allComponents = []
    for m in ["m10", "m20"]:
        data_path = path / i / m

        dataloader = DataLoader(data_path)
        data, components, offsets = dataloader()

        overallLen += len(data) * len(offsets)
        allComponents.append(components["Component Code"].unique())

    products[product] = {
        "len": overallLen,
        "comps": list(itertools.chain.from_iterable(allComponents)),
    }


def compare(p1, p2):
    l2 = len(list(set(p1) & set(p2)))
    l1 = len(p1)
    return l2 / l1


globalList = {}
for i in products.keys():
    currentList = products.copy()
    currentList.pop(i)
    # currentDict = {i: {"overlap": 1, "len": products[i]["len"]}}
    currentDict = [[products[i]["len"], 1]]
    for j in currentList:
        currentDict.append(
            [
                products[j]["len"],
                compare(products[i]["comps"], products[j]["comps"]),
            ]
        )
    globalList[i] = currentDict

print(globalList)

npArray = []
for key in globalList:
    for i in globalList[key]:
        npArray.append(i)


npArray = np.asarray(npArray)

plot_graph(npArray, 0)
plt.show()

[3, 4, 2, 19, 7, 14, 0, 16, 12, 17, 9, 10, 8, 18, ...]
