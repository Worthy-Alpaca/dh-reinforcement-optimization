from email import header
from logging import warning
import os
from random import random
import sqlalchemy
import pandas as pd
from pathlib import Path
from sqlalchemy import engine
import re
import numpy as np


class DataBaseLoader:
    def __init__(self, pathToProductTxT: Path, refEngine: engine = None) -> None:
        file = open(pathToProductTxT, "r")
        self.refEngine = refEngine

        def createCleanData(x: str):
            pattern = re.compile("[0-9]{7}-[0-9]")
            x = x.replace(" ", "")
            if len(x) < 7:
                return np.nan
            if not header:
                if not bool(re.match(pattern, x)):
                    return np.nan, np.nan
            cleanString = f"{x[:2]}.{x[2:5]}.{x[5:7]}"
            amount = x[8:] if x[8:] else 0
            amount = amount if amount == 0 or len(amount) == 1 else amount[-1]
            return cleanString, amount

        prodDB = {}
        refDB = {}
        for line in file:
            lineContent = line.strip().split(";")
            productName = lineContent.pop(0).replace(" ", "")
            productName, _ = createCleanData(productName, header=True)
            components = []
            runningAmount = 0
            for x in lineContent:
                cleanString, amount = createCleanData(x)
                components.append(cleanString)
                runningAmount += int(amount)

            prodDB[productName] = components
            refDB[productName] = runningAmount
        file.close()

        df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in prodDB.items()]))

        values = df.values.ravel("K")
        values = values[~pd.isnull(values)]
        values.sort()

        def count_and_append(a):  # For sorted arrays
            a0 = a[:]
            sf0 = np.flatnonzero(a0[1:] != a0[:-1]) + 1
            shift_idx = np.concatenate(([0], sf0, [a0.size]))
            c = shift_idx[1:] - shift_idx[:-1]
            out_col = np.repeat(c, c)
            return np.column_stack((a, out_col))

        values = count_and_append(values)
        values = pd.DataFrame(values, columns=["Component", "Occurance"])
        values = values.drop_duplicates()
        self.refDB = refDB
        self.prodData = df.columns.tolist()
        self.products = df
        self.compData = values["Component"].tolist()
        self.occuranceData = values

    def __call__(self, product):
        productData = self.products[product]
        components = productData.to_numpy()
        components = components[~pd.isnull(components)]
        score = 0
        for x in components:
            tempScore = self.occuranceData.query(f"Component == '{x}'")
            score += tempScore.Occurance.item() / len(self.prodData)
            score

        if self.refEngine == None:
            offsets = np.random.randint(1, 6)
        else:
            with self.refEngine.begin() as con:
                try:
                    offsets = con.execute(
                        f"SELECT * FROM 'products' WHERE product = '{product}' "
                    ).fetchall()[0][1]
                except:
                    offsets = 1
        return (self.refDB[product] * offsets, components, offsets, score)

    def getProductData(self, product):
        productData = self.products[product]
        components = productData.to_numpy()
        components = components[~pd.isnull(components)]
        return components


class KappaLoader:
    def __init__(
        self, path: Path, dbpath: str, startDate: str = None, endDate: str = None
    ) -> None:
        data = pd.read_excel(path)

        data = data[["Unnamed: 3", "Material", "VerursMenge"]]
        data["Date"] = pd.to_datetime(data["Unnamed: 3"], errors="coerce")
        if startDate is not None and endDate is not None:
            after_start_date = data["Date"] >= pd.to_datetime(startDate)
            before_end_date = data["Date"] <= pd.to_datetime(endDate)
            between_two_dates = after_start_date & before_end_date
            data = data.loc[between_two_dates]
        data = data.dropna(subset=["Material"])
        self.data = data
        databaseloader = DataBaseLoader(Path(dbpath))
        self.referenceData = databaseloader.prodData

    def __call__(self):
        return self.getData()

    def getData(self):
        rmData = self.data[~self.data["Material"].isin(self.referenceData)][
            "Material"
        ].tolist()
        if len(rmData) > 0:
            warning(
                f"REMOVING THE FOLLOWING ITEMS FROM LIST DUE TO LACK OF REFERENCE DATA: {rmData} "
            )
        self.data = self.data[self.data["Material"].isin(self.referenceData)]
        sampleList = self.data["Material"].tolist()
        sampleReqs = self.data["VerursMenge"].tolist()

        return sampleList, sampleReqs


if __name__ == "__main__":
    loader = DataBaseLoader(
        r"C:\Users\stephan.schumacher\Documents\repos\dh-reinforcement-optimization\data\SMD_Material_Stueli.txt"
    )

    test = loader("24.AAR.AB")
    test
