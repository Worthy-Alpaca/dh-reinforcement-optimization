import re
import numpy as np
import pandas as pd
from logging import warning
from pathlib import Path
from sqlalchemy import engine


class DataBaseLoader:
    def __init__(self, pathToProductTxT: Path, refEngine: engine = None) -> None:
        """Class to load data from a txt file.

        Args:
            pathToProductTxT (Path): The path to the data source.
            refEngine (engine, optional): SQL Engine for reference data. Defaults to None.
        """
        file = open(pathToProductTxT, "r")
        self.refEngine = refEngine

        def createCleanData(x: str, header: bool = False):
            pattern = re.compile("[0-9]{7}-[0-9]{1,}")
            x = x.replace(" ", "")
            if len(x) < 7:
                return np.nan, 0
            if not header:
                if not bool(re.match(pattern, x)):
                    return np.nan, 0
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

        def count_and_append(a):
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

    def __call__(self, product: str):
        """Method to generate all needed data by a given product ID.

        Args:
            product (str): The current product ID.

        Returns:
            tuple: Placements amount, component list, offset list, score
        """
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
        return (self.refDB[product], components, offsets, score)

    def getProductData(self, product: str):
        """Method to retrieve only the components for a given product ID.

        Args:
            product (str): The current product ID.

        Returns:
            np.ndarray: The retrieved components.
        """
        productData = self.products[product]
        components = productData.to_numpy()
        components = components[~pd.isnull(components)]
        return components


class KappaLoader:
    def __init__(
        self, path: Path, dbpath: str, startDate: str = None, endDate: str = None
    ) -> None:
        """Class to load all needed data from an exported SAP excel list.

        Args:
            path (Path): Path to the excel spreadsheet.
            dbpath (str): Path to the data source.
            startDate (str, optional): The given start date. Defaults to None.
            endDate (str, optional): The given end date. Defaults to None.
        """
        data = pd.read_excel(path)

        data = data[["Unnamed: 3", "Material", "VerursMenge", "Kurztext"]]
        data["Date"] = pd.to_datetime(data["Unnamed: 3"], errors="coerce")
        if startDate is not None and endDate is not None:
            after_start_date = data["Date"] >= pd.to_datetime(startDate)
            before_end_date = data["Date"] <= pd.to_datetime(endDate)
            between_two_dates = after_start_date & before_end_date
            data = data.loc[between_two_dates]
        data = data.dropna(subset=["Material"])
        self.data = data.drop_duplicates()
        databaseloader = DataBaseLoader(Path(dbpath))
        self.referenceData = databaseloader.prodData

    def __call__(self):
        return self.getData()

    def getData(self):
        """Method to return all needed data.

        Returns:
            tuple: Sample list, requirements list, short text array.
        """
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

        return (
            sampleList,
            sampleReqs,
            self.data[["Material", "Kurztext"]].drop_duplicates().to_numpy(),
        )


if __name__ == "__main__":
    loader = DataBaseLoader(
        r"C:\Users\stephan.schumacher\Documents\repos\dh-reinforcement-optimization\data\SMD_Material_Stueli.txt"
    )

    test = loader("24.AAR.AB")
    test
