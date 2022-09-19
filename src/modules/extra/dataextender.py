from logging import info, warning
from pathlib import Path
import random
import shutil
import os
import pandas as pd
from sqlalchemy import create_engine
import time
import numpy as np
import re

from os.path import exists


class DataExtender:
    def __init__(self, dbpath: str) -> None:
        self.dbpath = dbpath
        self.engine = create_engine(f"sqlite:///{self.dbpath}", echo=False)

    def __skipHeader(self, path):
        skip = -1
        while True:
            skip = skip + 1

            df = pd.read_excel(
                path,
                skiprows=skip,
            )
            if "No." in df.columns:
                break
            if skip > 200:
                return False

        if "D+H Artikelnummer" in df.columns:
            df = df.rename(columns={"D+H Artikelnummer": "Code"})
        elif "D+H Art.Nr." in df.columns:
            df = df.rename(columns={"D+H Art.Nr.": "Code"})

        columns = df.columns.tolist()
        sub = "X"
        x = next((s for s in columns if sub in s), None)
        columns = df.columns.tolist()
        sub = "Y"
        y = next((s for s in columns if sub in s), None)

        df = df.rename(columns={x: "X", y: "Y"})
        return df

    def __createCleanData(self, x: str):
        pattern = re.compile("[0-9]{2}\.[0-9]{3}\.[0-9]{2}")
        pattern3 = re.compile("[0-9]{7}")
        pattern2 = re.compile("(..)(...)(..)")
        x = x.replace(" ", "")
        if len(x) < 7:
            return np.nan

        if not bool(re.match(pattern2, x)):
            return np.nan

        if bool(re.match(pattern, x)):
            return x

        if not bool(re.match(pattern3, x)):
            return np.nan
        cleanString = f"{x[:2]}.{x[2:5]}.{x[5:7]} "
        return cleanString

    def checkDir(self, directory: Path):
        if not exists(directory):
            raise shutil.ExecError("Directory does not exist")
        pattern = re.compile("(..)\.(...)\.(..)")
        # all_files = [
        #     f
        #     for f in os.listdir(directory)
        #     if f.endswith(".xlx")
        #     or f.endswith(".xlsx")
        #     or f.endswith(".xls")
        #     or f.endswith(".ods")
        #     or f.endswith(".xlsm")
        # ]
        # if len(all_files) == 0:
        #     return
        info("Checking data...")

        outcome = True
        with self.engine.begin() as connection:
            refData = pd.read_sql("ReferenceData", connection)
            refProducts: list = refData["Material"]
            refOffsets: list = refData["Anzahl Abschaltungen je Nutzen"]
            connection.execute(
                f"CREATE TABLE IF NOT EXISTS 'allcomponents' (component varchar(255), occurance int)"
            )
            connection.execute(
                f"CREATE TABLE IF NOT EXISTS 'products' (product varchar(255), numOffsets int)"
            )

            path = directory
            # productName = f[:9]
            # if not bool(re.match(pattern, productName)):
            data = pd.read_excel(path, header=None).to_numpy()
            productName = str(data[0, 5])
            if not bool(re.match(pattern, productName)):
                productName = str(data[0, 6])
                if not bool(re.match(pattern, productName)):
                    warningStr = 'Unable to find Product name. Please make sure the product name is in the dedicated field "1F" and matches the normal pattern "xx.xxx.xx"'
                    warning(warningStr)
                    return False, warningStr

            if productName not in refProducts:
                warning(
                    "Unable to find product in reference table. Please enter the amount of dedicated PCBs."
                )
                outcome = None

            data = self.__skipHeader(path)
            if type(data) != pd.DataFrame:
                warningStr = "Unable to process data. Please make sure the excel file adheres to the normal standard."
                warning(warningStr)
                return False, warningStr

            data = data.dropna(subset=["Code"])

            data = data[data["Code"] != "xx.xxx.xx"]

            data["Code"] = (
                data["Code"].astype(str).apply(lambda x: self.__createCleanData(x))
            )
            data = data.dropna(subset=["Code"])

            data = data[data["No."] != "Beispiel"]

            data = data[["Code", "X", "Y"]]
            tableName = f"{productName}_placementData"
            data.to_sql(tableName, con=connection, if_exists="replace")
            info("Data checked. Data ist ok. Moving to database.")
            components = data["Code"].unique()
            if outcome:
                numOffsets = refOffsets[refProducts.index(productName)]
                connection.execute(
                    f"INSERT INTO 'products' (product, numOffsets) VALUES ('{productName}', {int(numOffsets)} )"
                )

            for x in components:
                existing = connection.execute(
                    f"SELECT * FROM 'allcomponents' WHERE component = '{x}'"
                ).fetchall()

                if len(existing) == 0:
                    connection.execute(
                        f"INSERT INTO 'allcomponents' (component, occurance) VALUES ('{x}', {1})"
                    )
                elif len(existing) == 1:
                    currentValue = existing[0][1] + 1
                    connection.execute(
                        f"UPDATE 'allcomponents' SET occurance = {currentValue} WHERE component = '{x}'"
                    )
            return outcome, productName
