import os
import sqlalchemy
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine


class DataLoader:
    """
    Returns a tuple that contains all needed DataFrames

    `returns:` (`data`, `components`, `offsets`)
    """

    def __init__(self, data_folder: Path, separator: str = ","):
        """Initializes a new DataLoader instance.

        Args:
            data_folder (Path): Path to the current SMD Program. May change to a DB connection.
            separator (str, optional): The seperator to be used. Defaults to ",".
        """
        matchers = ["Cmp", "Kyu", "Tou"]
        matching = [
            s for s in os.listdir(data_folder) if any(xs in s for xs in matchers)
        ]
        global_string = Path(
            os.getcwd() + os.path.normpath("/global/Components width.csv")
        )
        self.global_Feeder_Data = pd.read_csv(global_string, sep=";")

        for i in matching:
            skip = -1
            while True:
                skip = skip + 1

                df = pd.read_csv(
                    str(data_folder) + "/" + i,
                    sep=separator,
                    skiprows=skip,
                    encoding="unicode_escape",
                )
                if "Component Code" in df.columns:
                    break

            if "Cmp" in i:
                self.product_components_data = df
            elif "Kyu" in i:
                self.product_feeder_data = df
            elif "Tou" in i:
                self.product_data = df

    def __call__(
        self, *args: any, **kwds: any
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generate the needed Data for an assembly simulation.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Returns data, components, offsetlist
        """
        neededColumns_Data = ["Component Code", "X", "Y", "Task"]
        neededColumns_Components = [
            "Component Code",
            "Placement(Acceleration):X",
            "Placement(Acceleration):Y",
            "Placement(Acceleration):Z",
            "Priority Nozzle No.",
        ]
        neededColumns_Feeder = ["Component Code", "FeedStyle", "ST No."]
        data = self.product_data[neededColumns_Data]
        components_data = self.product_components_data[neededColumns_Components]
        components_feeder_data = self.product_feeder_data[neededColumns_Feeder]
        data = pd.merge(
            left=data,
            left_on="Component Code",
            right=components_data,
            right_on="Component Code",
            how="left",
        )
        components_data = pd.merge(
            left=components_data,
            left_on="Component Code",
            right=components_feeder_data,
            right_on="Component Code",
            how="left",
        )

        data = data.rename(columns={"Component Code": "Code"})
        components_data = components_data.rename(
            columns={
                "Component Code": "index",
                "Priority Nozzle No.": "Nozzle_No",
                "ST No.": "ST_No",
                "Placement(Acceleration):Z": "Dropoff",
            }
        )
        components_feeder_data = components_feeder_data.rename(
            columns={"Component Code": "index"}
        )

        # replace commas with decimal points
        data["X"] = data["X"].replace({",": "."}, regex=True).astype(float)
        data["Y"] = data["Y"].replace({",": "."}, regex=True).astype(float)

        components_data["FeedStyle"] = components_data["FeedStyle"].replace(
            {"^ST-F$": "ST-FL", "^ST-R$": "ST-RL"}, regex=True
        )

        # calculate the mean of X and Y acceleration
        components_data["mean_acceleration"] = components_data[
            ["Placement(Acceleration):X", "Placement(Acceleration):Y"]
        ].mean(axis=1)
        components_data["mean_acceleration"] = components_data["mean_acceleration"]
        # devide coordinates
        if data["X"].max() > 1000:
            data["X"] = data["X"] / 1000
            data["Y"] = data["Y"] / 1000

        # split offset and drop duplicates
        offsets = data.loc[data["Task"] == "Repeat Offset"]
        zero_offset = pd.DataFrame(
            {"Code": "", "X": 0, "Y": 0, "Task": "Repeat Offset"}, index=[0]
        )
        offsets = pd.concat([zero_offset, offsets], axis=0)
        offsets = offsets.drop_duplicates()
        offsets = offsets.reindex()
        offsetlist = []
        for index, row in offsets.iterrows():
            offset = (row.X, row.Y)
            offsetlist.append(offset)

        data = data.loc[data["Task"] != "B Mark Positive Logic"]
        fid = data[(data.Task == "Fiducial")]
        fid = fid.rename(columns={"Code": "index"})

        # create component dataset
        occ = data["Code"].value_counts()
        components = pd.DataFrame(occ, columns=["Code"]).reset_index()
        # create pickup coordinates
        components["Pickup_Y"] = 0
        components["Pickup_X"] = range(len(components.index))
        components = pd.merge(
            left=components,
            right=components_data,
            left_on="index",
            right_on="index",
            how="left",
        ).drop_duplicates()
        components = pd.merge(
            left=components,
            right=self.global_Feeder_Data,
            left_on="index",
            right_on="Component Code",
            how="left",
        ).drop_duplicates()
        components = components.drop(["Component Code"], axis=1)
        components["mean_acceleration"] = components["mean_acceleration"].fillna(1000.0)

        data = data.dropna()

        data.columns = data.columns.str.replace(" ", "_")
        components.columns = components.columns.str.replace(" ", "_")
        self.data = data
        self.components = components
        self.offsets = offsetlist
        return (data, components, offsetlist)


class DataBaseLoader:
    def __init__(
        self,
        dataBase: sqlalchemy.engine.base.Engine,
        product: str,
        allProducts: list,
    ) -> None:
        """Load the required Data from the ``products.db`` database.

        Args:
            dataBase (sqlalchemy.engine.base.Engine): The current database connection.
            product (str): The current product name.
            allProducts (list): List of all available products.
        """
        with dataBase.begin() as connection:
            tableName = f"{product}_placementData"
            self.data = pd.read_sql_table(tableName, connection)
            self.data["X"] = (
                self.data["X"]
                .replace({"\D+": ""}, regex=True)
                .replace({",": "."}, regex=True)
                .astype(float)
            )
            self.data["Y"] = (
                self.data["Y"]
                .replace({"\D+": ""}, regex=True)
                .replace({",": "."}, regex=True)
                .astype(float)
            )
            self.components = self.data["Code"].unique()

            try:
                numOffsets = connection.execute(
                    f"SELECT * FROM 'products' WHERE product = '{product}' "
                ).fetchall()[0][1]
            except:
                numOffsets = 1

            self.offsets = numOffsets

            score = 0
            allProducts = len(allProducts)
            for x in self.components:
                tempScore = connection.execute(
                    f"SELECT * FROM 'allcomponents' WHERE component = '{x}'"
                ).fetchall()[0][1]
                score += tempScore / allProducts

            self.score = score

            # tableName = f"{product}_{machine}_components"
            # self.components = pd.read_sql_table(tableName, connection)
            # tableName = f"{product}_{machine}_offsets"
            # offsets = pd.read_sql_table(tableName, connection)
            # offsetList = []
            # for i, r in offsets.iterrows():
            #     currentOffset = (r.x, r.y)
            #     offsetList.append(currentOffset)
            # self.offsets = offsetList

    def __call__(self):
        return (self.data, self.components, self.offsets, self.score)


class KappaLoader:
    def __init__(self, path: Path) -> None:
        data = pd.read_excel(path)
        print(data.info())
        data = data[["Material", "VerursMenge"]]
        data = data.dropna(subset=["Material"])
        self.data = data
        engine = create_engine("sqlite:///products.db", echo=False)
        with engine.begin() as con:
            self.referenceData = pd.read_sql_table("products", con=con)
            self.referenceData = self.referenceData["product"].tolist()

    def __call__(self):
        return self.getData()

    def getData(self):
        print("REMOVING THE FOLLOWING ITEMS FROM LIST DUE TO LACK OF REFERENCE DATA:")
        print(
            self.data[~self.data["Material"].isin(self.referenceData)][
                "Material"
            ].tolist()
        )
        self.data = self.data[self.data["Material"].isin(self.referenceData)]
        sampleList = self.data["Material"].tolist()
        sampleReqs = self.data["VerursMenge"].tolist()

        return sampleList, sampleReqs


if __name__ == "__main__":
    path = Path(os.getcwd() + os.path.normpath("/export.xlsx"))
    loader = KappaLoader(path)
    print(loader())
