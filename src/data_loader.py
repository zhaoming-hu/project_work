import pandas as pd
from pathlib import Path

class DataLoader:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)

    def load_rtm_price(self) -> pd.DataFrame:
        """load rtm price in May"""
        df = pd.read_csv(self.data_dir/"RTM_price_2025_06_03.csv")
        df = df.rename(columns={"Hours": "timeslot", "Price(€/MWh)": "Price"})
        df["timeslot"] = pd.to_datetime(df["timeslot"])
        df = df[["timeslot", "price"]].sort_values(by="timeslot")
        return df

    def load_dam_price(self) -> pd.DataFrame:
        """load dam price in May"""
        df = pd.read_csv(self.data_dir/"Day-ahead_prices_202505010000_202506010000_Quarterhour.csv")
        start_date = df[0].astype(str)

        start_time_raw = df[1].astype(str).str.split(";", expand=True)[0]
        end_time_price_split = df[2].astype(str).str.split(";", expand=True)

        end_time_raw = end_time_price_split[0]
        price = pd.to_numeric(end_time_price_split[1], errors="coerce")

        start_time = pd.to_datetime(start_date + "" + start_time_raw, errors="coerce", dayfirst=True)
        end_time = pd.to_datetime(start_date + "" + end_time_raw, errors="coerce", dayfirst=True)

        result = pd.DataFrame(
            {
                "start_time": start_time,
                "end_time": end_time,
                "price": price,
            }
        ).dropna().sort_values(by="start_time")

        return result

    def load_agc_signal(self) -> pd.DataFrame:
        """load agc signal and generate K"""
        df = pd.read_csv(
            self.data_dir /"SRL_Soll_20250501_20250531.csv",
            sep=";",
            names=["date", "time", "value"],
            skiprows=1
        )

        df["timeslot"] = pd.to_datetime(df["date"] + " " + df["time"], dayfirst=True)
        df = df[["timeslot", "value"]] #structured new dataframe
        df = df.set_index("timeslot").sort_index

        df_15min = df.resample("15T").mean()

        df_15min["agc_up"] = df_15min["value"].clip(lower=0)
        df_15min["agc_dn"] = (-df_15min["value"]).clip(upper=0)

        df_15min["K_up"] = df_15min.groupby(df_15min.index.time)["agc_up"].transform("max")*0.25  # *0.25 transfer MW to MWh
        df_15min["K_dn"] = df_15min.groupby(df_15min.index.time)["agc_dn"].transform("max")*0.25

        return df_15min[["agc_up", "agc_dn" "K_up", "K_dn"]]

    def load_capacity_price(self):

    def load_ev_profiles(self):




