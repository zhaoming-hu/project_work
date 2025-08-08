import pandas as pd
from pathlib import Path
import numpy as np
from typing import Tuple

class DataLoader:
    def __init__(self, *, data_dir: str = "../data"):
        self.data_dir = Path(data_dir)

    def load_rtm_price(self, *, filename: str = "RTM_price_2025_06_30.csv") -> pd.DataFrame:
        """读取实时市场价格"""
        df = pd.read_csv(self.data_dir/filename, encoding='utf-8-sig')
        df.columns = df.columns.str.strip()
        df = df.rename(columns={"Timeslot": "timeslot", "Price(€/MWh)": "price"})
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df = df[["timeslot", "price"]].sort_values(by="timeslot")
        return df

    def load_dam_price(self, *, filename: str = "DAM_price_2025_06_29.csv") -> pd.DataFrame:
        """读取日前市场价格"""
        df = pd.read_csv(self.data_dir/filename)
        df = df.rename(columns={"Timeslot": "timeslot", "Price(€/MWh)": "price"})
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df = df[["timeslot", "price"]].sort_values(by="timeslot")
        return df

    def load_agc_signal(self, filename: str = "SRL_Soll_20250501.csv") -> pd.DataFrame:
        """
        只读取AGC信号原始数据，返回DataFrame，包含timeslot和agc_delta
        """
        df = pd.read_csv(self.data_dir / filename)
        df = df.rename(columns={"Time": "timeslot", "AGC_delta(MW/MW)": "agc_delta"})
        df["timeslot"] = pd.to_datetime(df["timeslot"], dayfirst=True)
        df = df[["timeslot", "agc_delta"]].sort_values(by="timeslot")
        return df

    def load_capacity_price(self, *, filename: str = "Capacity_price_2025-05-01.csv") -> pd.DataFrame:
        """读取capacity_bids价格"""
        df = pd.read_csv(self.data_dir/filename)
        df = df.rename(columns={
            "Timeslot": "timeslot",
            "Up_price(€/MWh)": "afrr_up_cap_price",
            "Dn_price(€/MWh)": "afrr_dn_cap_price"
        })
        df["afrr_up_cap_price"] = pd.to_numeric(df["afrr_up_cap_price"], errors="coerce")
        df["afrr_dn_cap_price"] = pd.to_numeric(df["afrr_dn_cap_price"], errors="coerce")
        df = df[["timeslot", "afrr_up_cap_price", "afrr_dn_cap_price"]].sort_values(by="timeslot")
        return df

    def load_balancing_energy_and_price(self, filename: str = "Balancing_price_2025_05_01.csv") -> pd.DataFrame:
        """读取balancing price"""
        df = pd.read_csv(self.data_dir / filename)
        df = df.rename(columns={"Timeslot": "timeslot", "Price(€/MWh)": "price"})
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df = df[["timeslot", "price"]].sort_values(by="timeslot")
        return df
    
    def load_multiplier(self, filename: str = "Mileage_multiplier_2025_05_01.csv") -> pd.DataFrame:
        """读取mileage multiplier"""
        df = pd.read_csv(self.data_dir / filename)
        df = df.rename(columns={"Timeslot": "timeslot", "Multiplier_Up": "Mileage_Multiplier_Up", "Multiplier_Dn": "Mileage_Multiplier_Dn"})
        df["Mileage_Multiplier_Up"] = pd.to_numeric(df["Mileage_Multiplier_Up"], errors="coerce")
        df["Mileage_Multiplier_Dn"] = pd.to_numeric(df["Mileage_Multiplier_Dn"], errors="coerce")
        df = df[["timeslot", "Mileage_Multiplier_Up", "Mileage_Multiplier_Dn"]].sort_values(by="timeslot")
        return df






