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


    def load_ev_profiles(self, *, 
                        num_evs: int,
                        discount: float, 
                        charging_price: float,  #这个基础充电价格和折扣比例参考了论文
                        seed: int = None) -> pd.DataFrame:
        """生成基准EV数据
        
        Args:
            num_evs: 电动汽车数量
            discount: 充电折扣比例（如0.2表示8折）
            charging_price: 基础充电价格（$/MWh）
            seed: 随机种子
            
        Returns:
            pd.DataFrame: EV参数表，包含以下列：
                - ev_id: EV编号
                - ev_type: EV类型（'cc'表示可控，'uc'表示不可控）
                - arrival_time: 到达时间（小时）
                - departure_time: 离开时间（小时）
                - soc_arrival: 到达时初始SOC
                - soc_departure: 离开时目标SOC
                - soc_max: 最大SOC
                - soc_min: 最小SOC
                - battery_capacity: 电池容量（kWh）
                - max_charge_power: 最大充电功率（kW）
                - efficiency: 充电效率
                - charging_price: 充电价格（$/MWh）
        """
        if seed is not None:
            np.random.seed(seed)
        
        # 计算可控EV比例
        theta = discount
        rho = min(1.0, 4 * theta + 0.2)  # 可控EV比例
        num_controllable = int(rho * num_evs)
        
        # 生成EV类型
        ev_types = np.array(['cc'] * num_controllable + ['uc'] * (num_evs - num_controllable))
        np.random.shuffle(ev_types)
        
        # 创建DataFrame
        df = pd.DataFrame()
        df['ev_id'] = range(num_evs)
        df['ev_type'] = ev_types
        
        # 生成到达时间（假设大多数EV在早上8-10点到达）
        df['arrival_time'] = np.random.normal(8.82, 1.08, num_evs).clip(7.5, 10.5)
        
        # 生成离开时间（假设大多数EV在下午4-9点离开）
        df['departure_time'] = np.random.normal(18.55, 2.06, num_evs).clip(16.0, 21.0)
        
        # 生成初始SOC（假设大多数EV到达时SOC在20%-35%之间）
        df['soc_arrival'] = np.random.uniform(0.2, 0.35, num_evs)
        
        # 生成目标SOC（假设大多数EV希望充电到85%-95%）
        df['soc_departure'] = np.random.uniform(0.85, 0.95, num_evs)
        
        # 设置SOC限制
        df['soc_max'] = 0.95  # 最大SOC限制
        df['soc_min'] = 0.1   # 最小SOC限制
        
        # 设置电池参数（假设所有EV使用相同的电池）
        df['battery_capacity'] = 28  # 电池容量（kWh）
        df['max_charge_power'] = 6.6  # 最大充电功率（kW）
        df['efficiency'] = 0.95  # 充电效率
        
        # 设置充电价格（可控EV享受折扣）
        df['charging_price'] = np.where(
            df['ev_type'] == 'cc',
            charging_price * (1 - discount),
            charging_price
        )
        
        return df





