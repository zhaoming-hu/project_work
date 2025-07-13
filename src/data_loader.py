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
                        num_evs: int, # 总EV数量
                        discount: float, # 充电折扣比例
                        charging_price: float,  #基础充电价格参考了论文
                        seed: int = None,
                        use_timeslot: bool = True) -> pd.DataFrame:
        """生成基准EV数据            
        Args:
            num_evs: 总EV数量
            discount: 充电折扣比例
            charging_price: 基础充电价格
            seed: 随机种子
            use_timeslot: 是否将小时时间转换为15分钟时间槽索引 (默认为True)
            
        Returns:
            pd.DataFrame: EV参数表，包含以下列：
                - ev_id: EV编号
                - ev_type: EV类型（'cc'表示可控，'uc'表示不可控）
                - charging_type: 充电类型（'day'表示白天充电，'night'表示夜间充电）
                - arrival_time: 如果use_timeslot=False，则为到达时间（小时）；否则为时间槽索引（0-95）
                - departure_time: 如果use_timeslot=False，则为离开时间（小时）；否则为时间槽索引（0-95）
                - soc_arrival: 到达时初始SOC
                - soc_departure: 离开时目标SOC
                - soc_max: 最大SOC
                - soc_min: 最小SOC
                - battery_capacity: 电池容量（MWh）
                - max_charge_power: 最大充电功率（MW）
                - efficiency: 充电效率
                - charging_price: 充电价格（€/MWh）
        """
        if seed is not None:
            np.random.seed(seed)
        
        # 计算可控EV比例
        theta = discount
        rho = max(0.0, min(1.0, 4 * theta - 0.2))  # 可控EV比例，确保在0-1之间
        num_controllable = int(rho * num_evs)
        
        # 确保白天充电和夜间充电EV数量相等
        num_day = num_evs // 2
        num_night = num_evs - num_day
        
        # 生成EV类型，确保白天和夜间充电中各有相应比例的可控车辆
        num_cc_day = int(min(rho * num_day, num_controllable))
        num_cc_night = num_controllable - num_cc_day
        
        # 创建DataFrame
        df = pd.DataFrame()
        df['ev_id'] = range(num_evs)
        
        # 设置充电类型
        df['charging_type'] = np.concatenate([['day'] * num_day, ['night'] * num_night])
        
        # 设置EV控制类型
        day_types = ['cc'] * num_cc_day + ['uc'] * (num_day - num_cc_day)
        night_types = ['cc'] * num_cc_night + ['uc'] * (num_night - num_cc_night)
        np.random.shuffle(day_types)
        np.random.shuffle(night_types)
        df['ev_type'] = np.concatenate([day_types, night_types])
        
        # 初始化参数列
        df['arrival_time'] = 0.0
        df['departure_time'] = 0.0
        df['soc_arrival'] = 0.0
        df['soc_departure'] = 0.0
        df['soc_max'] = 0.95
        df['soc_min'] = 0.1
        df['battery_capacity'] = 0.028  # 将28.0 kWh转换为0.028 MWh
        df['max_charge_power'] = 0.0066  # 将6.6 kW转换为0.0066 MW
        df['efficiency'] = 0.95
        
        # 为白天充电的EV设置参数 (连续小时时间)
        day_mask = df['charging_type'] == 'day'
        num_day_evs = sum(day_mask)
        
        df.loc[day_mask, 'arrival_time'] = np.random.normal(8.82, 1.08, num_day_evs).clip(7.5, 10.5)
        df.loc[day_mask, 'departure_time'] = np.random.normal(18.55, 2.06, num_day_evs).clip(16.0, 21.0)
        df.loc[day_mask, 'soc_arrival'] = np.random.uniform(0.2, 0.35, num_day_evs)
        df.loc[day_mask, 'soc_departure'] = np.random.uniform(0.85, 0.95, num_day_evs)
        
        # 为夜间充电的EV设置参数 (连续小时时间)
        night_mask = df['charging_type'] == 'night'
        num_night_evs = sum(night_mask)
        
        df.loc[night_mask, 'arrival_time'] = np.random.normal(22.91, 3.2, num_night_evs).clip(20.9, 24.0)
        df.loc[night_mask, 'departure_time'] = np.random.normal(8.95, 1.4, num_night_evs).clip(7.0, 10.0)
        df.loc[night_mask, 'soc_arrival'] = np.random.uniform(0.2, 0.35, num_night_evs)
        df.loc[night_mask, 'soc_departure'] = np.random.uniform(0.85, 0.95, num_night_evs)
        
        # 设置充电价格（可控EV享受折扣）
        df['charging_price'] = np.where(
            df['ev_type'] == 'cc',
            charging_price * (1 - discount),
            charging_price
        )
        
        # 如果需要，将连续小时时间转换为15分钟时间槽索引 (0-95)
        if use_timeslot:
            # 创建源时间列的副本
            df['hour_arrival'] = df['arrival_time'].copy()
            df['hour_departure'] = df['departure_time'].copy()
            
            # 转换到达时间：小时 → 时间槽索引
            df['arrival_time'] = (df['arrival_time'] * 4).astype(int)
            
            # 特殊处理夜间充电的离开时间：
            # 对于夜间充电，离开时间小于到达时间，表示跨天，需要加上96来表示第二天
            df.loc[night_mask, 'departure_time'] = (df.loc[night_mask, 'departure_time'] * 4).astype(int)
            
            # 对于白天充电，直接转换
            df.loc[day_mask, 'departure_time'] = (df.loc[day_mask, 'departure_time'] * 4).astype(int)
            
            # 确保所有索引在有效范围内 (0-95)
            df['arrival_time'] = df['arrival_time'].clip(0, 95)
            df['departure_time'] = df['departure_time'].clip(0, 95)
        
        return df





