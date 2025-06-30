import pandas as pd
from pathlib import Path
import numpy as np
from typing import Tuple

class DataLoader:
    def __init__(self, *, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
    #这里因为很多数据表时间形式不同一  所以返回的dataframe不统一  仿真前需要先统一一下


    def load_rtm_price(self, *, filename: str = "RTM_price_2025_06_03.csv") -> pd.DataFrame:
        """load rtm price in May"""
        df = pd.read_csv(self.data_dir/filename)
        df = df.rename(columns={"Hours": "timeslot", "Price(€/MWh)": "Price"})
        df["timeslot"] = pd.to_datetime(df["timeslot"]) #转换为datetime数据类型
        df = df[["timeslot", "price"]].sort_values(by="timeslot") #按照时间顺序排列
        return df

    def load_dam_price(self, *,
                      filename: str = "Day-ahead_prices_202505010000_202506010000_Quarterhour.csv") -> pd.DataFrame:
        """load dam price in May"""
        df = pd.read_csv(self.data_dir/filename)
        start_date = df[0].astype(str)

        start_time_raw = df[1].astype(str).str.split(";", expand=True)[0]  # 2025 12:00 AM;May 1 舍去May 1
        end_time_price_split = df[2].astype(str).str.split(";", expand=True)

        end_time_raw = end_time_price_split[0]
        price = pd.to_numeric(end_time_price_split[1], errors="coerce")

        start_time = pd.to_datetime(start_date + "" + start_time_raw, errors="coerce", dayfirst=True)
        # 只用start_time作为timeslot
        result = pd.DataFrame({
            "timeslot": start_time,
            "price": price,
        }).dropna().sort_values(by="timeslot")
        return result

    def load_agc_signal(self, *, 
                       filename: str = "SRL_Soll_20250501_20250531.csv",
                       resample_freq: str = "15T") -> Tuple[pd.DataFrame, float, float]:
        """load agc signal and generate K"""
        df = pd.read_csv(
            self.data_dir / filename,
            sep=";",
            names=["date", "time", "value"],
            skiprows=1
        )

        df["timeslot"] = pd.to_datetime(df["date"] + " " + df["time"], dayfirst=True)
        df = df[["timeslot", "value"]] #只保留时间戳和值列
        df = df.set_index("timeslot").sort_index()  

        df_15min = df.resample(resample_freq).mean()

        df_15min["agc_up"] = df_15min["value"].clip(lower=0)
        df_15min["agc_dn"] = (-df_15min["value"]).clip(upper=0)  #这里设置了下调信号为负值 单位从表格来看是MW

        K_up = df_15min["agc_up"].max()*0.25  # *0.25 transfer MW to MWh
        K_dn = df_15min["agc_dn"].max()*0.25

        return df_15min[["agc_up", "agc_dn"]], K_up, K_dn
        #这里的逻辑是 每15分钟取上下调的平均值 再取出最大值乘时间转化为MWh 以用于后边的容量预存  这里的agc_up就是论文里的l_up
         
    def load_capacity_price(self, *, filename: str = "aFRR_prices_202505010000_202506010000.csv") -> pd.DataFrame:
        """load aFRR capacity price"""  "投标价"
        df = pd.read_csv(self.data_dir/filename)
        df["timeslot"] = pd.to_datetime(df["date"] + " " + df["time"])
        
        result = df.rename(columns={
            "up_price": "afrr_up_cap_price",
            "down_price": "afrr_dn_cap_price"
        })
        
        return result[["timeslot", "afrr_up_cap_price", "afrr_dn_cap_price"]].sort_values("timeslot")
        #这里先只搭了一个框架  后续再补

    def load_balancing_energy_and_price(self, filename: str = "Balancing_energy_202505010000_202506010000_Quarterhour.csv") -> pd.DataFrame:
        """
        读取balancing energy和价格数据  "激活价"
        返回DataFrame，包含：start_time, end_time, volume_pos, volume_neg, balancing_price
        """
        df = pd.read_csv(self.data_dir / filename, sep=";")
        # 解析时间
        df["start_time"] = pd.to_datetime(df["Start date"], errors="coerce")
        df["end_time"] = pd.to_datetime(df["End date"], errors="coerce")

        # 解析价格
        df["balancing_price"] = pd.to_numeric(df["Price [€/MWh] Original resolutions"].replace("--", np.nan), errors="coerce")

        return df[["start_time", "end_time", "balancing_price"]]



    def load_ev_profiles(self, *, 
                        num_evs: int = 400,
                        discount: float = 0.2, 
                        charging_price: float = 180,  #这个基础充电价格和折扣比例参考了论文
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





