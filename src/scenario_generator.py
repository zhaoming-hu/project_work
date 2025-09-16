import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from typing import Dict, List, Tuple, Optional

class ScenarioGenerator:
    def __init__(self, *, 
                 num_scenarios: int,
                 num_clusters: int,
                 seed: int = None):
        """初始化场景生成器
        
        Args:
            num_scenarios: 初始场景数量
            num_clusters: K-means聚类后的场景数量
            seed: 随机种子
        """
        self.num_scenarios = num_scenarios
        self.num_clusters = num_clusters
        self.seed = seed
        # 使用局部随机数生成器，避免污染全局随机状态
        self.rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    
    def generate_base_ev_profiles(self, *,
                                  num_evs: int,
                                  discount: float,
                                  charging_price: float,
                                  use_timeslot: bool = True) -> pd.DataFrame:
        """生成基准EV数据（使用本生成器的随机源），确保与场景生成共享同一seed。

        Args:
            num_evs: 总EV数量
            discount: 充电折扣比例
            charging_price: 基础充电价格（€/MWh）
            use_timeslot: 是否将小时时间转换为15分钟时间槽索引

        Returns:
            pd.DataFrame: EV参数表
        """
        rng = self.rng

        # 计算可控EV比例
        theta = discount
        rho = max(0.0, min(1.0, 4 * theta - 0.2))
        num_controllable = int(rho * num_evs)

        # 白天/夜间数量
        num_day = num_evs // 2
        num_night = num_evs - num_day

        # 分配可控数量
        num_cc_day = int(min(rho * num_day, num_controllable))
        num_cc_night = num_controllable - num_cc_day

        df = pd.DataFrame()
        df['ev_id'] = range(num_evs)

        # 充电类型
        df['charging_type'] = np.concatenate([['day'] * num_day, ['night'] * num_night])

        # 控制类型并打乱
        day_types = ['cc'] * num_cc_day + ['uc'] * (num_day - num_cc_day)
        night_types = ['cc'] * num_cc_night + ['uc'] * (num_night - num_cc_night)
        rng.shuffle(day_types)
        rng.shuffle(night_types)
        df['ev_type'] = np.concatenate([day_types, night_types])

        # 初始化参数
        df['arrival_time'] = 0.0
        df['departure_time'] = 0.0
        df['soc_arrival'] = 0.0
        df['soc_departure'] = 0.0
        df['soc_max'] = 0.95
        df['soc_min'] = 0.1
        df['battery_capacity'] = 0.028  # MWh（28 kWh）
        df['max_charge_power'] = 0.0066  # MW（6.6 kW）
        df['efficiency'] = 0.95

        # 掩码
        day_mask = df['charging_type'] == 'day'
        night_mask = df['charging_type'] == 'night'
        uc_mask = df['ev_type'] == 'uc'
        cc_mask = df['ev_type'] == 'cc'

        # UC：单独设置晚上到达（16:00-20:00），标记为 day，避免跨天逻辑
        num_uc = int(uc_mask.sum())
        if num_uc > 0:
            uc_arrival = rng.normal(18.0, 2.06, num_uc).clip(16.0, 20.0)
            df.loc[uc_mask, 'arrival_time'] = uc_arrival
            df.loc[uc_mask, 'charging_type'] = 'day'  
            df.loc[uc_mask, 'soc_arrival'] = rng.uniform(0.2, 0.35, num_uc)
            df.loc[uc_mask, 'soc_departure'] = rng.uniform(0.85, 0.95, num_uc)

        # CC 且 day：白天参数（小时）
        day_cc_mask = day_mask & cc_mask
        num_day_cc = int(day_cc_mask.sum())
        if num_day_cc > 0:
            df.loc[day_cc_mask, 'arrival_time'] = rng.normal(8.82, 1.08, num_day_cc).clip(7.5, 10.5)
            df.loc[day_cc_mask, 'departure_time'] = rng.normal(18.55, 2.06, num_day_cc).clip(16.0, 21.0)
            df.loc[day_cc_mask, 'soc_arrival'] = rng.uniform(0.2, 0.35, num_day_cc)
            df.loc[day_cc_mask, 'soc_departure'] = rng.uniform(0.85, 0.95, num_day_cc)

        # CC 且 night：夜间参数（小时）
        night_cc_mask = night_mask & cc_mask
        num_night_cc = int(night_cc_mask.sum())
        if num_night_cc > 0:
            df.loc[night_cc_mask, 'arrival_time'] = rng.normal(22.91, 3.2, num_night_cc).clip(20.9, 24.0)
            df.loc[night_cc_mask, 'departure_time'] = rng.normal(8.95, 1.4, num_night_cc).clip(7.0, 10.0)
            df.loc[night_cc_mask, 'soc_arrival'] = rng.uniform(0.2, 0.35, num_night_cc)
            df.loc[night_cc_mask, 'soc_departure'] = rng.uniform(0.85, 0.95, num_night_cc)

        # 充电价格（可控享受折扣）
        df['charging_price'] = np.where(
            df['ev_type'] == 'cc',
            charging_price * (1 - discount),
            charging_price
        )

        # 转换为15分钟时间槽
        if use_timeslot:
            df['hour_arrival'] = df['arrival_time'].copy()
            df['hour_departure'] = df['departure_time'].copy()

            df['arrival_time'] = (df['arrival_time'] * 4).astype(int)
            # 夜间（仅 CC 夜间）跨天
            df.loc[night_cc_mask, 'departure_time'] = (df.loc[night_cc_mask, 'departure_time'] * 4).astype(int) + 96
            # 其他（含 UC 和 CC 白天）不跨天
            non_overnight_mask = ~night_cc_mask
            df.loc[non_overnight_mask, 'departure_time'] = (df.loc[non_overnight_mask, 'departure_time'] * 4).astype(int)

            df['arrival_time'] = df['arrival_time'].clip(0, 95)
            # 非跨天（含 UC）
            df.loc[non_overnight_mask, 'departure_time'] = df.loc[non_overnight_mask, 'departure_time'].clip(0, 95)
            # 跨天（仅 CC 夜间）
            df.loc[night_cc_mask, 'departure_time'] = df.loc[night_cc_mask, 'departure_time'].clip(96, 191)

        return df
            
    def generate_ev_scenarios(self, *, ev_profiles: pd.DataFrame) -> List[pd.DataFrame]:
        """生成EV场景，每个场景都独立采样，符合原始分布
        
        Args:
            ev_profiles: 基准EV数据，包含充电类型（day/night）
            
        Returns:
            List[pd.DataFrame]: EV场景列表
        """
        scenarios = []
        num_evs = len(ev_profiles)
        
        # 检查输入数据是使用时间槽索引还是连续小时
        use_timeslot = 'hour_arrival' in ev_profiles.columns
        
        for _ in range(self.num_scenarios):
            scenario = pd.DataFrame()
            scenario['ev_id'] = range(num_evs)
            scenario['ev_type'] = ev_profiles['ev_type'].values
            scenario['charging_type'] = ev_profiles['charging_type'].values
            
            # 分别处理白天和夜间充电的EV
            day_mask = scenario['charging_type'] == 'day'
            night_mask = scenario['charging_type'] == 'night'
            uc_mask = ev_profiles['ev_type'].values == 'uc'
            
            num_day_evs = sum(day_mask)
            num_night_evs = sum(night_mask)
            
            # 初始化所有EV的参数
            scenario['arrival_time'] = 0.0
            scenario['departure_time'] = 0.0
            scenario['soc_arrival'] = 0.0
            scenario['soc_departure'] = 0.0
            scenario['soc_max'] = ev_profiles['soc_max'].values
            scenario['soc_min'] = ev_profiles['soc_min'].values
            scenario['battery_capacity'] = ev_profiles['battery_capacity'].values  # 已转换为MWh
            scenario['max_charge_power'] = ev_profiles['max_charge_power'].values  # 已转换为MW
            scenario['efficiency'] = ev_profiles['efficiency'].values
            scenario['charging_price'] = ev_profiles['charging_price'].values
            
            # 为白天充电的EV生成场景参数（连续小时时间）
            scenario.loc[day_mask, 'arrival_time'] = self.rng.normal(8.82, 1.08, num_day_evs).clip(7.5, 10.5)
            scenario.loc[day_mask, 'departure_time'] = self.rng.normal(18.55, 2.06, num_day_evs).clip(16.0, 21.0)
            scenario.loc[day_mask, 'soc_arrival'] = self.rng.uniform(0.2, 0.35, num_day_evs).clip(0.2, 0.35)
            scenario.loc[day_mask, 'soc_departure'] = self.rng.uniform(0.85, 0.95, num_day_evs).clip(0.85, 0.95)
            
            # 为夜间充电的EV生成场景参数（连续小时时间）
            scenario.loc[night_mask, 'arrival_time'] = self.rng.normal(22.91, 3.2, num_night_evs).clip(20.9, 24.0)
            scenario.loc[night_mask, 'departure_time'] = self.rng.normal(8.95, 1.4, num_night_evs).clip(7.0, 10.0)
            scenario.loc[night_mask, 'soc_arrival'] = self.rng.uniform(0.2, 0.35, num_night_evs).clip(0.2, 0.35)
            scenario.loc[night_mask, 'soc_departure'] = self.rng.uniform(0.85, 0.95, num_night_evs).clip(0.85, 0.95)

            # 覆盖 UC 到达时间：正态分布并截断到傍晚 16:00-20:00
            num_uc = int(uc_mask.sum())
            if num_uc > 0:
                uc_arrival = self.rng.normal(18.0, 2.06, num_uc).clip(16.0, 20.0)
                uc_departure = self.rng.normal(23.0, 2.06, num_uc).clip(23.0, 24.0)
                uc_soc_arrival = self.rng.uniform(0.3, 0.35, num_uc).clip(0.2, 0.95)
                uc_soc_departure = self.rng.uniform(0.85, 0.95, num_uc).clip(0.85, 0.95)
                scenario.loc[uc_mask, 'arrival_time'] = uc_arrival
                scenario.loc[uc_mask, 'departure_time'] = uc_departure
                scenario.loc[uc_mask, 'soc_arrival'] = uc_soc_arrival
                scenario.loc[uc_mask, 'soc_departure'] = uc_soc_departure

            # 如果输入数据使用了时间槽索引，则将连续小时时间转换为15分钟时间槽索引 (0-95)
            # if use_timeslot:
            # 创建源时间列的副本
            scenario['hour_arrival'] = scenario['arrival_time'].copy()
            scenario['hour_departure'] = scenario['departure_time'].copy()
            
            # 转换到达时间：小时 → 时间槽索引
            scenario['arrival_time'] = (scenario['arrival_time'] * 4).astype(int)
            
            # 特殊处理夜间充电的离开时间：
            # 对于夜间充电，离开时间小于到达时间，表示跨天，需要加上96来表示第二天
            scenario.loc[night_mask, 'departure_time'] = (scenario.loc[night_mask, 'departure_time'] * 4).astype(int) + 96
            
            # 对于白天充电，直接转换
            scenario.loc[day_mask, 'departure_time'] = (scenario.loc[day_mask, 'departure_time'] * 4).astype(int)
            
            # 到达时间限制在正确范围内
            scenario['arrival_time'] = scenario['arrival_time'].clip(0, 95)
            
            # 仅对白天充电的EV限制departure_time范围
            scenario.loc[day_mask, 'departure_time'] = scenario.loc[day_mask, 'departure_time'].clip(0, 95)
            # 对于跨天的夜间充电EV，保留其跨天信息，确保departure_time > 95
            scenario.loc[night_mask, 'departure_time'] = scenario.loc[night_mask, 'departure_time'].clip(96, 191)

            scenarios.append(scenario)
        return scenarios
    
    def generate_price_scenarios(self, *,
                               dam_prices: pd.DataFrame,
                               rtm_prices: pd.DataFrame,
                               capacity_price: pd.DataFrame,
                               balancing_prices: pd.DataFrame,
                               mileage_multiplier: pd.DataFrame,
                               T: int) -> List[pd.DataFrame]:
        """生成价格场景，每个场景为DataFrame
        Args:
            dam_prices: 日前市场价格
            rtm_prices: 实时市场价格
            capacity_price: 容量投标价格
            balancing_prices: 激活价格
            mileage_multiplier: 里程乘数
            T: 时间步数
        Returns:
            List[pd.DataFrame]: 价格场景列表
        """
        scenarios = []
        for _ in range(self.num_scenarios):
            dam_noise = self.rng.normal(1, 0.1, T)
            rtm_noise = self.rng.normal(1, 0.15, T)
            afrr_up_cap_noise = self.rng.normal(1, 0.2, T)
            afrr_dn_cap_noise = self.rng.normal(1, 0.2, T)
            balancing_noise = self.rng.normal(1, 0.1, T)
            # 乘数不添加随机性，保持原值
            mileage_multiplier_up = mileage_multiplier['Mileage_Multiplier_Up'].values
            mileage_multiplier_dn = mileage_multiplier['Mileage_Multiplier_Dn'].values

            scenario = pd.DataFrame({
                'dam_prices': dam_prices['price'].values * dam_noise,
                'rtm_prices': rtm_prices['price'].values * rtm_noise,
                'afrr_up_cap_prices': capacity_price['afrr_up_cap_price'].values* afrr_up_cap_noise,
                'afrr_dn_cap_prices': capacity_price['afrr_dn_cap_price'].values* afrr_dn_cap_noise,
                'balancing_prices': balancing_prices['price'].values* balancing_noise,
                'mileage_multiplier_up': mileage_multiplier_up,
                'mileage_multiplier_dn': mileage_multiplier_dn
            })

            scenarios.append(scenario)
        return scenarios
    
    def generate_agc_scenarios(self, *,
                             agc_signals: pd.DataFrame,
                             T: int,
                             resample_freq: str = "15min") -> Tuple[List[pd.DataFrame], pd.DataFrame]:
        """
        生成AGC信号场景，并为每个时间段计算单独的K_up和K_dn容量预留值

        Args:
            agc_signals: 原始AGC信号数据（含timeslot, agc_delta）
            T: 时间步数
            resample_freq: 重采样频率，默认15分钟

        Returns:
            List[pd.DataFrame]: AGC信号场景列表（平均值，未乘以0.25）
            pd.DataFrame: 每个时间段的K_up/K_dn容量（乘以0.25后的MWh/MW）
        """
        def pos_mean(x):
            return x[x > 0].mean() if (x > 0).any() else 0.0
        def neg_mean(x):
            return -x[x < 0].mean() if (x < 0).any() else 0.0

        # 确保 timeslot 是 datetime 类型
        if not pd.api.types.is_datetime64_any_dtype(agc_signals['timeslot']):
            agc_signals['timeslot'] = pd.to_datetime(agc_signals['timeslot'])

        # 添加日期和时间段索引
        agc_signals['date'] = agc_signals['timeslot'].dt.date
        agc_signals['time_of_day'] = agc_signals['timeslot'].dt.hour * 4 + agc_signals['timeslot'].dt.minute // 15

        # 按日和时间段分组，计算每15分钟的平均AGC_delta（未乘以0.25）
        daily_stats = agc_signals.groupby(['date', 'time_of_day'])['agc_delta'].agg(
            agc_up=pos_mean,
            agc_dn=neg_mean
        ).reset_index()

        # 计算每时间段30天内最大值，乘以0.25以得K_up/K_dn（单位 MWh/MW）
        timeslot_max = daily_stats.groupby('time_of_day').agg({
            'agc_up': 'max',
            'agc_dn': 'max'
        }).reset_index()
        timeslot_max['agc_up'] *= 0.25
        timeslot_max['agc_dn'] *= 0.25

        # 确保有完整的时间段
        all_timeslots = pd.DataFrame({'time_of_day': range(T)})
        timeslot_max = pd.merge(all_timeslots, timeslot_max, on='time_of_day', how='left').fillna(0)

        # 提取K值
        K_up_values = timeslot_max['agc_up'].values
        K_dn_values = timeslot_max['agc_dn'].values

        # 生成基础场景（取15分钟平均值，单位仍为 MW/MW）
        base_scenario = agc_signals.groupby(pd.Grouper(key="timeslot", freq=resample_freq))["agc_delta"].agg(
            l_agc_up=pos_mean,
            l_agc_dn=neg_mean
        ).iloc[:T].reset_index(drop=True)

        # 调试专用 用来调整agc数据 - 放大AGC信号使ES发挥backup作用
        # base_scenario["l_agc_up"] = base_scenario["l_agc_up"] * 2000000000.0  # 极端放大20倍
        # base_scenario["l_agc_dn"] = base_scenario["l_agc_dn"] * 1.0  # 极端放大20倍

        # 生成带噪声场景（MW/MW）
        scenarios = []
        for _ in range(self.num_scenarios):
            agc_up_noise = self.rng.normal(1, 1, T)
            agc_dn_noise = self.rng.normal(1, 1, T)
            scenario = pd.DataFrame({
                'agc_up': base_scenario['l_agc_up'] * agc_up_noise.clip(0.1, 0.9),
                'agc_dn': base_scenario['l_agc_dn'] * agc_dn_noise.clip(0.1, 0.9)
            })
            scenarios.append(scenario)

        # 构建 K 值 DataFrame（单位为 MWh/MW）
        capacity_reserves = pd.DataFrame({
            'timeslot': range(T),
            'K_up': K_up_values,
            'K_dn': K_dn_values
        })

        return scenarios, capacity_reserves
    
    def reduce_scenarios(self, *,
                        ev_scenarios: List[pd.DataFrame],
                        price_scenarios: List[pd.DataFrame],
                        agc_scenarios: List[pd.DataFrame]) -> Tuple[List[pd.DataFrame], List[pd.DataFrame], List[pd.DataFrame]]:
        """使用K-means方法缩减场景数量
        
        Args:
            ev_scenarios: EV场景列表
            price_scenarios: 价格场景列表
            agc_scenarios: AGC信号场景列表
            
        Returns:
            Tuple[List[pd.DataFrame], List[Dict[str, pd.Series]], List[pd.DataFrame]]: 缩减后的场景
        """
        # 将场景转换为特征向量
        features = []
        for i in range(self.num_scenarios):
            feature = []
            # EV特征
            ev_scenario = ev_scenarios[i]
            feature.extend(ev_scenario['arrival_time'].values)
            feature.extend(ev_scenario['departure_time'].values)
            feature.extend(ev_scenario['soc_arrival'].values)
            feature.extend(ev_scenario['soc_departure'].values)

            # 价格特征
            price_scenario = price_scenarios[i]
            feature.extend(price_scenario['dam_prices'].values)
            feature.extend(price_scenario['rtm_prices'].values)
            feature.extend(price_scenario['afrr_up_cap_prices'].values)
            feature.extend(price_scenario['afrr_dn_cap_prices'].values)
            feature.extend(price_scenario['balancing_prices'].values)
            feature.extend(price_scenario['mileage_multiplier_up'].values)
            feature.extend(price_scenario['mileage_multiplier_dn'].values)
            
            # AGC特征
            agc_scenario = agc_scenarios[i]
            feature.extend(agc_scenario['agc_up'].values)
            feature.extend(agc_scenario['agc_dn'].values)
            
            features.append(feature)
            
        # 使用K-means聚类
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=self.seed if self.seed is not None else 42)  #这里random_state看作seed
        labels = kmeans.fit_predict(features)  #fit(x)：对特征向量x进行训练 找到num_cluster个簇中心  predict(X)：将每个样本分配到最近的簇，返回每个样本所属的簇编号
        
        # 选择每个簇的中心场景
        reduced_ev_scenarios = []
        reduced_price_scenarios = []
        reduced_agc_scenarios = []
        
        for cluster_id in range(self.num_clusters):
            cluster_indices = np.where(labels == cluster_id)[0]  #这里的[0]代表tuple里第0个数组 也就是lable对应于cluster_id的元素编号
            if len(cluster_indices) > 0:  #循环条件:对应编号内有样本 也就是有某个场景归纳与某个簇中心了
                # 选择距离簇中心最近的场景
                center = kmeans.cluster_centers_[cluster_id] #向量
                distances = [np.linalg.norm(np.array(features[i]) - center) for i in cluster_indices] #feature[i] 第i个场景的特征向量
                closest_idx = cluster_indices[np.argmin(distances)]
                
                reduced_ev_scenarios.append(ev_scenarios[closest_idx])
                reduced_price_scenarios.append(price_scenarios[closest_idx])
                reduced_agc_scenarios.append(agc_scenarios[closest_idx])
                
        return reduced_ev_scenarios, reduced_price_scenarios, reduced_agc_scenarios 
    


class ScenarioGenerator1:

    def __init__(self, *, 
                num_scenarios: int,
                num_clusters: int,
                seed: int = None):
        """初始化场景生成器
           这个与ScenarioGenerator不同，这个是用来构造合适的agc信号的
        Args:
            num_scenarios: 初始场景数量
            num_clusters: K-means聚类后的场景数量
            seed: 随机种子
        """
        self.num_scenarios = num_scenarios
        self.num_clusters = num_clusters
        self.seed = seed
        # 使用局部随机数生成器，避免污染全局随机状态
        self.rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    
    def generate_base_ev_profiles(self, *,
                                  num_evs: int,
                                  discount: float,
                                  charging_price: float,
                                  use_timeslot: bool = True) -> pd.DataFrame:
        """生成基准EV数据（使用本生成器的随机源），确保与场景生成共享同一seed。

        Args:
            num_evs: 总EV数量
            discount: 充电折扣比例
            charging_price: 基础充电价格（€/MWh）
            use_timeslot: 是否将小时时间转换为15分钟时间槽索引

        Returns:
            pd.DataFrame: EV参数表
        """
        rng = self.rng

        # 计算可控EV比例
        theta = discount
        rho = max(0.0, min(1.0, 4 * theta - 0.2))
        num_controllable = int(rho * num_evs)

        # 白天/夜间数量
        num_day = num_evs // 2
        num_night = num_evs - num_day

        # 分配可控数量
        num_cc_day = int(min(rho * num_day, num_controllable))
        num_cc_night = num_controllable - num_cc_day

        df = pd.DataFrame()
        df['ev_id'] = range(num_evs)

        # 充电类型
        df['charging_type'] = np.concatenate([['day'] * num_day, ['night'] * num_night])

        # 控制类型并打乱
        day_types = ['cc'] * num_cc_day + ['uc'] * (num_day - num_cc_day)
        night_types = ['cc'] * num_cc_night + ['uc'] * (num_night - num_cc_night)
        rng.shuffle(day_types)
        rng.shuffle(night_types)
        df['ev_type'] = np.concatenate([day_types, night_types])

        # 初始化参数
        df['arrival_time'] = 0.0
        df['departure_time'] = 0.0
        df['soc_arrival'] = 0.0
        df['soc_departure'] = 0.0
        df['soc_max'] = 0.95
        df['soc_min'] = 0.1
        df['battery_capacity'] = 0.028  # MWh（28 kWh）
        df['max_charge_power'] = 0.0066 # MW（6.6 kW）
        df['efficiency'] = 0.95

        # 掩码
        day_mask = df['charging_type'] == 'day'
        night_mask = df['charging_type'] == 'night'
        uc_mask = df['ev_type'] == 'uc'
        cc_mask = df['ev_type'] == 'cc'

        # UC：单独设置晚上到达（18:00-20:00），标记为 day，避免跨天逻辑
        num_uc = int(uc_mask.sum())
        if num_uc > 0:
            uc_arrival = rng.normal(18.0, 2.06, num_uc).clip(16.0, 20.0)
            df.loc[uc_mask, 'arrival_time'] = uc_arrival
            df.loc[uc_mask, 'charging_type'] = 'day'
            df.loc[uc_mask, 'soc_arrival'] = rng.uniform(0.2, 0.35, num_uc)
            df.loc[uc_mask, 'soc_departure'] = rng.uniform(0.85, 0.95, num_uc)

        # CC 且 day：白天参数（小时）
        day_cc_mask = day_mask & cc_mask
        num_day_cc = int(day_cc_mask.sum())
        if num_day_cc > 0:
            df.loc[day_cc_mask, 'arrival_time'] = rng.normal(8.82, 1.08, num_day_cc).clip(7.5, 10.5)
            df.loc[day_cc_mask, 'departure_time'] = rng.normal(18.55, 2.06, num_day_cc).clip(16.0, 21.0)
            df.loc[day_cc_mask, 'soc_arrival'] = rng.uniform(0.2, 0.35, num_day_cc)
            df.loc[day_cc_mask, 'soc_departure'] = rng.uniform(0.85, 0.95, num_day_cc)

        # CC 且 night：夜间参数（小时）
        night_cc_mask = night_mask & cc_mask
        num_night_cc = int(night_cc_mask.sum())
        if num_night_cc > 0:
            df.loc[night_cc_mask, 'arrival_time'] = rng.normal(22.91, 3.2, num_night_cc).clip(20.9, 24.0)
            df.loc[night_cc_mask, 'departure_time'] = rng.normal(8.95, 1.4, num_night_cc).clip(7.0, 10.0)
            df.loc[night_cc_mask, 'soc_arrival'] = rng.uniform(0.2, 0.35, num_night_cc)
            df.loc[night_cc_mask, 'soc_departure'] = rng.uniform(0.85, 0.95, num_night_cc)

        # 充电价格（可控享受折扣）
        df['charging_price'] = np.where(
            df['ev_type'] == 'cc',
            charging_price * (1 - discount),
            charging_price
        )

        # 转换为15分钟时间槽
        if use_timeslot:
            df['hour_arrival'] = df['arrival_time'].copy()
            df['hour_departure'] = df['departure_time'].copy()

            df['arrival_time'] = (df['arrival_time'] * 4).astype(int)
            # 夜间（仅 CC 夜间）跨天
            df.loc[night_cc_mask, 'departure_time'] = (df.loc[night_cc_mask, 'departure_time'] * 4).astype(int) + 96
            # 其他（含 UC 和 CC 白天）不跨天
            non_overnight_mask = ~night_cc_mask
            df.loc[non_overnight_mask, 'departure_time'] = (df.loc[non_overnight_mask, 'departure_time'] * 4).astype(int)

            df['arrival_time'] = df['arrival_time'].clip(0, 95)
            # 非跨天（含 UC）
            df.loc[non_overnight_mask, 'departure_time'] = df.loc[non_overnight_mask, 'departure_time'].clip(0, 95)
            # 跨天（仅 CC 夜间）
            df.loc[night_cc_mask, 'departure_time'] = df.loc[night_cc_mask, 'departure_time'].clip(96, 191)

        return df
            
    def generate_ev_scenarios(self, *, ev_profiles: pd.DataFrame) -> List[pd.DataFrame]:
        """生成EV场景，每个场景都独立采样，符合原始分布
        
        Args:
            ev_profiles: 基准EV数据，包含充电类型（day/night）
            
        Returns:
            List[pd.DataFrame]: EV场景列表
        """
        scenarios = []
        num_evs = len(ev_profiles)
        
        # 检查输入数据是使用时间槽索引还是连续小时
        use_timeslot = 'hour_arrival' in ev_profiles.columns
        
        for _ in range(self.num_scenarios):
            scenario = pd.DataFrame()
            scenario['ev_id'] = range(num_evs)
            scenario['ev_type'] = ev_profiles['ev_type'].values
            scenario['charging_type'] = ev_profiles['charging_type'].values
            
            # 分别处理白天和夜间充电的EV
            day_mask = scenario['charging_type'] == 'day'
            night_mask = scenario['charging_type'] == 'night'
            uc_mask = ev_profiles['ev_type'].values == 'uc'
            
            num_day_evs = sum(day_mask)
            num_night_evs = sum(night_mask)
            
            # 初始化所有EV的参数
            scenario['arrival_time'] = 0.0
            scenario['departure_time'] = 0.0
            scenario['soc_arrival'] = 0.0
            scenario['soc_departure'] = 0.0
            scenario['soc_max'] = ev_profiles['soc_max'].values
            scenario['soc_min'] = ev_profiles['soc_min'].values
            scenario['battery_capacity'] = ev_profiles['battery_capacity'].values  # 已转换为MWh
            scenario['max_charge_power'] = ev_profiles['max_charge_power'].values  # 已转换为MW
            scenario['efficiency'] = ev_profiles['efficiency'].values
            scenario['charging_price'] = ev_profiles['charging_price'].values
            
            # 为白天充电的EV生成场景参数（连续小时时间）
            scenario.loc[day_mask, 'arrival_time'] = self.rng.normal(8.82, 1.08, num_day_evs).clip(7.5, 10.5)
            scenario.loc[day_mask, 'departure_time'] = self.rng.normal(18.55, 2.06, num_day_evs).clip(16.0, 21.0)
            scenario.loc[day_mask, 'soc_arrival'] = self.rng.uniform(0.2, 0.35, num_day_evs).clip(0.2, 0.35)
            scenario.loc[day_mask, 'soc_departure'] = self.rng.uniform(0.85, 0.95, num_day_evs).clip(0.85, 0.95)
            
            # 为夜间充电的EV生成场景参数（连续小时时间）
            scenario.loc[night_mask, 'arrival_time'] = self.rng.normal(22.91, 3.2, num_night_evs).clip(20.9, 24.0)
            scenario.loc[night_mask, 'departure_time'] = self.rng.normal(8.95, 1.4, num_night_evs).clip(7.0, 10.0)
            scenario.loc[night_mask, 'soc_arrival'] = self.rng.uniform(0.2, 0.35, num_night_evs).clip(0.2, 0.35)
            scenario.loc[night_mask, 'soc_departure'] = self.rng.uniform(0.85, 0.95, num_night_evs).clip(0.85, 0.95)

            # 选择“最后一辆 日间充电 的 cc 车”作为特殊车辆（18:00 到达，23:00 离开），不改变任何计数
            day_cc_mask_current = (scenario['ev_type'] == 'cc') & (scenario['charging_type'] == 'day')
            if day_cc_mask_current.any():
                special_idx = scenario.index[day_cc_mask_current].max()
            else:
                # 极端兜底：若不存在日间 cc，则不改 charging_type，仅选择最后一辆 cc
                cc_mask_current = scenario['ev_type'] == 'cc'
                if cc_mask_current.any():
                    special_idx = scenario.index[cc_mask_current].max()
                else:
                    special_idx = scenario.index.max()  # 兜底
            scenario.loc[special_idx, 'arrival_time'] = 16.0  # 小时
            scenario.loc[special_idx, 'departure_time'] = 23.0  # 小时（不跨天）
            scenario.loc[special_idx, 'soc_arrival'] = 0.6
            scenario.loc[special_idx, 'soc_departure'] = 0.95
            scenario.loc[special_idx, 'soc_max'] = 0.95
            scenario.loc[special_idx, 'soc_min'] = 0.1

            # 更新掩码
            day_mask = scenario['charging_type'] == 'day'
            night_mask = scenario['charging_type'] == 'night'

            # 覆盖 UC 到达时间：正态分布并截断到傍晚 16:00-20:00
            # 追加特殊车辆后，基于当前 scenario 重新计算 UC 掩码
            uc_mask = scenario['ev_type'] == 'uc'
            num_uc = int(uc_mask.sum())
            if num_uc > 0:
                uc_arrival = self.rng.normal(18.0, 2.06, num_uc).clip(16.0, 20.0)
                uc_departure = self.rng.normal(23.0, 2.06, num_uc).clip(23.0, 24.0)
                uc_soc_arrival = self.rng.uniform(0.3, 0.35, num_uc).clip(0.2, 0.95)
                uc_soc_departure = self.rng.uniform(0.85, 0.95, num_uc).clip(0.85, 0.95)
                scenario.loc[uc_mask, 'arrival_time'] = uc_arrival
                scenario.loc[uc_mask, 'departure_time'] = uc_departure
                scenario.loc[uc_mask, 'soc_arrival'] = uc_soc_arrival
                scenario.loc[uc_mask, 'soc_departure'] = uc_soc_departure

            # 如果输入数据使用了时间槽索引，则将连续小时时间转换为15分钟时间槽索引 (0-95)
            if use_timeslot:
                # 创建源时间列的副本
                scenario['hour_arrival'] = scenario['arrival_time'].copy()
                scenario['hour_departure'] = scenario['departure_time'].copy()
                
                # 转换到达时间：小时 → 时间槽索引
                scenario['arrival_time'] = (scenario['arrival_time'] * 4).astype(int)
                
                # 特殊处理夜间充电的离开时间：
                # 对于夜间充电，离开时间小于到达时间，表示跨天，需要加上96来表示第二天
                scenario.loc[night_mask, 'departure_time'] = (scenario.loc[night_mask, 'departure_time'] * 4).astype(int) + 96
                
                # 对于白天充电，直接转换
                scenario.loc[day_mask, 'departure_time'] = (scenario.loc[day_mask, 'departure_time'] * 4).astype(int)
                
                # 到达时间限制在正确范围内
                scenario['arrival_time'] = scenario['arrival_time'].clip(0, 95)
                
                # 仅对白天充电的EV限制departure_time范围
                scenario.loc[day_mask, 'departure_time'] = scenario.loc[day_mask, 'departure_time'].clip(0, 95)
                # 对于跨天的夜间充电EV，保留其跨天信息，确保departure_time > 95
                scenario.loc[night_mask, 'departure_time'] = scenario.loc[night_mask, 'departure_time'].clip(96, 191)

                # 覆盖特殊车辆的时间槽（到达 64，离开 92，不跨天）
                scenario.loc[special_idx, 'arrival_time'] = 16 * 4
                scenario.loc[special_idx, 'departure_time'] = 23 * 4

            # 重新编号 ev_id，确保连续且唯一
            scenario['ev_id'] = range(len(scenario))
            scenarios.append(scenario)
        return scenarios
    
    def generate_price_scenarios(self, *,
                               dam_prices: pd.DataFrame,
                               rtm_prices: pd.DataFrame,
                               capacity_price: pd.DataFrame,
                               balancing_prices: pd.DataFrame,
                               mileage_multiplier: pd.DataFrame,
                               T: int) -> List[pd.DataFrame]:
        """生成价格场景，每个场景为DataFrame
        Args:
            dam_prices: 日前市场价格
            rtm_prices: 实时市场价格
            capacity_price: 容量投标价格
            balancing_prices: 激活价格
            mileage_multiplier: 里程乘数
            T: 时间步数
        Returns:
            List[pd.DataFrame]: 价格场景列表
        """
        scenarios = []
        for _ in range(self.num_scenarios):
            dam_noise = self.rng.normal(1, 0.1, T)
            rtm_noise = self.rng.normal(1, 0.15, T)
            afrr_up_cap_noise = self.rng.normal(1, 0.2, T)
            afrr_dn_cap_noise = self.rng.normal(1, 0.2, T)
            balancing_noise = self.rng.normal(1, 0.1, T)
            # 乘数不添加随机性，保持原值
            mileage_multiplier_up = mileage_multiplier['Mileage_Multiplier_Up'].values
            mileage_multiplier_dn = mileage_multiplier['Mileage_Multiplier_Dn'].values

            scenario = pd.DataFrame({
                'dam_prices': dam_prices['price'].values * dam_noise,
                'rtm_prices': rtm_prices['price'].values * rtm_noise,
                'afrr_up_cap_prices': capacity_price['afrr_up_cap_price'].values* afrr_up_cap_noise,
                'afrr_dn_cap_prices': capacity_price['afrr_dn_cap_price'].values* afrr_dn_cap_noise,
                'balancing_prices': balancing_prices['price'].values* balancing_noise,
                'mileage_multiplier_up': mileage_multiplier_up,
                'mileage_multiplier_dn': mileage_multiplier_dn
            })

            scenarios.append(scenario)
        return scenarios
    
    def generate_agc_scenarios(self, *,
                             agc_signals: pd.DataFrame,
                             K_values: pd.DataFrame,
                              T: int) -> Tuple[List[pd.DataFrame], pd.DataFrame]:
        """
        这里生成自己构造的AGC信号场景，并为每个时间段计算单独的K_up和K_dn容量预留值

        Args:
            agc_signals: 原始AGC信号数据（含timeslot, agc_delta）
            T: 时间步数
            resample_freq: 重采样频率，默认15分钟

        Returns:
            List[pd.DataFrame]: AGC信号场景列表（基于输入，长度对齐T）
            pd.DataFrame: 每个时间段的K_up/K_dn容量（乘以0.25后的MWh/MW）
        """
        df = agc_signals.copy()
        base_up = pd.to_numeric(df['agc_up'], errors='coerce').fillna(0).values
        base_dn = pd.to_numeric(df['agc_dn'], errors='coerce').fillna(0).values

        # 对齐长度到T
        if len(base_up) < T:
            base_up = np.pad(base_up, (0, T - len(base_up)), mode='constant', constant_values=0.0)
        else:
            base_up = base_up[:T]
        if len(base_dn) < T:
            base_dn = np.pad(base_dn, (0, T - len(base_dn)), mode='constant', constant_values=0.0)
        else:
            base_dn = base_dn[:T]

        scenarios = []
        for _ in range(self.num_scenarios):
            agc_up_noise = self.rng.normal(1, 0.1, T)
            agc_dn_noise = self.rng.normal(1, 0.1, T)
            scenario = pd.DataFrame({
                'agc_up': base_up * agc_up_noise,
                'agc_dn': base_dn * agc_dn_noise
            })
            scenarios.append(scenario)
        
        df1 = K_values.copy()
        K_up = pd.to_numeric(df1['K_up'], errors='coerce').fillna(0).values
        K_dn = pd.to_numeric(df1['K_dn'], errors='coerce').fillna(0).values

        capacity_reserves = pd.DataFrame({
            'timeslot': range(T),
            'K_up': K_up,
            'K_dn': K_dn
        })

        return scenarios, capacity_reserves



    
    def reduce_scenarios(self, *,
                        ev_scenarios: List[pd.DataFrame],
                        price_scenarios: List[pd.DataFrame],
                        agc_scenarios: List[pd.DataFrame]) -> Tuple[List[pd.DataFrame], List[pd.DataFrame], List[pd.DataFrame]]:
        """使用K-means方法缩减场景数量
        
        Args:
            ev_scenarios: EV场景列表
            price_scenarios: 价格场景列表
            agc_scenarios: AGC信号场景列表
            
        Returns:
            Tuple[List[pd.DataFrame], List[Dict[str, pd.Series]], List[pd.DataFrame]]: 缩减后的场景
        """
        # 将场景转换为特征向量
        features = []
        for i in range(self.num_scenarios):
            feature = []
            # EV特征
            ev_scenario = ev_scenarios[i]
            feature.extend(ev_scenario['arrival_time'].values)
            feature.extend(ev_scenario['departure_time'].values)
            feature.extend(ev_scenario['soc_arrival'].values)
            feature.extend(ev_scenario['soc_departure'].values)

            # 价格特征
            price_scenario = price_scenarios[i]
            feature.extend(price_scenario['dam_prices'].values)
            feature.extend(price_scenario['rtm_prices'].values)
            feature.extend(price_scenario['afrr_up_cap_prices'].values)
            feature.extend(price_scenario['afrr_dn_cap_prices'].values)
            feature.extend(price_scenario['balancing_prices'].values)
            feature.extend(price_scenario['mileage_multiplier_up'].values)
            feature.extend(price_scenario['mileage_multiplier_dn'].values)
            
            # AGC特征
            agc_scenario = agc_scenarios[i]
            feature.extend(agc_scenario['agc_up'].values)
            feature.extend(agc_scenario['agc_dn'].values)
            
            features.append(feature)
            
        # 使用K-means聚类
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=self.seed if self.seed is not None else 42)  #这里random_state看作seed
        labels = kmeans.fit_predict(features)  #fit(x)：对特征向量x进行训练 找到num_cluster个簇中心  predict(X)：将每个样本分配到最近的簇，返回每个样本所属的簇编号
        
        # 选择每个簇的中心场景
        reduced_ev_scenarios = []
        reduced_price_scenarios = []
        reduced_agc_scenarios = []
        
        for cluster_id in range(self.num_clusters):
            cluster_indices = np.where(labels == cluster_id)[0]  #这里的[0]代表tuple里第0个数组 也就是lable对应于cluster_id的元素编号
            if len(cluster_indices) > 0:  #循环条件:对应编号内有样本 也就是有某个场景归纳与某个簇中心了
                # 选择距离簇中心最近的场景
                center = kmeans.cluster_centers_[cluster_id] #向量
                distances = [np.linalg.norm(np.array(features[i]) - center) for i in cluster_indices] #feature[i] 第i个场景的特征向量
                closest_idx = cluster_indices[np.argmin(distances)]
                
                reduced_ev_scenarios.append(ev_scenarios[closest_idx])
                reduced_price_scenarios.append(price_scenarios[closest_idx])
                reduced_agc_scenarios.append(agc_scenarios[closest_idx])
                
        return reduced_ev_scenarios, reduced_price_scenarios, reduced_agc_scenarios 