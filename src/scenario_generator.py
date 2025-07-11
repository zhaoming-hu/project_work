import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from typing import Dict, List, Tuple

class ScenarioGenerator:
    def __init__(self, *, 
                 num_scenarios: int = 1000,
                 num_clusters: int = 48,
                 seed: int = None):
        """初始化场景生成器
        
        Args:
            num_scenarios: 初始场景数量
            num_clusters: K-means聚类后的场景数量
            seed: 随机种子
        """
        self.num_scenarios = num_scenarios
        self.num_clusters = num_clusters
        if seed is not None:
            np.random.seed(seed)
            
    def generate_ev_scenarios(self, *, ev_profiles: pd.DataFrame) -> List[pd.DataFrame]:
        """生成EV场景，每个场景都独立采样，符合原始分布"""
        scenarios = []
        num_evs = len(ev_profiles)
        for _ in range(self.num_scenarios):
            scenario = pd.DataFrame()
            scenario['ev_id'] = range(num_evs)
            scenario['ev_type'] = ev_profiles['ev_type'].values

            # 按原始分布采样
            scenario['arrival_time'] = np.random.normal(ev_profiles['arrival_time'].mean(), ev_profiles['arrival_time'].std(), num_evs).clip(7.5, 10.5)   #np.random.normal(均值, 标准差, 数量)
            scenario['departure_time'] = np.random.normal(ev_profiles['departure_time'].mean(), ev_profiles['departure_time'].std(), num_evs).clip(16.0, 21.0)
            scenario['soc_arrival'] = np.random.uniform(ev_profiles['soc_arrival'].min(), ev_profiles['soc_arrival'].max(), num_evs)  #np.random.uniform(最小值, 最大值, 数量)
            scenario['soc_departure'] = np.random.uniform(ev_profiles['soc_departure'].min(), ev_profiles['soc_departure'].max(), num_evs)
            scenario['soc_max'] = ev_profiles['soc_max'].values
            scenario['soc_min'] = ev_profiles['soc_min'].values
            scenario['battery_capacity'] = ev_profiles['battery_capacity'].values
            scenario['max_charge_power'] = ev_profiles['max_charge_power'].values
            scenario['efficiency'] = ev_profiles['efficiency'].values
            scenario['charging_price'] = ev_profiles['charging_price'].values

            scenarios.append(scenario)
        return scenarios
    
    def generate_price_scenarios(self, *,
                               dam_prices: pd.DataFrame,
                               rtm_prices: pd.DataFrame,
                               capacity_price: pd.DataFrame,
                               balancing_prices: pd.DataFrame,
                               T: int) -> List[pd.DataFrame]:
        """生成价格场景，每个场景为DataFrame
        Args:
            dam_prices: 日前市场价格
            rtm_prices: 实时市场价格
            capacity_price: 容量投标价格
            balancing_prices: 激活价格
            T: 时间步数
        Returns:
            List[pd.DataFrame]: 价格场景列表
        """
        scenarios = []
        for _ in range(self.num_scenarios):
            dam_noise = np.random.normal(1, 0.1, T)
            rtm_noise = np.random.normal(1, 0.15, T)
            afrr_up_cap_noise = np.random.normal(1, 0.2, T)
            afrr_dn_cap_noise = np.random.normal(1, 0.2, T)
            balancing_noise = np.random.normal(1, 0.1, T)

            scenario = pd.DataFrame({
                'dam_prices': dam_prices['price'].values * dam_noise,
                'rtm_prices': rtm_prices['price'].values * rtm_noise,
                'afrr_up_cap_prices': capacity_price['afrr_up_cap_price'].values * afrr_up_cap_noise,
                'afrr_dn_cap_prices': capacity_price['afrr_dn_cap_price'].values * afrr_dn_cap_noise,
                'balancing_prices': balancing_prices['price'].values * balancing_noise
            })

            scenarios.append(scenario)
        return scenarios
    
    def generate_agc_scenarios(self, *,
                             agc_signals: pd.DataFrame,
                             T: int,
                             resample_freq: str = "15min") -> Tuple[List[pd.DataFrame], float, float]:
        """
        生成AGC信号场景，先分组采样，再生成带噪声场景，并返回K_up和K_dn
        Args:
            agc_signals: 原始AGC信号数据（含timeslot, agc_delta）
            T: 时间步数
            resample_freq: 重采样频率，默认15分钟
        Returns:
            List[pd.DataFrame]: AGC信号场景列表
            float: K_up
            float: K_dn
        """
        # 先分组采样，分别对正负分开求均值
        def pos_mean(x):
            return x[x > 0].mean() if (x > 0).any() else 0.0
        def neg_mean(x):
            return -x[x < 0].mean() if (x < 0).any() else 0.0

        grouped = agc_signals.groupby(pd.Grouper(key="timeslot", freq=resample_freq))["agc_delta"].agg(
            l_agc_up=pos_mean,
            l_agc_dn=neg_mean
        )
        grouped = grouped.iloc[:T].reset_index(drop=True)
        grouped["l_agc_up"] = grouped["l_agc_up"] * 0.25
        grouped["l_agc_dn"] = grouped["l_agc_dn"] * 0.25  #按照理解 agc数据在每秒都应该是确定的要么正要么负 这里之所以两个是因为15min内正负都有

        # 计算K_up和K_dn
        K_up = grouped["l_agc_up"].max()
        K_dn = grouped["l_agc_dn"].max()

        # 生成带噪声的场景
        scenarios = []
        for _ in range(self.num_scenarios):
            agc_up_noise = np.random.normal(1, 0.1, T)
            agc_dn_noise = np.random.normal(1, 0.1, T)
            scenario = pd.DataFrame({
                'agc_up': grouped['l_agc_up'] * agc_up_noise,
                'agc_dn': grouped['l_agc_dn'] * agc_dn_noise
            })
            scenarios.append(scenario)
        return scenarios, K_up, K_dn
    
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
            
            # AGC特征
            agc_scenario = agc_scenarios[i]
            feature.extend(agc_scenario['agc_up'].values)
            feature.extend(agc_scenario['agc_dn'].values)
            
            features.append(feature)
            
        # 使用K-means聚类
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=42)  #这里random_state看作seed
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