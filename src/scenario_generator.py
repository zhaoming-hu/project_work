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
            
    def generate_ev_scenarios(self, *, 
                            ev_profiles: pd.DataFrame,
                            T: int) -> List[pd.DataFrame]:
        """生成EV场景
        
        Args:
            ev_profiles: EV参数表
            T: 时间步数
            
        Returns:
            List[pd.DataFrame]: 场景列表
        """
        scenarios = []
        for _ in range(self.num_scenarios):
            # 为每个EV生成随机到达和离开时间
            scenario = ev_profiles.copy()
            scenario['arrival_time'] = np.random.normal(
                ev_profiles['arrival_time'].mean(),
                ev_profiles['arrival_time'].std(),
                len(ev_profiles)
            ).clip(7.5, 10.5)
            
            scenario['departure_time'] = np.random.normal(
                ev_profiles['departure_time'].mean(),
                ev_profiles['departure_time'].std(),
                len(ev_profiles)
            ).clip(16.0, 21.0)   #从load_ev_profile出发 每生成一个场景 其中就包含400辆EV的状态
            
            # 生成随机初始SOC
            scenario['soc_initial'] = np.random.uniform(0.2, 0.35, len(ev_profiles))
            
            scenarios.append(scenario)
            
        return scenarios
    
    def generate_price_scenarios(self, *,
                               dam_prices: pd.Series,  #这个数据类型是说 一组一维带可重复的可哈希类型的索引的series（这里索引对应时间 series对应prices）
                               rtm_prices: pd.Series,
                               afrr_up_prices: pd.Series,
                               afrr_dn_prices: pd.Series,
                               balancing_prices: pd.Series,
                               T: int) -> List[Dict[str, pd.Series]]:
        """生成价格场景
        
        Args:
            dam_prices: 日前市场价格
            rtm_prices: 实时市场价格
            afrr_up_prices: 上调容量价格
            afrr_dn_prices: 下调容量价格
            balancing_prices: 平衡价格
            T: 时间步数
            
        Returns:
            List[Dict[str, pd.Series]]: 价格场景列表
        """
        scenarios = []
        for _ in range(self.num_scenarios):
            # 生成随机价格波动
            dam_noise = np.random.normal(1, 0.1, T)
            rtm_noise = np.random.normal(1, 0.15, T)
            afrr_up_noise = np.random.normal(1, 0.2, T)
            afrr_dn_noise = np.random.normal(1, 0.2, T)
            balancing_noise = np.random.normal(1, 0.1, T)
            
            scenario = {
                'dam_prices': dam_prices * dam_noise,
                'rtm_prices': rtm_prices * rtm_noise,
                'afrr_up_prices': afrr_up_prices * afrr_up_noise,
                'afrr_dn_prices': afrr_dn_prices * afrr_dn_noise,
                'balancing_prices': balancing_prices * balancing_noise
            }
            scenarios.append(scenario)
            
        return scenarios
    
    def generate_agc_scenarios(self, *,
                             agc_signals: pd.DataFrame,
                             T: int) -> List[pd.DataFrame]:
        """生成AGC信号场景
        
        Args:
            agc_signals: AGC信号数据
            T: 时间步数
            
        Returns:
            List[pd.DataFrame]: AGC信号场景列表
        """
        scenarios = []
        for _ in range(self.num_scenarios):
            # 生成随机AGC信号波动
            agc_up_noise = np.random.normal(1, 0.1, T)
            agc_dn_noise = np.random.normal(1, 0.1, T)
            
            scenario = agc_signals.copy()
            scenario['agc_up'] = agc_signals['agc_up'] * agc_up_noise
            scenario['agc_dn'] = agc_signals['agc_dn'] * agc_dn_noise
            
            scenarios.append(scenario)
            
        return scenarios
    
    def reduce_scenarios(self, *,
                        ev_scenarios: List[pd.DataFrame],
                        price_scenarios: List[Dict[str, pd.Series]],
                        agc_scenarios: List[pd.DataFrame]) -> Tuple[List[pd.DataFrame], List[Dict[str, pd.Series]], List[pd.DataFrame]]:
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
            feature.extend(ev_scenario['soc_initial'].values)
            
            # 价格特征
            price_scenario = price_scenarios[i]
            feature.extend(price_scenario['dam_prices'].values)
            feature.extend(price_scenario['rtm_prices'].values)
            feature.extend(price_scenario['afrr_up_prices'].values)
            feature.extend(price_scenario['afrr_dn_prices'].values)
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