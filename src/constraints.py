import gurobipy as gp
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

class V2GConstraints:
    def __init__(self, *, model: gp.Model, params: Dict):
        """初始化约束管理器
        
        Args:
            model: Gurobi优化模型
            params: 包含所有必要参数的字典  写主程序的时候需要定义一下
        """
        self.model = model
        self.params = params
        
    def add_uc_ev_constraints(self, *, ev_profiles: pd.DataFrame = None, T: int = None, delta_t: float = 0.25) -> None:
        """添加uncontrollable EV的充电功率约束（直接赋值，不作为优化变量）"""
        ev_profiles = ev_profiles or self.params["ev_profiles"]
        T = T or self.params["T"]
        P_ev_uc = self.params["P_ev_uc"]   #这里在写主程序的时候要定义一下
        for i, row in ev_profiles[ev_profiles["ev_type"] == "uc"].iterrows():    #遍历所有uc类型的ev
            Ta_i = int(row["arrival_time"] / delta_t)  #把时间转化到具体的时间步  这样可以和timeslot对应
            Sa_i = row["soc_initial"]
            Sd_i = row["soc_target"]
            Eev_i = row["battery_capacity"]
            Pmax_i = row["max_charge_power"]
            eta_ev = row["efficiency"]
            Tk_i = int(Ta_i + (Sd_i - Sa_i) * Eev_i / (Pmax_i * eta_ev * delta_t))  #计算驶离时间
            for t in range(T):
                if Ta_i <= t <= Tk_i:
                    self.model.addConstr(P_ev_uc[i, t] == Pmax_i)  #addConstr是Gurobi的约束添加函数
                else:
                    self.model.addConstr(P_ev_uc[i, t] == 0)

    def add_cc_ev_constraints(self, *, ev_profiles: pd.DataFrame = None, T: int = None, delta_t: float = 0.25) -> None:
        """添加controllable EV的充电功率和SOC约束"""
        ev_profiles = ev_profiles or self.params["ev_profiles"]
        T = T or self.params["T"]
        P_ev_cc = self.params["P_ev_cc"]  # 这里记得在主程序定义一下
        soc = self.params["soc"]           # 这里需要在主程序定义
        for i, row in ev_profiles[ev_profiles["ev_type"] == "cc"].iterrows():
            Ta_i = int(row["arrival_time"] / delta_t)
            Td_i = int(row["departure_time"] / delta_t)
            Sa_i = row["soc_initial"]
            Sd_i = row["soc_target"]
            Smax_i = row["soc_max"]
            Smin_i = row["soc_min"]
            Eev_i = row["battery_capacity"]
            Pmax_i = row["max_charge_power"]
            eta_ev = row["efficiency"]
            # 充电功率约束
            for t in range(T):
                if Ta_i <= t <= Td_i:
                    self.model.addConstr(P_ev_cc[i, t] >= 0)
                    self.model.addConstr(P_ev_cc[i, t] <= Pmax_i)
                else:
                    self.model.addConstr(P_ev_cc[i, t] == 0)
            # SOC动态约束
            for t in range(Ta_i, Td_i+1):
                self.model.addConstr(
                    soc[i, t] == Sa_i + gp.quicksum(P_ev_cc[i, tau] * eta_ev * delta_t / Eev_i for tau in range(Ta_i, t+1))
                )
            # 离开时SOC约束
            self.model.addConstr(soc[i, Td_i] >= Sd_i)
            self.model.addConstr(soc[i, Td_i] <= Smax_i)
            self.model.addConstr(soc[i, Td_i] >= Smin_i)

    def add_es_constraints(self, *,
                         T: int = None,
                         P_es: Dict = None,
                         P_es1: Dict = None,
                         P_es2: Dict = None,
                         R_es_up: Dict = None,
                         R_es_dn: Dict = None,
                         u_es: Dict = None,
                         E_es: Dict = None,
                         P_es_max: float = None,
                         E_es_max: float = None,
                         eta_es: float = None) -> None:
        """添加储能系统相关约束
        
        Args:
            T: 时间步数
            P_es: ES总功率变量
            P_es1: ES1功率变量（调频）
            P_es2: ES2功率变量（补偿）
            R_es_up: ES上调容量变量
            R_es_dn: ES下调容量变量
            u_es: ES充电状态变量
            E_es: ES能量状态变量
            P_es_max: 最大功率
            E_es_max: 最大能量
            eta_es: 充放电效率
        """
        T = T or self.params["T"]
        P_es = P_es or self.params["P_es"]
        P_es1 = P_es1 or self.params["P_es1"]
        P_es2 = P_es2 or self.params["P_es2"]
        R_es_up = R_es_up or self.params["R_es_up"]
        R_es_dn = R_es_dn or self.params["R_es_dn"]
        u_es = u_es or self.params["u_es"]
        E_es = E_es or self.params["E_es"]
        P_es_max = P_es_max or self.params["P_es_max"]
        E_es_max = E_es_max or self.params["E_es_max"]
        eta_es = eta_es or self.params["eta_es"]
        
        # ES容量分配约束
        for t in range(T):
            # 总功率等于调频功率加补偿功率
            self.model.addConstr(P_es[t] == P_es1[t] + P_es2[t])
            
            # 调频功率约束
            self.model.addConstr(P_es1[t] <= R_es_up[t])
            self.model.addConstr(P_es1[t] >= -R_es_dn[t])
            
            # 充放电逻辑约束
            self.model.addConstr(P_es[t] <= P_es_max * u_es[t])
            self.model.addConstr(P_es[t] >= -P_es_max * (1 - u_es[t]))
            
            # 能量约束
            if t > 0:
                self.model.addConstr(
                    E_es[t] == E_es[t-1] + 
                    (P_es[t] * eta_es if P_es[t] >= 0 else P_es[t] / eta_es)
                )
            
            # 能量上下限约束
            self.model.addConstr(E_es[t] >= 0)
            self.model.addConstr(E_es[t] <= E_es_max)
    
    def add_market_constraints(self, *,
                             T: int = None,
                             P_ev0: Dict = None,
                             P_es0: Dict = None,
                             R_ev_up: Dict = None,
                             R_ev_dn: Dict = None,
                             R_es_up: Dict = None,
                             R_es_dn: Dict = None,
                             P_ev: Dict = None,
                             P_es: Dict = None,
                             N: int = None) -> None:
        """添加市场相关约束
        
        Args:
            T: 时间步数
            P_ev0: EV能量投标变量
            P_es0: ES能量投标变量
            R_ev_up: EV上调容量变量
            R_ev_dn: EV下调容量变量
            R_es_up: ES上调容量变量
            R_es_dn: ES下调容量变量
            P_ev: EV实际功率变量
            P_es: ES实际功率变量
            N: 电动汽车数量
        """
        T = T or self.params["T"]
        P_ev0 = P_ev0 or self.params["P_ev0"]
        P_es0 = P_es0 or self.params["P_es0"]
        R_ev_up = R_ev_up or self.params["R_ev_up"]
        R_ev_dn = R_ev_dn or self.params["R_ev_dn"]
        R_es_up = R_es_up or self.params["R_es_up"]
        R_es_dn = R_es_dn or self.params["R_es_dn"]
        P_ev = P_ev or self.params["P_ev"]
        P_es = P_es or self.params["P_es"]
        N = N or self.params["N"]
        
        # 日前市场投标约束
        for t in range(T):
            # EV能量投标约束
            self.model.addConstr(
                P_ev0[t] == sum(P_ev[t,n] for n in range(N))
            )
            
            # 容量投标约束
            self.model.addConstr(R_ev_up[t] >= 0)
            self.model.addConstr(R_ev_dn[t] >= 0)
            self.model.addConstr(R_es_up[t] >= 0)
            self.model.addConstr(R_es_dn[t] >= 0)
    
    def add_cvar_constraints(self, *,
                           T: int = None,
                           eta: gp.Var = None,
                           s: Dict = None,
                           beta: float = None,
                           revenue_scenarios: List[float] = None) -> None:
        """添加CVaR相关约束
        
        Args:
            T: 时间步数
            eta: VaR变量
            s: 辅助变量
            beta: 置信水平
            revenue_scenarios: 收益场景列表
        """
        T = T or self.params["T"]
        eta = eta or self.params["eta"]
        s = s or self.params["s"]
        beta = beta or self.params["beta"]
        revenue_scenarios = revenue_scenarios or self.params["revenue_scenarios"]
        
        # CVaR约束
        for t in range(T):
            self.model.addConstr(
                s[t] >= revenue_scenarios[t] - eta
            )
            self.model.addConstr(s[t] >= 0)
    
    def add_all_constraints(self) -> None:
        self.add_uc_ev_constraints()
        self.add_cc_ev_constraints()
        self.add_es_constraints()
        self.add_market_constraints()
        self.add_cvar_constraints() 