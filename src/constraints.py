import gurobipy as gp
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

class V2GConstraints:
    def __init__(self, model: gp.Model):
        """初始化约束管理器
        
        Args:
            model: Gurobi优化模型
        """
        self.model = model

    def add_uc_ev_constraints(self, *, ev_profiles: pd.DataFrame, T: int, P_ev_uc: Dict, scenario_idx: int) -> None:
        """添加 uc ev 相关约束"""
        for i, row in ev_profiles[ev_profiles["ev_type"] == "uc"].iterrows():
            Ta_i = row["arrival_time"]
            Sa_i = row["soc_arrival"]
            Sd_i = row["soc_departure"]
            Eev_i = row["battery_capacity"]
            Pmax_i = row["max_charge_power"]
            eta_ev = row["efficiency"]
            
            # 计算 Tk_i,w
            Tk_i = Ta_i + ((Sd_i - Sa_i) * Eev_i) / (Pmax_i * eta_ev)
            
            for t in range(T):
                if Ta_i <= t <= Tk_i:
                    self.model.addConstr(P_ev_uc[scenario_idx, t, i] == Pmax_i)
                else:
                    self.model.addConstr(P_ev_uc[scenario_idx, t, i] == 0)

    def add_cc_ev_constraints(self, *, 
                               ev_profiles: pd.DataFrame,
                               delta_t: float,
                               T: int,
                               scenario_idx: int,
                               P_ev_cc: dict,
                               soc: dict,
                               P_ev0_cc: dict,
                               R_ev_up_i: dict,
                               R_ev_dn_i: dict,
                               R_ev_up: dict,
                               R_ev_dn: dict,
                               K_dn: dict) -> None:
        """
        添加cc ev相关约束
        P_ev_cc: cc类充电功率
        P_ev0_cc: cc类ev dam energy bids PSP值
        R_ev_up_i: 第i个cc类ev可上调容量
        R_ev_dn_i: 第i个cc类可下调容量
        R_ev_up: 所有cc类的可上调容量
        R_ev_dn: 所有cc类的可下调容量
        K_dn: capacity reservation(MWh)
        """
        cc_evs = ev_profiles[ev_profiles["ev_type"] == "cc"]

        for t in range(T):
            self.model.addConstr(
                R_ev_up[t] == gp.quicksum(R_ev_up_i[scenario_idx, t, i] for i in cc_evs.index)
            )
            self.model.addConstr(
                R_ev_dn[t] == gp.quicksum(R_ev_dn_i[scenario_idx, t, i] for i in cc_evs.index)
            )

            for i, row in cc_evs.iterrows():
                Ta = int(row["arrival_time"])
                Td = int(row["departure_time"])
                Sa = row["soc_arrival"]
                Sd = row["soc_departure"]
                Smax = row["soc_max"]
                Smin = row["soc_min"]
                Eev = row["battery_capacity"]
                Pmax = row["max_charge_power"]
                eta = row["efficiency"]

                if Ta <= t <= Td:
                    self.model.addConstr(P_ev0_cc[scenario_idx, t, i] >= 0)
                    self.model.addConstr(P_ev0_cc[scenario_idx, t, i] <= Pmax)
                    self.model.addConstr(R_ev_up_i[scenario_idx, t, i] >= 0)
                    self.model.addConstr(R_ev_up_i[scenario_idx, t, i] <= P_ev0_cc[scenario_idx, t, i])
                    self.model.addConstr(R_ev_dn_i[scenario_idx, t, i] >= 0)
                    self.model.addConstr(R_ev_dn_i[scenario_idx, t, i] <= Pmax - P_ev0_cc[scenario_idx, t, i])
                    self.model.addConstr(P_ev_cc[scenario_idx, t, i] >= 0)
                    self.model.addConstr(P_ev_cc[scenario_idx, t, i] <= Pmax)
                else:
                    self.model.addConstr(R_ev_up_i[scenario_idx, t, i] == 0)
                    self.model.addConstr(R_ev_dn_i[scenario_idx, t, i] == 0)
                    self.model.addConstr(P_ev0_cc[scenario_idx, t, i] == 0)
                    self.model.addConstr(P_ev_cc[scenario_idx, t, i] == 0)

        for i, row in cc_evs.iterrows():
            Ta = int(row["arrival_time"])
            Td = int(row["departure_time"])
            Sa = row["soc_arrival"]
            Sd = row["soc_departure"]
            Smax = row["soc_max"]
            Smin = row["soc_min"]
            Eev = row["battery_capacity"]
            eta = row["efficiency"]

            for t in range(Ta, Td + 1):
                self.model.addConstr(
                    soc[scenario_idx, t, i] == Sa +
                    gp.quicksum(P_ev_cc[scenario_idx, tau, i] * eta * delta_t / Eev for tau in range(Ta, t + 1))
                )

            self.model.addConstr(soc[scenario_idx, Td, i] >= Sd)
            self.model.addConstr(soc[scenario_idx, Td, i] <= Smax)
            self.model.addConstr(soc[scenario_idx, Td, i] >= Smin)

            for t in range(Ta, Td):
                self.model.addConstr(
                    soc[scenario_idx, t, i] <= Smax - K_dn[t + 1] / Eev
                )

    def add_ev_fleet_aggregate_constraints(self, *,
                                        ev_profiles: pd.DataFrame,
                                        scenario_idx: int,
                                        T: int,
                                        P_ev_uc: dict,
                                        P_ev_cc: dict,
                                        P_ev0_uc: dict,
                                        P_ev0_cc: dict,
                                        P_ev_total: dict,
                                        P_ev0_total: dict) -> None:
        """
        添加ev fleet相关约束
        P_ev_uc: uc ev充电功率
        P_ev_cc: cc ev充电功率
        P_ev0_uc: uc ev dam energy bids
        P_ev0_cc: cc ev dam energy bids
        P_ev_total: ev fleet 充电功率
        P_ev0_total: ev fleet dam energy bids
        """
        uc_evs = ev_profiles[ev_profiles["ev_type"] == "uc"]
        cc_evs = ev_profiles[ev_profiles["ev_type"] == "cc"]

        for t in range(T):
            self.model.addConstr(
                P_ev_total[t, scenario_idx] ==
                gp.quicksum(P_ev_uc[scenario_idx, t, i] for i in uc_evs.index) +
                gp.quicksum(P_ev_cc[scenario_idx, t, i] for i in cc_evs.index)
            )
            self.model.addConstr(
                P_ev0_total[t] ==
                gp.quicksum(P_ev0_uc[scenario_idx, t, i] for i in uc_evs.index) +
                gp.quicksum(P_ev0_cc[scenario_idx, t, i] for i in cc_evs.index)
            )


    def add_es_constraints(self, *,
                       T: int,
                       scenario_idx: int,
                       P_es: dict,
                       P_es_ch: dict,
                       P_es_dis: dict,
                       E_es: dict,
                       P_es1: dict,
                       P_es2: dict,
                       P_es1_ch: dict,
                       P_es1_dis: dict,
                       E_es1: dict,
                       E_es2: dict,
                       mu_es_ch: dict,
                       mu_es_dis: dict,
                       R_es_up: dict,
                       R_es_dn: dict,
                       P_es0: dict,
                       agc_up: dict,
                       agc_dn: dict,
                       K_up: float,
                       K_dn: float,
                       P_es_max: float,
                       E_es_max: float,
                       E_es_init: float,
                       dod_max: float,
                       eta_ch: float,
                       eta_dis: float,
                       delta_t: float,
                       kappa: float,
                       gamma: float 
                       ) -> None:
        """
        添加ES相关约束

        P_es_max: ES最大充放电功率（MW）
        E_es_max: ES最大能量容量（MWh）
        E_es_init: ES初始能量（MWh）
        dod_max: 最大允许DOD（0~1）
        gamma: 最终能量与初始能量偏差容忍率
        其余参数同前
        """
        for t in range(T):
            # 总功率 = 充电 - 放电
            self.model.addConstr(P_es1[scenario_idx, t] + P_es2[scenario_idx, t] ==
                                P_es_ch[scenario_idx, t] - P_es_dis[scenario_idx, t])

            # 最大功率和互斥充放电逻辑
            self.model.addConstr(P_es_ch[scenario_idx, t] <= mu_es_ch[t] * P_es_max)
            self.model.addConstr(P_es_dis[scenario_idx, t] <= mu_es_dis[t] * P_es_max)
            self.model.addConstr(mu_es_ch[t] + mu_es_dis[t] <= 1)

            # ES1 拆分充放电
            self.model.addConstr(P_es1[scenario_idx, t] == P_es1_ch[scenario_idx, t] - P_es1_dis[scenario_idx, t])
            self.model.addConstr(P_es1_ch[scenario_idx, t] <= mu_es_ch[t] * P_es_max)
            self.model.addConstr(P_es1_dis[scenario_idx, t] <= mu_es_dis[t] * P_es_max)

            # ES2 能量限制
            self.model.addConstr(E_es2[scenario_idx, t] >= 0)
            self.model.addConstr(E_es2[scenario_idx, t] <= E_es_max)

            # 总能量限制
            self.model.addConstr(E_es1[scenario_idx, t] + E_es2[scenario_idx, t] <= E_es_max)

            # ES1 功率上下限
            self.model.addConstr(P_es1[scenario_idx, t] >= -P_es_max)
            self.model.addConstr(P_es1[scenario_idx, t] <= P_es_max)

            # 调频边界（基于 PSP 偏移量）
            self.model.addConstr(R_es_up[t] >= 0)
            self.model.addConstr(R_es_up[t] <= P_es0[t] + P_es_max)
            self.model.addConstr(R_es_dn[t] >= 0)
            self.model.addConstr(R_es_dn[t] <= P_es_max - P_es0[t])

            # ES1 实际功率等于 PSP + 调频响应
            self.model.addConstr(
                P_es1[scenario_idx, t] ==
                P_es0[t] - R_es_up[t] * agc_up[t] + R_es_dn[t] * agc_dn[t]
            )

            # ES1 电量演化
            if t == 0:
                self.model.addConstr(E_es1[scenario_idx, 0] == E_es_init)
            else:
                self.model.addConstr(
                    E_es1[scenario_idx, t] ==
                    E_es1[scenario_idx, t - 1] +
                    (eta_ch * P_es1_ch[scenario_idx, t] -
                    P_es1_dis[scenario_idx, t] / eta_dis) * delta_t
                )

            # 动态 SOC 上限（考虑未来调频预留容量）
            if t + 1 < T:
                self.model.addConstr(
                    K_up * R_es_up[t+1] <= E_es1[scenario_idx, t]
                )
                self.model.addConstr(
                    E_es1[scenario_idx, t] <= E_es_max - K_dn * R_es_dn[t+1]
            )

            # DOD约束（最大允许放电深度）
            self.model.addConstr(E_es1[scenario_idx, t] >= (1 - dod_max) * E_es_max)

            # 最终电量边界（允许 ±gamma 偏移）
            self.model.addConstr(
                E_es1[scenario_idx, T-1] + E_es2[scenario_idx, T-1] >= (1 - gamma) * E_es_init
            )
            self.model.addConstr(
                E_es1[scenario_idx, T-1] + E_es2[scenario_idx, T-1] <= (1 + gamma) * E_es_init
            )


    def add_energy_balance_constraints(self, *,
                                        T: int,
                                        ev_profiles: pd.DataFrame,
                                        scenario_idx: int,
                                        P_ev0: Dict,         # EV 投标功率（日前）
                                        P_es0: Dict,         # ES 投标功率（日前）
                                        R_ev_up: Dict,       # EV上调容量
                                        R_ev_dn: Dict,       # EV下调容量
                                        agc_up: Dict,       # 单位MW上调能量偏移
                                        agc_dn: Dict,       # 单位MW下调能量偏移
                                        P_ev_uc: Dict,       # 不可控EV功率
                                        P_ev_cc: Dict,       # 可控EV功率
                                        P_es2_ch: Dict,      # ES2 充电功率
                                        P_es2_dis: Dict,     # ES2 放电功率
                                        ) -> None:
        """
        添加能量平衡约束：
        实际充电功率 - 上调放弃 + 下调增加 = 实际总功率 + ES补偿

        P_ev0: ev日前投标功率
        P_es0: es日前投标功率
        R_ev_up: ev上调频容量
        R_ev_dn: ev下调频容量
        agc_up: 上调频指令 MWh/MW
        agc_dn: 下调频指令 MWh/MW

        P_ev_uc: uc ev实际充电功率
        P_ev_cc: cc ev实际充电功率
        P_es2_ch: es2充电功率(补偿ev)
        P_es2_dis: es2放电功率(补偿ev)
        """
        uc_evs = ev_profiles[ev_profiles["ev_type"] == "uc"]
        cc_evs = ev_profiles[ev_profiles["ev_type"] == "cc"]
        for t in range(T):
            lhs = (P_ev0[t]
                - agc_up[t, scenario_idx] * R_ev_up[t]
                + agc_dn[t, scenario_idx] * R_ev_dn[t])   #参考的是论文里的式子 表示考虑调频后，EV这一时刻理论上应该充进去的总能量（MW）

            rhs = (
                gp.quicksum(P_ev_uc[scenario_idx, t, i] for i in uc_evs) +
                gp.quicksum(P_ev_cc[scenario_idx, t, i] for i in cc_evs) +
                P_es2_ch[scenario_idx, t] - P_es2_dis[scenario_idx, t]
            )  #表示系统在这一时刻对 EV 和 ES2 实际提供的总功率（MW）

            self.model.addConstr(lhs == rhs)



    def add_cvar_constraints(self, *,
                            num_scenarios: int,
                            sigma: gp.Var,  
                            phi: Dict[int, gp.Var], 
                            f: List[gp.LinExpr]               
                            ) -> None:
        """
        添加CVaR约束

            num_scenarios: 场景数量
            sigma: VaR变量 σ  利润的风险值
            phi: 辅助变量 φ_w 的字典
            f: 每个场景对应的损失或负收益表达式 f_w
        """
        for w in range(num_scenarios):
            # φ_w ≥ σ - f_w
            self.model.addConstr(phi[w] >= sigma - f[w])
            # φ_w ≥ 0
            self.model.addConstr(phi[w] >= 0)



    def add_battery_degradation_constraints(self, *,
                                            d: gp.Var,
                                            L: gp.Var,
                                            dod_life_product_max: float) -> None:
        """
        添加DOD和电池生命周期的乘积约束，用于控制电池退化行为
        
        Args:
            d: DOD变量（Depth of Discharge，0~1）
            L: 电池循环寿命变量（life cycles）
            dod_life_product_max: 常数，用于近似控制 d * L ≤ constant，例如来自经验图像的上界
        """
        # 加入 d * L ≤ 上界 的约束
        self.model.addConstr(d * L <= dod_life_product_max)  #这里具体的表达式得看下参考文献 暂时先这样



    def add_all_constraints(self) -> None:
        self.add_uc_ev_constraints()
        self.add_cc_ev_constraints()
        self.add_es_constraints()
        self.add_ev_fleet_aggregate_constraints()
        self.add_cvar_constraints()
        self.add_battery_degradation_constraints()
