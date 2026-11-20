from typing import Dict, List
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd


class V2GConstraintsCase1:

    def __init__(self, model: gp.Model):
        """初始化约束管理器
        
        Args:
            model: Gurobi优化模型
            W: 场景数量，默认为4
        """
        self.model = model

    def add_uc_ev_constraints(self, *, 
                                ev_profiles: pd.DataFrame, 
                                T: int, 
                                delta_t: float,
                                P_ev_uc: dict, 
                                P_ev0_uc: dict,
                                scenario_idx: int, 
                                soc: dict,
                                N_uc: int) -> None:
        uc_evs = ev_profiles[ev_profiles["ev_type"] == "uc"].reset_index(drop=True)
        for n in range(N_uc):
            row = uc_evs.iloc[n]
            Ta_i = int(row["arrival_time"])
            Td_i = int(row["departure_time"])
            Sa_i = row["soc_arrival"]
            Sd_i = row["soc_departure"]
            Eev_i = row["battery_capacity"]
            Pmax = row["max_charge_power"]
            eta_ev = row["efficiency"]
            
            # 计算UC车辆的固定功率曲线 
            power_profile = np.zeros(T) # 初始化功率曲线全为0
            
            if Eev_i > 0 and Pmax > 0 and Sa_i < Sd_i:
                # 需要充入的总能量 (MWh)
                energy_needed = (Sd_i - Sa_i) * Eev_i
                # 每个满功率时段能充入的能量 (MWh)
                energy_per_slot = Pmax * eta_ev * delta_t                
                # 需要的总充电时段数（大概率是小数）
                duration_slots_float = energy_needed / energy_per_slot                
                # 满功率充电的时段数
                num_full_slots = int(duration_slots_float)               
                # 最后一个时段需要充的能量
                remaining_energy = energy_needed - num_full_slots * energy_per_slot                
                # 最后一个时段的精确功率 (MW)
                final_slot_power = remaining_energy / (eta_ev * delta_t)
                # 填充所有满功率时段
                for i in range(num_full_slots):
                    t = Ta_i + i
                    if t < Td_i: # 确保充电在离开前
                        power_profile[t] = Pmax
                
                # 填充最后一个部分功率时段
                t_final = Ta_i + num_full_slots
                if t_final < Td_i: # 确保在离开前
                    power_profile[t_final] = final_slot_power

            # 在模型中施加约束 
            for t in range(T):
                # 将预先计算好的刚性功率曲线作为约束
                self.model.addConstr(P_ev_uc[scenario_idx, t, n] == power_profile[t])
                # UC 车辆：投标等于其充电功率
                self.model.addConstr(P_ev0_uc[scenario_idx, t, n] == P_ev_uc[scenario_idx, t, n])
                
                # SoC 的演变现在依赖于这个固定的功率曲线
                if t < Ta_i:
                    self.model.addConstr(soc[scenario_idx, t, n] == 0)
                elif t == Ta_i:
                    self.model.addConstr(soc[scenario_idx, t, n] == Sa_i + (power_profile[t] * eta_ev * delta_t / Eev_i))
                else: # t > Ta_i
                    self.model.addConstr(soc[scenario_idx, t, n] == soc[scenario_idx, t - 1, n] + (power_profile[t] * eta_ev * delta_t / Eev_i))

            # 添加最终SoC约束作为验证
            final_t = min(Td_i - 1, T - 1)
            if final_t >= Ta_i:
                # 使用小的容差避免浮点数精度问题
                self.model.addConstr(soc[scenario_idx, final_t, n] >= Sd_i - 1e-6)

    def add_cc_ev_constraints(self, *, 
                           ev_profiles: pd.DataFrame,
                           delta_t: float,
                           T: int,
                           scenario_idx: int,
                           P_ev_cc: dict,
                           soc: dict,
                           P_ev_cc_ch: dict,
                           P_ev_cc_dis: dict,
                           P_ev_cc_ch_eff: dict,
                           P_ev_cc_dis_eff: dict,
                           y_ev: dict,                           
                           P_ev0_cc: dict,
                           R_ev_up_i: dict,
                           R_ev_dn_i: dict,
                           R_ev_up: dict,  #不依赖场景
                           R_ev_dn: dict,  #不依赖场景
                           K_up_values: np.ndarray,  # 每个时段的上调频容量预留值
                           K_dn_values: np.ndarray,  # 每个时段的下调频容量预留值
                           N_cc: int) -> None:
        """添加可控EV约束"""
        cc_evs = ev_profiles[ev_profiles["ev_type"] == "cc"].reset_index(drop=True)
        
        for t in range(T):
            self.model.addConstr(
                R_ev_up[t] == gp.quicksum(R_ev_up_i[scenario_idx, t, n] for n in range(N_cc))
            )
            self.model.addConstr(
                R_ev_dn[t] == gp.quicksum(R_ev_dn_i[scenario_idx, t, n] for n in range(N_cc))
            )
            for n in range(N_cc):
                row = cc_evs.iloc[n]
                Ta = int(row["arrival_time"])
                Td = int(row["departure_time"])
                Sa = row["soc_arrival"]
                Sd = row["soc_departure"]
                Smax = row["soc_max"]
                Smin = row["soc_min"]
                Eev = row["battery_capacity"]  # 现在是MWh
                Pmax = row["max_charge_power"]  # 现在是MW
                eta = row["efficiency"]
                
                # 处理夜间充电跨天的情况
                is_overnight = row["charging_type"] == "night" if "charging_type" in row else False
                
                # 夜间充电EV：
                # 当天晚上到达，第二天早上离开 (Td > 95)
                if is_overnight:
                    actual_td = Td - 96  # 计算实际的第二天离开时间槽
                    is_charging_time = (t >= Ta) or (t <= actual_td) 
                else:
                    # 处理白天充电（左闭右开）
                    is_charging_time = (Ta <= t < Td)
                    
                if is_charging_time:
                    # Baseline and net power are symmetric for V2G
                    self.model.addConstr(P_ev0_cc[scenario_idx, t, n] >= -Pmax)
                    self.model.addConstr(P_ev0_cc[scenario_idx, t, n] <= Pmax)
                    # Up capacity limited by headroom to discharge side
                    self.model.addConstr(R_ev_up_i[scenario_idx, t, n] <= Pmax + P_ev0_cc[scenario_idx, t, n])
                    self.model.addConstr(R_ev_up_i[scenario_idx, t, n] >= 0)
                    self.model.addConstr(R_ev_dn_i[scenario_idx, t, n] >= 0)
                    self.model.addConstr(R_ev_dn_i[scenario_idx, t, n] <= Pmax - P_ev0_cc[scenario_idx, t, n])
                    self.model.addConstr(P_ev_cc[scenario_idx, t, n] >= -Pmax)
                    self.model.addConstr(P_ev_cc[scenario_idx, t, n] <= Pmax)
                    # Charge/discharge components
                    self.model.addConstr(P_ev_cc_ch[scenario_idx, t, n] >= 0)
                    self.model.addConstr(P_ev_cc_dis[scenario_idx, t, n] >= 0)
                    self.model.addConstr(P_ev_cc[scenario_idx, t, n] == P_ev_cc_ch[scenario_idx, t, n] - P_ev_cc_dis[scenario_idx, t, n])
                    # 最大功率和互斥充放电逻辑（逐车逐时段）
                    self.model.addConstr(P_ev_cc_ch[scenario_idx, t, n] <= y_ev[scenario_idx, t, n] * Pmax)
                    self.model.addConstr(P_ev_cc_dis[scenario_idx, t, n] <= (1 - y_ev[scenario_idx, t, n]) * Pmax)
                    self.model.addConstr(y_ev[scenario_idx, t, n] + (1 - y_ev[scenario_idx, t, n]) <= 1)

                    # 使用PWL表达效率阈值（4kW=0.004MW）
                    # 充电：小于4kW效率0.5，≥4kW效率0.95
                    x_breaks_ch = [0.0, 4e-3, Pmax]
                    y_breaks_ch = [0.0, 0.5 * 4e-3, 0.5 * 4e-3 + 0.95 * (Pmax - 4e-3)]
                    self.model.addGenConstrPWL(P_ev_cc_ch[scenario_idx, t, n], P_ev_cc_ch_eff[scenario_idx, t, n], x_breaks_ch, y_breaks_ch)

                    # 放电：小于4kW效率0.5，≥4kW效率0.95（等效为能量消耗系数 1/效率）
                    x_breaks_dis = [0.0, 4e-3, Pmax]
                    y_breaks_dis = [0.0, 0.5 * 4e-3, 0.5 * 4e-3 + (1.0 / 0.95) * (Pmax - 4e-3)]
                    self.model.addGenConstrPWL(P_ev_cc_dis[scenario_idx, t, n], P_ev_cc_dis_eff[scenario_idx, t, n], x_breaks_dis, y_breaks_dis)

                    # 修改SOC计算逻辑，处理跨天充电情况
                    if is_overnight:
                        actual_td = Td - 96
                        if t == Ta:  # 初始时刻
                            self.model.addConstr(
                                soc[scenario_idx, t, n] == Sa +
                                ((P_ev_cc_ch_eff[scenario_idx, t, n] * eta - P_ev_cc_dis_eff[scenario_idx, t, n] * 1/eta) * delta_t / Eev)
                            )
                        if t > Ta or t <= actual_td:  # 过程
                            if t == 0:  # 新的一天开始
                                prev_t = T - 1  # 使用前一天的最后时刻
                            else:
                                prev_t = t - 1
                                
                            self.model.addConstr(
                                soc[scenario_idx, t, n] == soc[scenario_idx, prev_t, n] +
                                ((P_ev_cc_ch_eff[scenario_idx, t, n] * eta - P_ev_cc_dis_eff[scenario_idx, t, n] * 1/eta) * delta_t / Eev)
                            )
                            
                        # 离开时SOC要求
                        if t == actual_td - 1:
                            self.model.addConstr(soc[scenario_idx, t, n] >= Sd)
                            self.model.addConstr(soc[scenario_idx, t, n] <= Smax)
                    else:  # 常规情况（不跨天）
                        if t == Ta:  # 刚到达
                            self.model.addConstr(
                                soc[scenario_idx, t, n] == Sa + 
                                ((P_ev_cc_ch_eff[scenario_idx, t, n] * eta - P_ev_cc_dis_eff[scenario_idx, t, n] * 1/eta) * delta_t / Eev)
                            )
                        elif t > Ta:  # 过程
                            self.model.addConstr(
                                soc[scenario_idx, t, n] == soc[scenario_idx, t-1, n] +
                                ((P_ev_cc_ch_eff[scenario_idx, t, n] * eta - P_ev_cc_dis_eff[scenario_idx, t, n] * 1/eta) * delta_t / Eev)
                            )
                        
                        # 离开时SOC要求
                        if t == Td - 1:
                            self.model.addConstr(soc[scenario_idx, t, n] >= Sd)
                            self.model.addConstr(soc[scenario_idx, t, n] <= Smax)
                   
                    # 确保所有时刻的SOC都在合理范围内
                    self.model.addConstr(soc[scenario_idx, t, n] >= Smin)
                    self.model.addConstr(soc[scenario_idx, t, n] <= Smax)
                    
                else:
                    self.model.addConstr(R_ev_up_i[scenario_idx, t, n] == 0)
                    self.model.addConstr(R_ev_dn_i[scenario_idx, t, n] == 0)
                    self.model.addConstr(P_ev0_cc[scenario_idx, t, n] == 0)
                    self.model.addConstr(P_ev_cc[scenario_idx, t, n] == 0)
                    self.model.addConstr(P_ev_cc_ch[scenario_idx, t, n] == 0)
                    self.model.addConstr(P_ev_cc_dis[scenario_idx, t, n] == 0)
                    self.model.addConstr(y_ev[scenario_idx, t, n] == 0)
                
            for n in range(N_cc): #这里再循环一次是因为时间不同
                row = cc_evs.iloc[n]
                Ta = int(row["arrival_time"])
                Td = int(row["departure_time"])
                Eev = row["battery_capacity"]  # 现在是MWh
                Smax = row["soc_max"]
                Smin = row["soc_min"]
                
                # 处理夜间充电跨天的情况
                is_overnight = row["charging_type"] == "night" if "charging_type" in row else False
                
                if is_overnight:
                    # 当天晚上到达，第二天早上离开
                    actual_td = Td - 96  # 计算实际的第二天离开时间槽
                    # 容量预留约束，分两段
                    if Ta <= t <= T-2:  # 第一天晚上的部分
                        self.model.addConstr(
                            soc[scenario_idx, t, n] <= Smax - K_dn_values[t+1] * R_ev_dn_i[scenario_idx, t+1, n] / Eev
                        )
                        # 上调侧能量可行性：需要足够可放电余量
                        self.model.addConstr(
                            soc[scenario_idx, t, n] >= Smin + K_up_values[t+1] * R_ev_up_i[scenario_idx, t+1, n] / Eev
                        )
                    elif 0 <= t <= actual_td-1:  # 第二天早上的部分
                        self.model.addConstr(
                            soc[scenario_idx, t, n] <= Smax - K_dn_values[t+1] * R_ev_dn_i[scenario_idx, t+1, n] / Eev
                        )
                        self.model.addConstr(
                            soc[scenario_idx, t, n] >= Smin + K_up_values[t+1] * R_ev_up_i[scenario_idx, t+1, n] / Eev
                        )
                else:
                    # 常规情况
                    if Ta <= t < min(Td-1, T-1):
                        self.model.addConstr(
                            soc[scenario_idx, t, n] <= Smax - K_dn_values[t+1] * R_ev_dn_i[scenario_idx, t+1, n] / Eev
                        )
                        self.model.addConstr(
                            soc[scenario_idx, t, n] >= Smin + K_up_values[t+1] * R_ev_up_i[scenario_idx, t+1, n] / Eev
                        )


    def add_ev_fleet_aggregate_constraints(self, *,
                                        scenario_idx: int,
                                        T: int,
                                        P_ev_uc: dict,
                                        P_ev_cc: dict,
                                        P_ev0_uc: dict,
                                        P_ev0_cc: dict,
                                        P_ev_total: dict,
                                        P_ev0_total: dict,
                                        N_cc: int,
                                        N_uc: int) -> None:

        for t in range(T):
            # 实际充电功率约束
            self.model.addConstr(
                P_ev_total[scenario_idx, t] ==
                gp.quicksum(P_ev_uc[scenario_idx, t, n] for n in range(N_uc)) +
                gp.quicksum(P_ev_cc[scenario_idx, t, n] for n in range(N_cc))
            )
            
            # 投标总和等于全局投标
            self.model.addConstr(
                P_ev0_total[t] == 
                gp.quicksum(P_ev0_uc[scenario_idx, t, n] for n in range(N_uc)) +
                gp.quicksum(P_ev0_cc[scenario_idx, t, n] for n in range(N_cc))
            )

    def add_es_constraints(self, *,
                       T: int,
                       scenario_idx: int,
                       P_es: dict,
                       P_es_ch: dict,
                       P_es_dis: dict,
                       E_es: dict,
                       y_es: dict,
                       P_es_max: float,
                       E_es_max: float,
                       E_es_init: float,
                       eta_ch: float,
                       eta_dis: float,
                       delta_t: float,                                            
                       dod: gp.Var,
                       gamma: float,
                       kappa: float
                       ) -> None:
            """
            添加ES2相关约束
            区别于ES1 ES2不直接参与调频 作为了EV调频任务的backup 不会在dam bids
            """

            for t in range(T):
                # ES 能量限制  DOD约束（最大允许放电深度）
                self.model.addConstr(E_es[scenario_idx, t] >= dod*E_es_max)
                self.model.addConstr(E_es[scenario_idx, t] <= E_es_max)

                # ES 拆分充放电&充放电互斥
                self.model.addConstr(P_es[scenario_idx, t] == P_es_ch[scenario_idx, t] - P_es_dis[scenario_idx, t])
                self.model.addConstr(P_es_ch[scenario_idx, t] <= y_es[scenario_idx, t] * P_es_max)
                self.model.addConstr(P_es_dis[scenario_idx, t] <= (1 - y_es[scenario_idx, t]) * P_es_max)
                self.model.addConstr(y_es[scenario_idx, t] + (1 - y_es[scenario_idx, t]) <= 1)
                
                # ES 电量演化
                if t == 0:
                    self.model.addConstr(E_es[scenario_idx, 0] == E_es_init)
                else:
                    self.model.addConstr(
                        E_es[scenario_idx, t] ==
                        E_es[scenario_idx, t - 1] +
                        (eta_ch * P_es_ch[scenario_idx, t] -
                        P_es_dis[scenario_idx, t] / eta_dis) * delta_t
                    )

            # 最终电量边界（允许 ±gamma 偏移）
            self.model.addConstr(
                E_es[scenario_idx, T-1] >= (1 - gamma) * E_es_init,
                name=f"es_final_energy_lower_{scenario_idx}"
            )
            self.model.addConstr(
                E_es[scenario_idx, T-1] <= (1 + gamma) * E_es_init,
                name=f"es_final_energy_upper_{scenario_idx}"
            )

            self.model.addConstr(P_es_max == E_es_max * kappa)

    def add_es_backup_constraints(self, *,
                                        T: int,
                                        scenario_idx: int,
                                        P_ev0_total: dict,         
                                        R_ev_up: dict,       
                                        R_ev_dn: dict,       
                                        agc_up: pd.Series,       
                                        agc_dn: pd.Series,             
                                        P_es_ch: dict,     
                                        P_es_dis: dict,
                                        P_es_ch_i: dict,     
                                        P_es_dis_i: dict,
                                        P_ev_total: dict,
                                        N_cc: int
                                        ) -> None:

        for t in range(T):
            # ES总功率约束
            self.model.addConstr(
                    P_es_ch[scenario_idx, t] == gp.quicksum(P_es_ch_i[scenario_idx, t, n] for n in range(N_cc))
                )
            self.model.addConstr(
                    P_es_dis[scenario_idx, t] == gp.quicksum(P_es_dis_i[scenario_idx, t, n] for n in range(N_cc))
                )
            
            # 简单的功率平衡约束：EV调频目标 = EV实际功率 + ES净功率
            # P_ev0_total - agc_up * R_ev_up + agc_dn * R_ev_dn = P_ev_total + P_es_ch - P_es_dis
            self.model.addConstr(
                P_ev0_total[t] - agc_up[t] * R_ev_up[t] + agc_dn[t] * R_ev_dn[t] == 
                P_ev_total[scenario_idx, t] + P_es_ch[scenario_idx, t] - P_es_dis[scenario_idx, t],
                name=f"power_balance_backup[{scenario_idx},{t}]"
            )

    def add_cvar_constraints(self, *,
                            num_scenarios: int,
                            sigma: gp.Var,  
                            phi: Dict[int, gp.Var], 
                            f: List[gp.LinExpr],
                            pi: float,
                            beta: float                
                            ) -> None:
        """
        添加CVaR约束
        
        Args:
            num_scenarios: 场景数量
            sigma: VaR变量σ（利润的风险值）
            phi: 辅助变量 φ_w 的字典
            f: 每个场景对应的收益表达式 f_w
            pi: 每个场景的概率
            beta: 置信水平 (0到1之间)
        """
        for w in range(num_scenarios):
            # φ_w ≥ 0
            self.model.addConstr(phi[w] >= 0, name=f"phi_positive[{w}]")
            # φ_w ≥ σ - f_w
            self.model.addConstr(phi[w] >= sigma - f[w], name=f"phi_def[{w}]")



    # def add_battery_degradation_constraints(self, *,
    #                                         d: gp.Var,      # DoD ∈ (0, 0.9]
    #                                         L: gp.Var,      # cycle life
    #                                         N0: float,
    #                                         beta: float
    #                                         ) -> None:
    #     """
    #     添加电池退化约束：L = N0 / d^beta
    #     即：d^beta * L = N0
    #     """

    #     # 创建中间变量：d^beta
    #     d_pow = self.model.addVar(name="d_pow")

    #     # 添加幂约束：d_pow = d^beta
    #     self.model.addGenConstrPow(d, d_pow, beta, name="d_pow_constr")

    #     # 添加主约束：d_pow * L = N0
    #     self.model.addQConstr(d_pow * L == N0, name="cycle_dod") 


    def add_all_constraints(self) -> None:
        self.add_uc_ev_constraints()
        self.add_cc_ev_constraints()
        self.add_ev_fleet_aggregate_constraints()
        self.add_es_constraints()
        self.add_es_backup_constraints()
        self.add_cvar_constraints()
        # self.add_battery_degradation_constraints()

class V2GConstraintsCase2:
    def __init__(self, model: gp.Model):
        """初始化约束管理器
        
        Args:
            model: Gurobi优化模型
        """
        self.model = model

    def add_uc_ev_constraints(self, *, 
                                ev_profiles: pd.DataFrame, 
                                T: int, 
                                delta_t: float,
                                P_ev_uc: dict, 
                                P_ev0_uc: dict,
                                scenario_idx: int, 
                                soc: dict,
                                N_uc: int) -> None:
        uc_evs = ev_profiles[ev_profiles["ev_type"] == "uc"].reset_index(drop=True)
        for n in range(N_uc):
            row = uc_evs.iloc[n]
            Ta_i = int(row["arrival_time"])
            Td_i = int(row["departure_time"])
            Sa_i = row["soc_arrival"]
            Sd_i = row["soc_departure"]
            Eev_i = row["battery_capacity"]
            Pmax = row["max_charge_power"]
            eta_ev = row["efficiency"]
            
            # 计算UC车辆的固定功率曲线 
            power_profile = np.zeros(T) # 初始化功率曲线全为0
            
            if Eev_i > 0 and Pmax > 0 and Sa_i < Sd_i:
                # 需要充入的总能量 (MWh)
                energy_needed = (Sd_i - Sa_i) * Eev_i
                # 每个满功率时段能充入的能量 (MWh)
                energy_per_slot = Pmax * eta_ev * delta_t                
                # 需要的总充电时段数（大概率是小数）
                duration_slots_float = energy_needed / energy_per_slot                
                # 满功率充电的时段数
                num_full_slots = int(duration_slots_float)               
                # 最后一个时段需要充的能量
                remaining_energy = energy_needed - num_full_slots * energy_per_slot                
                # 最后一个时段的精确功率 (MW)
                final_slot_power = remaining_energy / (eta_ev * delta_t)
                # 填充所有满功率时段
                for i in range(num_full_slots):
                    t = Ta_i + i
                    if t < Td_i: # 确保充电在离开前
                        power_profile[t] = Pmax
                
                # 填充最后一个部分功率时段
                t_final = Ta_i + num_full_slots
                if t_final < Td_i: # 确保在离开前
                    power_profile[t_final] = final_slot_power

            # 在模型中施加约束 
            for t in range(T):
                # 将预先计算好的刚性功率曲线作为约束
                self.model.addConstr(P_ev_uc[scenario_idx, t, n] == power_profile[t])
                # UC 车辆：投标等于其充电功率
                self.model.addConstr(P_ev0_uc[scenario_idx, t, n] == P_ev_uc[scenario_idx, t, n])
                
                # SoC 的演变现在依赖于这个固定的功率曲线
                if t < Ta_i:
                    self.model.addConstr(soc[scenario_idx, t, n] == 0)
                elif t == Ta_i:
                    self.model.addConstr(soc[scenario_idx, t, n] == Sa_i + (power_profile[t] * eta_ev * delta_t / Eev_i))
                else: # t > Ta_i
                    self.model.addConstr(soc[scenario_idx, t, n] == soc[scenario_idx, t - 1, n] + (power_profile[t] * eta_ev * delta_t / Eev_i))

            # 添加最终SoC约束作为验证
            final_t = min(Td_i - 1, T - 1)
            if final_t >= Ta_i:
                # 使用小的容差避免浮点数精度问题
                self.model.addConstr(soc[scenario_idx, final_t, n] >= Sd_i - 1e-6)

    def add_cc_ev_constraints(self, *, 
                           ev_profiles: pd.DataFrame,
                           delta_t: float,
                           T: int,
                           scenario_idx: int,
                           P_ev_cc: dict,
                           soc: dict,
                           P_ev_cc_ch: dict,
                           P_ev_cc_dis: dict,
                           P_ev_cc_ch_eff: dict,
                           P_ev_cc_dis_eff: dict,
                           y_ev: dict,                           
                           P_ev0_cc: dict,
                           R_ev_up_i: dict,
                           R_ev_dn_i: dict,
                           R_ev_up: dict,  #不依赖场景
                           R_ev_dn: dict,  #不依赖场景
                           K_up_values: np.ndarray,  # 每个时段的上调频容量预留值
                           K_dn_values: np.ndarray,  # 每个时段的下调频容量预留值
                           N_cc: int) -> None:
        """添加可控EV约束"""
        cc_evs = ev_profiles[ev_profiles["ev_type"] == "cc"].reset_index(drop=True)
        
        for t in range(T):
            self.model.addConstr(
                R_ev_up[t] == gp.quicksum(R_ev_up_i[scenario_idx, t, n] for n in range(N_cc))
            )
            self.model.addConstr(
                R_ev_dn[t] == gp.quicksum(R_ev_dn_i[scenario_idx, t, n] for n in range(N_cc))
            )
            for n in range(N_cc):
                row = cc_evs.iloc[n]
                Ta = int(row["arrival_time"])
                Td = int(row["departure_time"])
                Sa = row["soc_arrival"]
                Sd = row["soc_departure"]
                Smax = row["soc_max"]
                Smin = row["soc_min"]
                Eev = row["battery_capacity"]  # 现在是MWh
                Pmax = row["max_charge_power"]  # 现在是MW
                eta = row["efficiency"]
                
                # 处理夜间充电跨天的情况
                is_overnight = row["charging_type"] == "night" if "charging_type" in row else False
                
                # 夜间充电EV：
                # 当天晚上到达，第二天早上离开 (Td > 95)
                if is_overnight:
                    actual_td = Td - 96  # 计算实际的第二天离开时间槽
                    is_charging_time = (t >= Ta) or (t <= actual_td) 
                else:
                    # 处理白天充电（左闭右开）
                    is_charging_time = (Ta <= t < Td)
                    
                if is_charging_time:
                    # Baseline and net power are symmetric for V2G
                    self.model.addConstr(P_ev0_cc[scenario_idx, t, n] >= -Pmax)
                    self.model.addConstr(P_ev0_cc[scenario_idx, t, n] <= Pmax)
                    # Up capacity limited by headroom to discharge side
                    self.model.addConstr(R_ev_up_i[scenario_idx, t, n] <= Pmax + P_ev0_cc[scenario_idx, t, n])
                    self.model.addConstr(R_ev_up_i[scenario_idx, t, n] >= 0)
                    self.model.addConstr(R_ev_dn_i[scenario_idx, t, n] >= 0)
                    self.model.addConstr(R_ev_dn_i[scenario_idx, t, n] <= Pmax - P_ev0_cc[scenario_idx, t, n])
                    self.model.addConstr(P_ev_cc[scenario_idx, t, n] >= -Pmax)
                    self.model.addConstr(P_ev_cc[scenario_idx, t, n] <= Pmax)
                    # Charge/discharge components
                    self.model.addConstr(P_ev_cc_ch[scenario_idx, t, n] >= 0)
                    self.model.addConstr(P_ev_cc_dis[scenario_idx, t, n] >= 0)
                    self.model.addConstr(P_ev_cc[scenario_idx, t, n] == P_ev_cc_ch[scenario_idx, t, n] - P_ev_cc_dis[scenario_idx, t, n])
                    # 最大功率和互斥充放电逻辑（逐车逐时段）
                    self.model.addConstr(P_ev_cc_ch[scenario_idx, t, n] <= y_ev[scenario_idx, t, n] * Pmax)
                    self.model.addConstr(P_ev_cc_dis[scenario_idx, t, n] <= (1 - y_ev[scenario_idx, t, n]) * Pmax)
                    self.model.addConstr(y_ev[scenario_idx, t, n] + (1 - y_ev[scenario_idx, t, n]) <= 1)

                    # 使用PWL表达效率阈值（4kW=0.004MW）
                    # 充电：小于4kW效率0.5，≥4kW效率0.95
                    x_breaks_ch = [0.0, 4e-3, Pmax]
                    y_breaks_ch = [0.0, 0.5 * 4e-3, 0.5 * 4e-3 + 0.95 * (Pmax - 4e-3)]
                    self.model.addGenConstrPWL(P_ev_cc_ch[scenario_idx, t, n], P_ev_cc_ch_eff[scenario_idx, t, n], x_breaks_ch, y_breaks_ch)

                    # 放电：小于4kW效率0.5，≥4kW效率0.95（等效为能量消耗系数 1/效率）
                    x_breaks_dis = [0.0, 4e-3, Pmax]
                    y_breaks_dis = [0.0, 0.5 * 4e-3, 0.5 * 4e-3 + (1.0 / 0.95) * (Pmax - 4e-3)]
                    self.model.addGenConstrPWL(P_ev_cc_dis[scenario_idx, t, n], P_ev_cc_dis_eff[scenario_idx, t, n], x_breaks_dis, y_breaks_dis)

                    # 修改SOC计算逻辑，处理跨天充电情况
                    if is_overnight:
                        actual_td = Td - 96
                        if t == Ta:  # 初始时刻
                            self.model.addConstr(
                                soc[scenario_idx, t, n] == Sa +
                                ((P_ev_cc_ch_eff[scenario_idx, t, n] * eta - P_ev_cc_dis_eff[scenario_idx, t, n] * 1/eta) * delta_t / Eev)
                            )
                        if t > Ta or t <= actual_td:  # 过程
                            if t == 0:  # 新的一天开始
                                prev_t = T - 1  # 使用前一天的最后时刻
                            else:
                                prev_t = t - 1
                                
                            self.model.addConstr(
                                soc[scenario_idx, t, n] == soc[scenario_idx, prev_t, n] +
                                ((P_ev_cc_ch_eff[scenario_idx, t, n] * eta - P_ev_cc_dis_eff[scenario_idx, t, n] * 1/eta) * delta_t / Eev)
                            )
                            
                        # 离开时SOC要求
                        if t == actual_td - 1:
                            self.model.addConstr(soc[scenario_idx, t, n] >= Sd)
                            self.model.addConstr(soc[scenario_idx, t, n] <= Smax)
                    else:  # 常规情况（不跨天）
                        if t == Ta:  # 刚到达
                            self.model.addConstr(
                                soc[scenario_idx, t, n] == Sa + 
                                ((P_ev_cc_ch_eff[scenario_idx, t, n] * eta - P_ev_cc_dis_eff[scenario_idx, t, n] * 1/eta) * delta_t / Eev)
                            )
                        elif t > Ta:  # 过程
                            self.model.addConstr(
                                soc[scenario_idx, t, n] == soc[scenario_idx, t-1, n] +
                                ((P_ev_cc_ch_eff[scenario_idx, t, n] * eta - P_ev_cc_dis_eff[scenario_idx, t, n] * 1/eta) * delta_t / Eev)
                            )
                        
                        # 离开时SOC要求
                        if t == Td - 1:
                            self.model.addConstr(soc[scenario_idx, t, n] >= Sd)
                            self.model.addConstr(soc[scenario_idx, t, n] <= Smax)
                   
                    # 确保所有时刻的SOC都在合理范围内
                    self.model.addConstr(soc[scenario_idx, t, n] >= Smin)
                    self.model.addConstr(soc[scenario_idx, t, n] <= Smax)
                    
                else:
                    self.model.addConstr(R_ev_up_i[scenario_idx, t, n] == 0)
                    self.model.addConstr(R_ev_dn_i[scenario_idx, t, n] == 0)
                    self.model.addConstr(P_ev0_cc[scenario_idx, t, n] == 0)
                    self.model.addConstr(P_ev_cc[scenario_idx, t, n] == 0)
                    self.model.addConstr(P_ev_cc_ch[scenario_idx, t, n] == 0)
                    self.model.addConstr(P_ev_cc_dis[scenario_idx, t, n] == 0)
                    self.model.addConstr(y_ev[scenario_idx, t, n] == 0)
                
            for n in range(N_cc): #这里再循环一次是因为时间不同
                row = cc_evs.iloc[n]
                Ta = int(row["arrival_time"])
                Td = int(row["departure_time"])
                Eev = row["battery_capacity"]  # 现在是MWh
                Smax = row["soc_max"]
                Smin = row["soc_min"]
                
                # 处理夜间充电跨天的情况
                is_overnight = row["charging_type"] == "night" if "charging_type" in row else False
                
                if is_overnight:
                    # 当天晚上到达，第二天早上离开
                    actual_td = Td - 96  # 计算实际的第二天离开时间槽
                    # 容量预留约束，分两段
                    if Ta <= t <= T-2:  # 第一天晚上的部分
                        self.model.addConstr(
                            soc[scenario_idx, t, n] <= Smax - K_dn_values[t+1] * R_ev_dn_i[scenario_idx, t+1, n] / Eev
                        )
                        # 上调侧能量可行性：需要足够可放电余量
                        self.model.addConstr(
                            soc[scenario_idx, t, n] >= Smin + K_up_values[t+1] * R_ev_up_i[scenario_idx, t+1, n] / Eev
                        )
                    elif 0 <= t <= actual_td-1:  # 第二天早上的部分
                        self.model.addConstr(
                            soc[scenario_idx, t, n] <= Smax - K_dn_values[t+1] * R_ev_dn_i[scenario_idx, t+1, n] / Eev
                        )
                        self.model.addConstr(
                            soc[scenario_idx, t, n] >= Smin + K_up_values[t+1] * R_ev_up_i[scenario_idx, t+1, n] / Eev
                        )
                else:
                    # 常规情况
                    if Ta <= t < min(Td-1, T-1):
                        self.model.addConstr(
                            soc[scenario_idx, t, n] <= Smax - K_dn_values[t+1] * R_ev_dn_i[scenario_idx, t+1, n] / Eev
                        )
                        self.model.addConstr(
                            soc[scenario_idx, t, n] >= Smin + K_up_values[t+1] * R_ev_up_i[scenario_idx, t+1, n] / Eev
                        )


    def add_ev_fleet_aggregate_constraints(self, *,
                                        scenario_idx: int,
                                        T: int,
                                        P_ev_uc: dict,
                                        P_ev_cc: dict,
                                        P_ev0_uc: dict,
                                        P_ev0_cc: dict,
                                        P_ev_total: dict,
                                        P_ev0_total: dict,
                                        N_cc: int,
                                        N_uc: int) -> None:

        for t in range(T):
            # 实际充电功率约束
            self.model.addConstr(
                P_ev_total[scenario_idx, t] ==
                gp.quicksum(P_ev_uc[scenario_idx, t, n] for n in range(N_uc)) +
                gp.quicksum(P_ev_cc[scenario_idx, t, n] for n in range(N_cc))
            )
            
            # 投标总和等于全局投标
            self.model.addConstr(
                P_ev0_total[t] == 
                gp.quicksum(P_ev0_uc[scenario_idx, t, n] for n in range(N_uc)) +
                gp.quicksum(P_ev0_cc[scenario_idx, t, n] for n in range(N_cc))
            )


    def add_es1_constraints(self, *,
                       T: int,
                       scenario_idx: int,
                       P_es1: dict,
                       P_es1_ch: dict,  #Charging power of ES in hour t in scenario w (MW)
                       P_es1_dis: dict, #Discharging power of ES in hour t in scenario w (MW)
                       E_es1: dict,
                       y_es1: dict,
                       P_es0: dict,  # Energy bids (also PSP) of ES in hour t (MW)
                       P_es1_max: gp.Var,  #Maximum charging(discharging) power of ES1/ES2 (MW)
                       E_es1_max: gp.Var,  #Maximum energy stored in ES 1(MW)
                       E_es1_init: gp.Var,
                       eta_ch: float,
                       eta_dis: float,
                       delta_t: float
                       ) -> None:
        """
        添加ES1相关约束
        区别于ES2 ES1类似于EV 直接参与调频和DAM bid套利任务
        """
        for t in range(T):
            # ES bids上下限  PSP
            self.model.addConstr(P_es0[t] >= -P_es1_max)
            self.model.addConstr(P_es0[t] <= P_es1_max)

            # ES1 能量限制
            self.model.addConstr(P_es1[scenario_idx, t] == P_es0[t]) # 这里case2中 es1只负责套利 所以P_es1 = P_es0
            self.model.addConstr(E_es1[scenario_idx, t] >= 0)
            self.model.addConstr(E_es1[scenario_idx, t] <= E_es1_max)

            # ES1 拆分充放电&充放电互斥
            self.model.addConstr(P_es1_ch[scenario_idx, t] >= 0)
            self.model.addConstr(P_es1_dis[scenario_idx, t] >= 0)
            self.model.addConstr(P_es1_ch[scenario_idx, t] <= P_es1_max)
            self.model.addConstr(P_es1_dis[scenario_idx, t] <= P_es1_max)
            self.model.addConstr(P_es1[scenario_idx, t] == P_es1_ch[scenario_idx, t] - P_es1_dis[scenario_idx, t])
            self.model.addConstr(P_es1_ch[scenario_idx, t] <= y_es1[scenario_idx, t] * P_es1_max)
            self.model.addConstr(P_es1_dis[scenario_idx, t] <= (1 - y_es1[scenario_idx, t]) * P_es1_max)
            self.model.addConstr(y_es1[scenario_idx, t] + (1 - y_es1[scenario_idx, t]) <= 1)

            # ES1 电量演化
            if t == 0:
                self.model.addConstr(E_es1[scenario_idx, 0] == E_es1_init)
            else:
                self.model.addConstr(
                    E_es1[scenario_idx, t] ==
                    E_es1[scenario_idx, t - 1] +
                    (eta_ch * P_es1_ch[scenario_idx, t] -
                    P_es1_dis[scenario_idx, t] / eta_dis) * delta_t
                )


    def add_es2_constraints(self, *,
                       T: int,
                       scenario_idx: int,
                       P_es2: dict,
                       P_es2_ch: dict,
                       P_es2_dis: dict,
                       E_es2: dict,
                       y_es2: dict,
                       P_es2_max: gp.Var,
                       E_es2_max: gp.Var,
                       E_es2_init: gp.Var,
                       eta_ch: float,
                       eta_dis: float,
                       delta_t: float,
                       P_es2_ch_i: dict,
                       P_es2_dis_i: dict,
                       N_cc: int = 0
                       ) -> None:
            """
            添加ES2相关约束
            区别于ES1 ES2不直接参与调频 作为了EV调频任务的backup 也不会在dam bids
            """

            for t in range(T):
                # ES2 能量限制
                self.model.addConstr(E_es2[scenario_idx, t] >= 0)
                self.model.addConstr(E_es2[scenario_idx, t] <= E_es2_max)

                # ES2 拆分充放电&充放电互斥
                self.model.addConstr(P_es2_ch[scenario_idx, t] >= 0)
                self.model.addConstr(P_es2_dis[scenario_idx, t] >= 0)
                self.model.addConstr(P_es2[scenario_idx, t] == P_es2_ch[scenario_idx, t] - P_es2_dis[scenario_idx, t])
                self.model.addConstr(P_es2_ch[scenario_idx, t] <= y_es2[scenario_idx, t] * P_es2_max)
                self.model.addConstr(P_es2_dis[scenario_idx, t] <= (1 - y_es2[scenario_idx, t]) * P_es2_max)
                self.model.addConstr(y_es2[scenario_idx, t] + (1 - y_es2[scenario_idx, t]) <= 1)
                
                # 添加ES2总功率与分配给各EV的功率之间的关联约束
                self.model.addConstr(
                    P_es2_ch[scenario_idx, t] == gp.quicksum(P_es2_ch_i[scenario_idx, t, n] for n in range(N_cc))
                )
                self.model.addConstr(
                    P_es2_dis[scenario_idx, t] == gp.quicksum(P_es2_dis_i[scenario_idx, t, n] for n in range(N_cc))
                )
                
                # ES2 电量演化
                if t == 0:
                    self.model.addConstr(E_es2[scenario_idx, 0] == E_es2_init)
                else:
                    self.model.addConstr(
                        E_es2[scenario_idx, t] ==
                        E_es2[scenario_idx, t - 1] +
                        (eta_ch * P_es2_ch[scenario_idx, t] -
                        P_es2_dis[scenario_idx, t] / eta_dis) * delta_t
                    )


    def add_es_total_constraints(self, *,
                       T: int,
                       scenario_idx: int,
                       P_es_ch: dict,
                       P_es_dis: dict,
                       P_es1: dict,
                       P_es2: dict,
                       E_es1: dict,
                       E_es2: dict,
                       y_es: dict,
                       P_es1_max: gp.Var,
                       P_es2_max: gp.Var,
                       P_es_max: float,
                       E_es1_max: gp.Var,
                       E_es2_max: gp.Var,
                       E_es_max: float,
                       E_es1_init: float,
                       E_es2_init: float,
                       E_es_init: float,
                       dod: gp.Var,
                       kappa: float,
                       gamma: float 
                       ) -> None:
        """
        添加ES整体的相关约束
        """
        for t in range(T):
        # 总功率 = 充电 - 放电
            self.model.addConstr(P_es1[scenario_idx, t] + P_es2[scenario_idx, t] ==
                                P_es_ch[scenario_idx, t] - P_es_dis[scenario_idx, t])

            # 删除ES功率平衡约束，让ES1和ES2独立操作

            # ES总功率限制和充放电互斥
            self.model.addConstr(P_es_ch[scenario_idx, t] <= y_es[scenario_idx, t] * P_es_max)
            self.model.addConstr(P_es_dis[scenario_idx, t] <= (1 - y_es[scenario_idx, t]) * P_es_max)
            self.model.addConstr(y_es[scenario_idx, t] + (1 - y_es[scenario_idx, t]) <= 1)

        # DOD约束（最大允许放电深度）
            self.model.addConstr(E_es1[scenario_idx, t] +E_es2[scenario_idx, t] >= dod * E_es_max)
            self.model.addConstr(E_es1[scenario_idx, t] +E_es2[scenario_idx, t] <= E_es_max)

        # 最终电量边界（允许 ±gamma 偏移）
        self.model.addConstr(
            E_es1[scenario_idx, T-1] + E_es2[scenario_idx, T-1] >= (1 - gamma) * E_es_init
        )
        self.model.addConstr(
            E_es1[scenario_idx, T-1] + E_es2[scenario_idx, T-1] <= (1 + gamma) * E_es_init
        )
        self.model.addConstr(P_es_max == E_es_max * kappa)
        self.model.addConstr(E_es1_init + E_es2_init == E_es_init)
        self.model.addConstr(P_es1_max + P_es2_max == P_es_max)
        self.model.addConstr(E_es1_max + E_es2_max == E_es_max)
        self.model.addConstr(P_es1_max == E_es1_max * kappa)
        self.model.addConstr(P_es2_max == E_es2_max * kappa)


    def add_es2_backup_constraints(self, *,
                                        T: int,
                                        scenario_idx: int,
                                        P_ev0_total: dict,         
                                        R_ev_up: dict,       
                                        R_ev_dn: dict,       
                                        agc_up: pd.Series,       
                                        agc_dn: pd.Series,             
                                        P_es2_ch: dict,     
                                        P_es2_dis: dict,
                                        P_es2_ch_i: dict,     
                                        P_es2_dis_i: dict,
                                        P_ev_total: dict,
                                        N_cc: int
                                        ) -> None:

        for t in range(T):
            # ES2总功率约束
            self.model.addConstr(
                    P_es2_ch[scenario_idx, t] == gp.quicksum(P_es2_ch_i[scenario_idx, t, n] for n in range(N_cc))
                )
            self.model.addConstr(
                    P_es2_dis[scenario_idx, t] == gp.quicksum(P_es2_dis_i[scenario_idx, t, n] for n in range(N_cc))
                )
            
            # 简单的功率平衡约束：EV调频目标 = EV实际功率 + ES2净功率  
            # P_ev0_total - agc_up * R_ev_up + agc_dn * R_ev_dn = P_ev_total + P_es2_ch - P_es2_dis
            self.model.addConstr(
                P_ev0_total[t] - agc_up[t] * R_ev_up[t] + agc_dn[t] * R_ev_dn[t] == 
                P_ev_total[scenario_idx, t] + P_es2_ch[scenario_idx, t] - P_es2_dis[scenario_idx, t],
                name=f"power_balance_backup[{scenario_idx},{t}]"
            )

    def add_cvar_constraints(self, *,
                            num_scenarios: int,
                            sigma: gp.Var,  
                            phi: Dict[int, gp.Var], 
                            f: List[gp.LinExpr],
                            pi: float,
                            beta: float                
                            ) -> None:
        """
        添加CVaR约束
        
        Args:
            num_scenarios: 场景数量
            sigma: VaR变量σ（利润的风险值）
            phi: 辅助变量 φ_w 的字典
            f: 每个场景对应的收益表达式 f_w
            pi: 每个场景的概率
            beta: 置信水平 (0到1之间)
        """
        for w in range(num_scenarios):
            # φ_w ≥ 0
            self.model.addConstr(phi[w] >= 0, name=f"phi_positive[{w}]")
            # φ_w ≥ σ - f_w
            self.model.addConstr(phi[w] >= sigma - f[w], name=f"phi_def[{w}]")



    # def add_battery_degradation_constraints(self, *,
    #                                         d: gp.Var,      # DoD ∈ (0, 0.9]
    #                                         L: gp.Var,      # cycle life
    #                                         N0: float,
    #                                         beta: float
    #                                         ) -> None:
    #     """
    #     添加电池退化约束：L = N0 / d^beta
    #     即：d^beta * L = N0
    #     """

    #     # 创建中间变量：d^beta
    #     d_pow = self.model.addVar(name="d_pow")

    #     # 添加幂约束：d_pow = d^beta
    #     self.model.addGenConstrPow(d, d_pow, beta, name="d_pow_constr")

    #     # 添加主约束：d_pow * L = N0
    #     self.model.addQConstr(d_pow * L == N0, name="cycle_dod")




    def add_all_constraints(self) -> None:
        self.add_uc_ev_constraints()
        self.add_cc_ev_constraints()
        self.add_ev_fleet_aggregate_constraints()
        self.add_es1_constraints()
        self.add_es2_constraints()
        self.add_es_total_constraints()
        self.add_es2_backup_constraints()
        self.add_cvar_constraints()
        # self.add_battery_degradation_constraints()

class V2GConstraintsCase3:
    def __init__(self, model: gp.Model):
        """初始化约束管理器
        
        Args:
            model: Gurobi优化模型
        """
        self.model = model

    def add_uc_ev_constraints(self, *, 
                                ev_profiles: pd.DataFrame, 
                                T: int, 
                                delta_t: float,
                                P_ev_uc: dict, 
                                P_ev0_uc: dict,
                                scenario_idx: int, 
                                soc: dict,
                                N_uc: int) -> None:
        uc_evs = ev_profiles[ev_profiles["ev_type"] == "uc"].reset_index(drop=True)
        for n in range(N_uc):
            row = uc_evs.iloc[n]
            Ta_i = int(row["arrival_time"])
            Td_i = int(row["departure_time"])
            Sa_i = row["soc_arrival"]
            Sd_i = row["soc_departure"]
            Eev_i = row["battery_capacity"]
            Pmax = row["max_charge_power"]
            eta_ev = row["efficiency"]
            
            # 计算UC车辆的固定功率曲线 
            power_profile = np.zeros(T) # 初始化功率曲线全为0
            
            if Eev_i > 0 and Pmax > 0 and Sa_i < Sd_i:
                # 需要充入的总能量 (MWh)
                energy_needed = (Sd_i - Sa_i) * Eev_i
                # 每个满功率时段能充入的能量 (MWh)
                energy_per_slot = Pmax * eta_ev * delta_t                
                # 需要的总充电时段数（大概率是小数）
                duration_slots_float = energy_needed / energy_per_slot                
                # 满功率充电的时段数
                num_full_slots = int(duration_slots_float)               
                # 最后一个时段需要充的能量
                remaining_energy = energy_needed - num_full_slots * energy_per_slot                
                # 最后一个时段的精确功率 (MW)
                final_slot_power = remaining_energy / (eta_ev * delta_t)
                # 填充所有满功率时段
                for i in range(num_full_slots):
                    t = Ta_i + i
                    if t < Td_i: # 确保充电在离开前
                        power_profile[t] = Pmax
                
                # 填充最后一个部分功率时段
                t_final = Ta_i + num_full_slots
                if t_final < Td_i: # 确保在离开前
                    power_profile[t_final] = final_slot_power

            # 在模型中施加约束 
            for t in range(T):
                # 将预先计算好的刚性功率曲线作为约束
                self.model.addConstr(P_ev_uc[scenario_idx, t, n] == power_profile[t])
                # UC 车辆：投标等于其充电功率
                self.model.addConstr(P_ev0_uc[scenario_idx, t, n] == P_ev_uc[scenario_idx, t, n])

                # SoC 的演变现在依赖于这个固定的功率曲线
                if t < Ta_i:
                    self.model.addConstr(soc[scenario_idx, t, n] == 0)
                elif t == Ta_i:
                    self.model.addConstr(soc[scenario_idx, t, n] == Sa_i + (power_profile[t] * eta_ev * delta_t / Eev_i))
                else: # t > Ta_i
                    self.model.addConstr(soc[scenario_idx, t, n] == soc[scenario_idx, t - 1, n] + (power_profile[t] * eta_ev * delta_t / Eev_i))

            # 添加最终SoC约束作为验证
            final_t = min(Td_i - 1, T - 1)
            if final_t >= Ta_i:
                # 使用小的容差避免浮点数精度问题
                self.model.addConstr(soc[scenario_idx, final_t, n] >= Sd_i - 1e-6)

    def add_cc_ev_constraints(self, *, 
                            ev_profiles: pd.DataFrame,
                            delta_t: float,
                            T: int,
                            scenario_idx: int,
                            P_ev_cc: dict,
                            soc: dict,
                            P_ev_cc_ch: dict,
                            P_ev_cc_dis: dict,
                            P_ev_cc_ch_eff: dict,
                            P_ev_cc_dis_eff: dict,
                            y_ev: dict,                            # <--- 替换 mu_ev_ch/dis
                            P_ev0_cc: dict,
                            R_ev_up_i: dict,
                            R_ev_dn_i: dict,
                            R_ev_up: dict,
                            R_ev_dn: dict,
                            K_up_values: np.ndarray,
                            K_dn_values: np.ndarray,
                            N_cc: int) -> None:
        cc_evs = ev_profiles[ev_profiles["ev_type"] == "cc"].reset_index(drop=True)

        for t in range(T):
            self.model.addConstr(
                R_ev_up[t] == gp.quicksum(R_ev_up_i[scenario_idx, t, n] for n in range(N_cc))
            )
            self.model.addConstr(
                R_ev_dn[t] == gp.quicksum(R_ev_dn_i[scenario_idx, t, n] for n in range(N_cc))
            )
            for n in range(N_cc):
                row = cc_evs.iloc[n]
                Ta = int(row["arrival_time"])
                Td = int(row["departure_time"])
                Sa = row["soc_arrival"]
                Sd = row["soc_departure"]
                Smax = row["soc_max"]
                Smin = row["soc_min"]
                Eev = row["battery_capacity"]
                Pmax = row["max_charge_power"]
                eta = row["efficiency"]

                is_overnight = row["charging_type"] == "night" if "charging_type" in row else False

                if is_overnight:
                    actual_td = Td - 96
                    is_charging_time = (t >= Ta) or (t <= actual_td)
                else:
                    is_charging_time = (Ta <= t < Td)

                if is_charging_time:
                    self.model.addConstr(P_ev0_cc[scenario_idx, t, n] >= -Pmax)
                    self.model.addConstr(P_ev0_cc[scenario_idx, t, n] <= Pmax)
                    self.model.addConstr(R_ev_up_i[scenario_idx, t, n] <= Pmax + P_ev0_cc[scenario_idx, t, n])
                    self.model.addConstr(R_ev_up_i[scenario_idx, t, n] >= 0)
                    self.model.addConstr(R_ev_dn_i[scenario_idx, t, n] >= 0)
                    self.model.addConstr(R_ev_dn_i[scenario_idx, t, n] <= Pmax - P_ev0_cc[scenario_idx, t, n])
                    self.model.addConstr(P_ev_cc[scenario_idx, t, n] >= -Pmax)
                    self.model.addConstr(P_ev_cc[scenario_idx, t, n] <= Pmax)

                    # 充放分解 + 单二进制互斥
                    self.model.addConstr(P_ev_cc_ch[scenario_idx, t, n] >= 0)
                    self.model.addConstr(P_ev_cc_dis[scenario_idx, t, n] >= 0)
                    self.model.addConstr(P_ev_cc[scenario_idx, t, n] == P_ev_cc_ch[scenario_idx, t, n] - P_ev_cc_dis[scenario_idx, t, n])
                    self.model.addConstr(P_ev_cc_ch[scenario_idx, t, n] <= y_ev[scenario_idx, t, n] * Pmax)
                    self.model.addConstr(P_ev_cc_dis[scenario_idx, t, n] <= (1 - y_ev[scenario_idx, t, n]) * Pmax)

                    # PWL 效率
                    x_breaks_ch = [0.0, 4e-3, Pmax]
                    y_breaks_ch = [0.0, 0.5 * 4e-3, 0.5 * 4e-3 + 0.95 * (Pmax - 4e-3)]
                    self.model.addGenConstrPWL(P_ev_cc_ch[scenario_idx, t, n], P_ev_cc_ch_eff[scenario_idx, t, n], x_breaks_ch, y_breaks_ch)

                    x_breaks_dis = [0.0, 4e-3, Pmax]
                    y_breaks_dis = [0.0, 0.5 * 4e-3, 0.5 * 4e-3 + (1.0 / 0.95) * (Pmax - 4e-3)]
                    self.model.addGenConstrPWL(P_ev_cc_dis[scenario_idx, t, n], P_ev_cc_dis_eff[scenario_idx, t, n], x_breaks_dis, y_breaks_dis)

                    # SOC 动力学（与你原逻辑一致）
                    if is_overnight:
                        actual_td = Td - 96
                        if t == Ta:
                            self.model.addConstr(
                                soc[scenario_idx, t, n] == Sa +
                                ((P_ev_cc_ch_eff[scenario_idx, t, n] * eta - P_ev_cc_dis_eff[scenario_idx, t, n] * 1/eta) * delta_t / Eev)
                            )
                        if t > Ta or t <= actual_td:
                            prev_t = (T - 1) if t == 0 else (t - 1)
                            self.model.addConstr(
                                soc[scenario_idx, t, n] == soc[scenario_idx, prev_t, n] +
                                ((P_ev_cc_ch_eff[scenario_idx, t, n] * eta - P_ev_cc_dis_eff[scenario_idx, t, n] * 1/eta) * delta_t / Eev)
                            )
                        if t == actual_td - 1:
                            self.model.addConstr(soc[scenario_idx, t, n] >= Sd)
                            self.model.addConstr(soc[scenario_idx, t, n] <= Smax)
                    else:
                        if t == Ta:
                            self.model.addConstr(
                                soc[scenario_idx, t, n] == Sa + 
                                ((P_ev_cc_ch_eff[scenario_idx, t, n] * eta - P_ev_cc_dis_eff[scenario_idx, t, n] * 1/eta) * delta_t / Eev)
                            )
                        elif t > Ta:
                            self.model.addConstr(
                                soc[scenario_idx, t, n] == soc[scenario_idx, t-1, n] +
                                ((P_ev_cc_ch_eff[scenario_idx, t, n] * eta - P_ev_cc_dis_eff[scenario_idx, t, n] * 1/eta) * delta_t / Eev)
                            )
                        if t == Td - 1:
                            self.model.addConstr(soc[scenario_idx, t, n] >= Sd)
                            self.model.addConstr(soc[scenario_idx, t, n] <= Smax)

                    self.model.addConstr(soc[scenario_idx, t, n] >= Smin)
                    self.model.addConstr(soc[scenario_idx, t, n] <= Smax)

                else:
                    self.model.addConstr(R_ev_up_i[scenario_idx, t, n] == 0)
                    self.model.addConstr(R_ev_dn_i[scenario_idx, t, n] == 0)
                    self.model.addConstr(P_ev0_cc[scenario_idx, t, n] == 0)
                    self.model.addConstr(P_ev_cc[scenario_idx, t, n] == 0)
                    self.model.addConstr(P_ev_cc_ch[scenario_idx, t, n] == 0)
                    self.model.addConstr(P_ev_cc_dis[scenario_idx, t, n] == 0)
                    self.model.addConstr(y_ev[scenario_idx, t, n] == 0)   # <--- 新增：离桩时刻禁止充放
            
            #容量预留逻辑       
            for n in range(N_cc):
                row = cc_evs.iloc[n]
                Ta = int(row["arrival_time"])
                Td = int(row["departure_time"])
                Eev = row["battery_capacity"]  # MWh
                Smax = row["soc_max"]
                Smin = row["soc_min"]

                is_overnight = row["charging_type"] == "night" if "charging_type" in row else False

                for t in range(T):
                    # 容量预留约束只对“连桩期间且 t+1 还在时间轴内”的时段有效
                    if t + 1 >= T:
                        continue

                    if is_overnight:
                        # 夜间跨天：当天晚上到，第二天早上走
                        actual_td = Td - 96  # 第二天离开时间槽
                        # 第一晚：Ta ~ T-2
                        if Ta <= t <= T-2:
                            self.model.addConstr(
                                soc[scenario_idx, t, n] <= Smax - K_dn_values[t+1] * R_ev_dn_i[scenario_idx, t+1, n] / Eev
                            )
                            self.model.addConstr(
                                soc[scenario_idx, t, n] >= Smin + K_up_values[t+1] * R_ev_up_i[scenario_idx, t+1, n] / Eev
                            )
                        # 第二天早上：0 ~ actual_td-1
                        elif 0 <= t <= actual_td - 1:
                            self.model.addConstr(
                                soc[scenario_idx, t, n] <= Smax - K_dn_values[t+1] * R_ev_dn_i[scenario_idx, t+1, n] / Eev
                            )
                            self.model.addConstr(
                                soc[scenario_idx, t, n] >= Smin + K_up_values[t+1] * R_ev_up_i[scenario_idx, t+1, n] / Eev
                            )
                    else:
                        # 普通白天充电：Ta ~ Td-1 之间需要预留容量
                        if Ta <= t < min(Td - 1, T - 1):
                            self.model.addConstr(
                                soc[scenario_idx, t, n] <= Smax - K_dn_values[t+1] * R_ev_dn_i[scenario_idx, t+1, n] / Eev
                            )
                            self.model.addConstr(
                                soc[scenario_idx, t, n] >= Smin + K_up_values[t+1] * R_ev_up_i[scenario_idx, t+1, n] / Eev
                            )


    def add_ev_fleet_aggregate_constraints(self, *,
                                        scenario_idx: int,
                                        T: int,
                                        P_ev_uc: dict,
                                        P_ev_cc: dict,
                                        P_ev0_uc: dict,
                                        P_ev0_cc: dict,
                                        P_ev_total: dict,
                                        P_ev0_total: dict,
                                        N_cc: int,
                                        N_uc: int,
                                        R_ev_dn_i: dict,
                                        R_ev_up_i: dict,
                                        P_ev0_cc_total: dict,
                                        R_ev_dn_cc_total: dict,
                                        R_ev_up_cc_total: dict) -> None:

        for t in range(T):
            # 实际充电功率约束
            self.model.addConstr(
                P_ev_total[scenario_idx, t] ==
                gp.quicksum(P_ev_uc[scenario_idx, t, n] for n in range(N_uc)) +
                gp.quicksum(P_ev_cc[scenario_idx, t, n] for n in range(N_cc))    #在cc约束里已限定cc功率等于充电功率减去放电功率
            )
            
            # 投标总和等于全局投标
            self.model.addConstr(
                P_ev0_total[t] == 
                gp.quicksum(P_ev0_uc[scenario_idx, t, n] for n in range(N_uc)) +
                gp.quicksum(P_ev0_cc[scenario_idx, t, n] for n in range(N_cc))
            )

            self.model.addConstr(
                P_ev0_cc_total[scenario_idx, t] ==
                gp.quicksum(P_ev0_cc[scenario_idx, t, n] for n in range(N_cc))
            )
            self.model.addConstr(
                R_ev_dn_cc_total[scenario_idx, t] ==
                gp.quicksum(R_ev_dn_i[scenario_idx, t, n] for n in range(N_cc))
            )
            self.model.addConstr(
                R_ev_up_cc_total[scenario_idx, t] ==
                gp.quicksum(R_ev_up_i[scenario_idx, t, n] for n in range(N_cc))
            )



    def add_es1_constraints(self, *,
                        T: int,
                        scenario_idx: int,
                        P_es1: dict,
                        P_es1_ch: dict,
                        P_es1_dis: dict,
                        E_es1: dict,
                        y_es1: dict,                 # <--- 替换 mu_es1_ch/dis
                        R_es_up: dict,
                        R_es_dn: dict,
                        P_es0: dict,
                        agc_up: pd.Series,
                        agc_dn: pd.Series,
                        K_up_values: np.ndarray,
                        K_dn_values: np.ndarray,
                        P_es1_max: gp.Var,
                        E_es1_max: gp.Var,
                        E_es1_init: gp.Var,
                        eta_ch: float,
                        eta_dis: float,
                        delta_t: float,
                        ) -> None:

        for t in range(T):
            self.model.addConstr(P_es0[t] >= -P_es1_max)
            self.model.addConstr(P_es0[t] <= P_es1_max)

            self.model.addConstr(R_es_up[t] >= 0)
            self.model.addConstr(R_es_up[t] <= P_es0[t] + P_es1_max)
            self.model.addConstr(R_es_dn[t] >= 0)
            self.model.addConstr(R_es_dn[t] <= P_es1_max - P_es0[t])

            self.model.addConstr(
                P_es1[scenario_idx, t] ==
                P_es0[t] - R_es_up[t] * agc_up[t] + R_es_dn[t] * agc_dn[t]
            )

            # 单二进制互斥
            self.model.addConstr(P_es1[scenario_idx, t] == P_es1_ch[scenario_idx, t] - P_es1_dis[scenario_idx, t])
            self.model.addConstr(P_es1_ch[scenario_idx, t] <= y_es1[scenario_idx, t] * P_es1_max)
            self.model.addConstr(P_es1_dis[scenario_idx, t] <= (1 - y_es1[scenario_idx, t]) * P_es1_max)

            # SOC
            if t == 0:
                self.model.addConstr(E_es1[scenario_idx, 0] == E_es1_init)
            else:
                self.model.addConstr(
                    E_es1[scenario_idx, t] ==
                    E_es1[scenario_idx, t - 1] +
                    (eta_ch * P_es1_ch[scenario_idx, t] -
                    P_es1_dis[scenario_idx, t] / eta_dis) * delta_t
                )

            # （如需）容量预留能量约束
            if t + 1 < T:
                self.model.addConstr(
                    K_up_values[t+1] * R_es_up[t+1] <= E_es1[scenario_idx, t]
                )
                self.model.addConstr(
                    E_es1[scenario_idx, t] <= E_es1_max - K_dn_values[t+1] * R_es_dn[t+1]
                )

                

    def add_es2_constraints(self, *,
                       T: int,
                       scenario_idx: int,
                       P_es2: dict,
                       P_es2_ch: dict,
                       P_es2_dis: dict,
                       E_es2: dict,
                       y_es2: dict,                 # <--- 替换 mu_es2_ch/dis
                       P_es2_max: gp.Var,
                       E_es2_max: gp.Var,
                       E_es2_init: gp.Var,
                       eta_ch: float,
                       eta_dis: float,
                       delta_t: float,
                       P_es2_ch_i: dict = None,
                       P_es2_dis_i: dict = None,
                       N_cc: int = 0
                       ) -> None:

        for t in range(T):
            self.model.addConstr(E_es2[scenario_idx, t] >= 0)
            self.model.addConstr(E_es2[scenario_idx, t] <= E_es2_max)

            self.model.addConstr(P_es2_ch[scenario_idx, t] >= 0)
            self.model.addConstr(P_es2_dis[scenario_idx, t] >= 0)
            self.model.addConstr(P_es2[scenario_idx, t] == P_es2_ch[scenario_idx, t] - P_es2_dis[scenario_idx, t])
            self.model.addConstr(P_es2_ch[scenario_idx, t] <= y_es2[scenario_idx, t] * P_es2_max)
            self.model.addConstr(P_es2_dis[scenario_idx, t] <= (1 - y_es2[scenario_idx, t]) * P_es2_max)

            # 若你暂不做改 C，则保留逐车分配的聚合关系
            self.model.addConstr(
                P_es2_ch[scenario_idx, t] == gp.quicksum(P_es2_ch_i[scenario_idx, t, n] for n in range(N_cc))
            )
            self.model.addConstr(
                P_es2_dis[scenario_idx, t] == gp.quicksum(P_es2_dis_i[scenario_idx, t, n] for n in range(N_cc))
            )

            if t == 0:
                self.model.addConstr(E_es2[scenario_idx, 0] == E_es2_init)
            else:
                self.model.addConstr(
                    E_es2[scenario_idx, t] ==
                    E_es2[scenario_idx, t - 1] +
                    (eta_ch * P_es2_ch[scenario_idx, t] -
                     P_es2_dis[scenario_idx, t] / eta_dis) * delta_t
                )

    def add_es_total_constraints(self, *,
                        T: int,
                        scenario_idx: int,
                        P_es_ch: dict,
                        P_es_dis: dict,
                        P_es1: dict,
                        P_es2: dict,
                        E_es1: dict,
                        E_es2: dict,
                        y_es: dict,                 # <--- 替换 mu_es_ch/dis
                        P_es1_max: gp.Var,
                        P_es2_max: gp.Var,
                        P_es_max: float,
                        E_es1_max: gp.Var,
                        E_es2_max: gp.Var,
                        E_es_max: float,
                        E_es1_init: float,
                        E_es2_init: float,
                        E_es_init: float,
                        dod: gp.Var,
                        kappa: float,
                        gamma: float 
                        ) -> None:

        for t in range(T):
            self.model.addConstr(P_es1[scenario_idx, t] + P_es2[scenario_idx, t] ==
                                P_es_ch[scenario_idx, t] - P_es_dis[scenario_idx, t])

            # 单二进制互斥（总 ES 层）
            self.model.addConstr(P_es_ch[scenario_idx, t] <= y_es[scenario_idx, t] * P_es_max)
            self.model.addConstr(P_es_dis[scenario_idx, t] <= (1 - y_es[scenario_idx, t]) * P_es_max)

            # DOD + 总能量上界
            self.model.addConstr(E_es1[scenario_idx, t] + E_es2[scenario_idx, t] >= dod * E_es_max)
            self.model.addConstr(E_es1[scenario_idx, t] + E_es2[scenario_idx, t] <= E_es_max)

        self.model.addConstr(E_es1[scenario_idx, T-1] + E_es2[scenario_idx, T-1] >= (1 - gamma) * E_es_init)
        self.model.addConstr(E_es1[scenario_idx, T-1] + E_es2[scenario_idx, T-1] <= (1 + gamma) * E_es_init)
        self.model.addConstr(P_es1_max + P_es2_max == P_es_max)
        self.model.addConstr(E_es1_max + E_es2_max == E_es_max)
        self.model.addConstr(P_es1_max == E_es1_max * kappa)
        self.model.addConstr(P_es2_max == E_es2_max * kappa)
        self.model.addConstr(P_es_max == E_es_max * kappa)
        self.model.addConstr(E_es1_init + E_es2_init == E_es_init)



    def add_es2_backup_constraints(self, *,
                                        T: int,
                                        scenario_idx: int,
                                        P_ev0_total: dict,         
                                        R_ev_up: dict,       
                                        R_ev_dn: dict,       
                                        agc_up: pd.Series,       
                                        agc_dn: pd.Series,             
                                        P_es2_ch: dict,     
                                        P_es2_dis: dict,
                                        P_es2_ch_i: dict,     
                                        P_es2_dis_i: dict,
                                        P_ev_total: dict,
                                        N_cc: int
                                        ) -> None:

        for t in range(T):
            # ES2总功率约束
            self.model.addConstr(
                    P_es2_ch[scenario_idx, t] == gp.quicksum(P_es2_ch_i[scenario_idx, t, n] for n in range(N_cc))
                )
            self.model.addConstr(
                    P_es2_dis[scenario_idx, t] == gp.quicksum(P_es2_dis_i[scenario_idx, t, n] for n in range(N_cc))
                )
            
            # 简单的功率平衡约束：EV调频目标 = EV实际功率 + ES2净功率  
            # P_ev0_total - agc_up * R_ev_up + agc_dn * R_ev_dn = P_ev_total + P_es2_ch - P_es2_dis
            self.model.addConstr(
                P_ev0_total[t] - agc_up[t] * R_ev_up[t] + agc_dn[t] * R_ev_dn[t] == 
                P_ev_total[scenario_idx, t] + P_es2_ch[scenario_idx, t] - P_es2_dis[scenario_idx, t],
                name=f"power_balance_backup[{scenario_idx},{t}]"
            )

    def add_cvar_constraints(self, *,
                            num_scenarios: int,
                            sigma: gp.Var,  
                            phi: Dict[int, gp.Var], 
                            f: List[gp.LinExpr],
                            pi: float,
                            beta: float                
                            ) -> None:
        """
        添加CVaR约束
        
        Args:
            num_scenarios: 场景数量
            sigma: VaR变量σ（利润的风险值）
            phi: 辅助变量 φ_w 的字典
            f: 每个场景对应的收益表达式 f_w
            pi: 每个场景的概率
            beta: 置信水平 (0到1之间)
        """
        for w in range(num_scenarios):
            # φ_w ≥ 0
            self.model.addConstr(phi[w] >= 0, name=f"phi_positive[{w}]")
            # φ_w ≥ σ - f_w
            self.model.addConstr(phi[w] >= sigma - f[w], name=f"phi_def[{w}]")




    # def add_battery_degradation_constraints(self, *,
    #                                         d: gp.Var,      # DoD ∈ (0, 0.9]
    #                                         L: gp.Var,      # cycle life
    #                                         N0: float,
    #                                         beta: float
    #                                         ) -> None:
    #     """
    #     添加电池退化约束：L = N0 / d^beta
    #     即：d^beta * L = N0
    #     """

    #     # 创建中间变量：d^beta
    #     d_pow = self.model.addVar(name="d_pow")

    #     # 添加幂约束：d_pow = d^beta
    #     self.model.addGenConstrPow(d, d_pow, beta, name="d_pow_constr")

    #     # 添加主约束：d_pow * L = N0
    #     self.model.addQConstr(d_pow * L == N0, name="cycle_dod")


    def add_all_constraints(self) -> None:
        self.add_uc_ev_constraints()
        self.add_cc_ev_constraints()
        self.add_ev_fleet_aggregate_constraints()
        self.add_es1_constraints()
        self.add_es2_constraints()
        self.add_es_total_constraints()
        self.add_es2_backup_constraints()
        self.add_cvar_constraints()
        # self.add_battery_degradation_constraints()

class V2GConstraintsCase4:
    def __init__(self, model: gp.Model):
        """初始化约束管理器
        
        Args:
            model: Gurobi优化模型
        """
        self.model = model

    def add_uc_ev_constraints(self, *, 
                                ev_profiles: pd.DataFrame, 
                                T: int, 
                                delta_t: float,
                                P_ev_uc: dict, 
                                P_ev0_uc: dict,
                                scenario_idx: int, 
                                soc: dict,
                                N_uc: int) -> None:
        uc_evs = ev_profiles[ev_profiles["ev_type"] == "uc"].reset_index(drop=True)
        for n in range(N_uc):
            row = uc_evs.iloc[n]
            Ta_i = int(row["arrival_time"])
            Td_i = int(row["departure_time"])
            Sa_i = row["soc_arrival"]
            Sd_i = row["soc_departure"]
            Eev_i = row["battery_capacity"]
            Pmax = row["max_charge_power"]
            eta_ev = row["efficiency"]
            
            # 计算UC车辆的固定功率曲线 
            power_profile = np.zeros(T) # 初始化功率曲线全为0
            
            if Eev_i > 0 and Pmax > 0 and Sa_i < Sd_i:
                # 需要充入的总能量 (MWh)
                energy_needed = (Sd_i - Sa_i) * Eev_i
                # 每个满功率时段能充入的能量 (MWh)
                energy_per_slot = Pmax * eta_ev * delta_t                
                # 需要的总充电时段数（大概率是小数）
                duration_slots_float = energy_needed / energy_per_slot                
                # 满功率充电的时段数
                num_full_slots = int(duration_slots_float)               
                # 最后一个时段需要充的能量
                remaining_energy = energy_needed - num_full_slots * energy_per_slot                
                # 最后一个时段的精确功率 (MW)
                final_slot_power = remaining_energy / (eta_ev * delta_t)
                # 填充所有满功率时段
                for i in range(num_full_slots):
                    t = Ta_i + i
                    if t < Td_i: # 确保充电在离开前
                        power_profile[t] = Pmax
                
                # 填充最后一个部分功率时段
                t_final = Ta_i + num_full_slots
                if t_final < Td_i: # 确保在离开前
                    power_profile[t_final] = final_slot_power

            # 在模型中施加约束 
            for t in range(T):
                # 将预先计算好的刚性功率曲线作为约束
                self.model.addConstr(P_ev_uc[scenario_idx, t, n] == power_profile[t])
                # UC 车辆：投标等于其充电功率
                self.model.addConstr(P_ev0_uc[scenario_idx, t, n] == P_ev_uc[scenario_idx, t, n])

                # SoC 的演变现在依赖于这个固定的功率曲线
                if t < Ta_i:
                    self.model.addConstr(soc[scenario_idx, t, n] == 0)
                elif t == Ta_i:
                    self.model.addConstr(soc[scenario_idx, t, n] == Sa_i + (power_profile[t] * eta_ev * delta_t / Eev_i))
                else: # t > Ta_i
                    self.model.addConstr(soc[scenario_idx, t, n] == soc[scenario_idx, t - 1, n] + (power_profile[t] * eta_ev * delta_t / Eev_i))

            # 添加最终SoC约束作为验证
            final_t = min(Td_i - 1, T - 1)
            if final_t >= Ta_i:
                # 使用小的容差避免浮点数精度问题
                self.model.addConstr(soc[scenario_idx, final_t, n] >= Sd_i - 1e-6)


    def add_cc_ev_constraints(self, *, 
                           ev_profiles: pd.DataFrame,
                           delta_t: float,
                           T: int,
                           scenario_idx: int,
                           P_ev_cc: dict,
                           soc: dict,
                           P_ev_cc_ch: dict,
                           P_ev_cc_dis: dict,
                           P_ev_cc_ch_eff: dict,
                           P_ev_cc_dis_eff: dict,
                           y_ev: dict,                           
                           P_ev0_cc: dict,
                           R_ev_up_i: dict,
                           R_ev_dn_i: dict,
                           R_ev_up: dict,  #不依赖场景
                           R_ev_dn: dict,  #不依赖场景
                           K_up_values: np.ndarray,  # 每个时段的上调频容量预留值
                           K_dn_values: np.ndarray,  # 每个时段的下调频容量预留值
                           N_cc: int) -> None:
        """添加可控EV约束"""
        cc_evs = ev_profiles[ev_profiles["ev_type"] == "cc"].reset_index(drop=True)
        
        for t in range(T):
            self.model.addConstr(
                R_ev_up[t] == gp.quicksum(R_ev_up_i[scenario_idx, t, n] for n in range(N_cc))
            )
            self.model.addConstr(
                R_ev_dn[t] == gp.quicksum(R_ev_dn_i[scenario_idx, t, n] for n in range(N_cc))
            )
            for n in range(N_cc):
                row = cc_evs.iloc[n]
                Ta = int(row["arrival_time"])
                Td = int(row["departure_time"])
                Sa = row["soc_arrival"]
                Sd = row["soc_departure"]
                Smax = row["soc_max"]
                Smin = row["soc_min"]
                Eev = row["battery_capacity"]  # 现在是MWh
                Pmax = row["max_charge_power"]  # 现在是MW
                eta = row["efficiency"]
                
                # 处理夜间充电跨天的情况
                is_overnight = row["charging_type"] == "night" if "charging_type" in row else False
                
                # 夜间充电EV：
                # 当天晚上到达，第二天早上离开 (Td > 95)
                if is_overnight:
                    actual_td = Td - 96  # 计算实际的第二天离开时间槽
                    is_charging_time = (t >= Ta) or (t <= actual_td) 
                else:
                    # 处理白天充电（左闭右开）
                    is_charging_time = (Ta <= t < Td)
                    
                if is_charging_time:
                    # Baseline and net power are symmetric for V2G
                    self.model.addConstr(P_ev0_cc[scenario_idx, t, n] >= -Pmax)
                    self.model.addConstr(P_ev0_cc[scenario_idx, t, n] <= Pmax)
                    # Up capacity limited by headroom to discharge side
                    self.model.addConstr(R_ev_up_i[scenario_idx, t, n] <= Pmax + P_ev0_cc[scenario_idx, t, n])
                    self.model.addConstr(R_ev_up_i[scenario_idx, t, n] >= 0)
                    self.model.addConstr(R_ev_dn_i[scenario_idx, t, n] >= 0)
                    self.model.addConstr(R_ev_dn_i[scenario_idx, t, n] <= Pmax - P_ev0_cc[scenario_idx, t, n])
                    self.model.addConstr(P_ev_cc[scenario_idx, t, n] >= -Pmax)
                    self.model.addConstr(P_ev_cc[scenario_idx, t, n] <= Pmax)
                    # Charge/discharge components
                    self.model.addConstr(P_ev_cc_ch[scenario_idx, t, n] >= 0)
                    self.model.addConstr(P_ev_cc_dis[scenario_idx, t, n] >= 0)
                    self.model.addConstr(P_ev_cc[scenario_idx, t, n] == P_ev_cc_ch[scenario_idx, t, n] - P_ev_cc_dis[scenario_idx, t, n])
                    # 最大功率和互斥充放电逻辑（逐车逐时段）
                    self.model.addConstr(P_ev_cc_ch[scenario_idx, t, n] <= y_ev[scenario_idx, t, n] * Pmax)
                    self.model.addConstr(P_ev_cc_dis[scenario_idx, t, n] <= (1 - y_ev[scenario_idx, t, n]) * Pmax)
                    self.model.addConstr(y_ev[scenario_idx, t, n] + (1 - y_ev[scenario_idx, t, n]) <= 1)

                    # 使用PWL表达效率阈值（4kW=0.004MW）
                    # 充电：小于4kW效率0.5，≥4kW效率0.95
                    x_breaks_ch = [0.0, 4e-3, Pmax]
                    y_breaks_ch = [0.0, 0.5 * 4e-3, 0.5 * 4e-3 + 0.95 * (Pmax - 4e-3)]
                    self.model.addGenConstrPWL(P_ev_cc_ch[scenario_idx, t, n], P_ev_cc_ch_eff[scenario_idx, t, n], x_breaks_ch, y_breaks_ch)

                    # 放电：小于4kW效率0.5，≥4kW效率0.95（等效为能量消耗系数 1/效率）
                    x_breaks_dis = [0.0, 4e-3, Pmax]
                    y_breaks_dis = [0.0, 0.5 * 4e-3, 0.5 * 4e-3 + (1.0 / 0.95) * (Pmax - 4e-3)]
                    self.model.addGenConstrPWL(P_ev_cc_dis[scenario_idx, t, n], P_ev_cc_dis_eff[scenario_idx, t, n], x_breaks_dis, y_breaks_dis)

                    # 修改SOC计算逻辑，处理跨天充电情况
                    if is_overnight:
                        actual_td = Td - 96
                        if t == Ta:  # 初始时刻
                            self.model.addConstr(
                                soc[scenario_idx, t, n] == Sa +
                                ((P_ev_cc_ch_eff[scenario_idx, t, n] * eta - P_ev_cc_dis_eff[scenario_idx, t, n] * 1/eta) * delta_t / Eev)
                            )
                        if t > Ta or t <= actual_td:  # 过程
                            if t == 0:  # 新的一天开始
                                prev_t = T - 1  # 使用前一天的最后时刻
                            else:
                                prev_t = t - 1
                                
                            self.model.addConstr(
                                soc[scenario_idx, t, n] == soc[scenario_idx, prev_t, n] +
                                ((P_ev_cc_ch_eff[scenario_idx, t, n] * eta - P_ev_cc_dis_eff[scenario_idx, t, n] * 1/eta) * delta_t / Eev)
                            )
                            
                        # 离开时SOC要求
                        if t == actual_td - 1:
                            self.model.addConstr(soc[scenario_idx, t, n] >= Sd)
                            self.model.addConstr(soc[scenario_idx, t, n] <= Smax)
                    else:  # 常规情况（不跨天）
                        if t == Ta:  # 刚到达
                            self.model.addConstr(
                                soc[scenario_idx, t, n] == Sa + 
                                ((P_ev_cc_ch_eff[scenario_idx, t, n] * eta - P_ev_cc_dis_eff[scenario_idx, t, n] * 1/eta) * delta_t / Eev)
                            )
                        elif t > Ta:  # 过程
                            self.model.addConstr(
                                soc[scenario_idx, t, n] == soc[scenario_idx, t-1, n] +
                                ((P_ev_cc_ch_eff[scenario_idx, t, n] * eta - P_ev_cc_dis_eff[scenario_idx, t, n] * 1/eta) * delta_t / Eev)
                            )
                        
                        # 离开时SOC要求
                        if t == Td - 1:
                            self.model.addConstr(soc[scenario_idx, t, n] >= Sd)
                            self.model.addConstr(soc[scenario_idx, t, n] <= Smax)
                   
                    # 确保所有时刻的SOC都在合理范围内
                    self.model.addConstr(soc[scenario_idx, t, n] >= Smin)
                    self.model.addConstr(soc[scenario_idx, t, n] <= Smax)
                    
                else:
                    self.model.addConstr(R_ev_up_i[scenario_idx, t, n] == 0)
                    self.model.addConstr(R_ev_dn_i[scenario_idx, t, n] == 0)
                    self.model.addConstr(P_ev0_cc[scenario_idx, t, n] == 0)
                    self.model.addConstr(P_ev_cc[scenario_idx, t, n] == 0)
                    self.model.addConstr(P_ev_cc_ch[scenario_idx, t, n] == 0)
                    self.model.addConstr(P_ev_cc_dis[scenario_idx, t, n] == 0)
                    self.model.addConstr(y_ev[scenario_idx, t, n] == 0)

        #容量预留逻辑        
            # for n in range(N_cc): #这里再循环一次是因为时间不同
            #     row = cc_evs.iloc[n]
            #     Ta = int(row["arrival_time"])
            #     Td = int(row["departure_time"])
            #     Eev = row["battery_capacity"]  # 现在是MWh
            #     Smax = row["soc_max"]
            #     Smin = row["soc_min"]
                
            #     # 处理夜间充电跨天的情况
            #     is_overnight = row["charging_type"] == "night" if "charging_type" in row else False
                
            #     if is_overnight:
            #         # 当天晚上到达，第二天早上离开
            #         actual_td = Td - 96  # 计算实际的第二天离开时间槽
            #         # 容量预留约束，分两段
            #         if Ta <= t <= T-2:  # 第一天晚上的部分
            #             self.model.addConstr(
            #                 soc[scenario_idx, t, n] <= Smax - K_dn_values[t+1] * R_ev_dn_i[scenario_idx, t+1, n] / Eev
            #             )
            #             # 上调侧能量可行性：需要足够可放电余量
            #             self.model.addConstr(
            #                 soc[scenario_idx, t, n] >= Smin + K_up_values[t+1] * R_ev_up_i[scenario_idx, t+1, n] / Eev
            #             )
            #         elif 0 <= t <= actual_td-1:  # 第二天早上的部分
            #             self.model.addConstr(
            #                 soc[scenario_idx, t, n] <= Smax - K_dn_values[t+1] * R_ev_dn_i[scenario_idx, t+1, n] / Eev
            #             )
            #             self.model.addConstr(
            #                 soc[scenario_idx, t, n] >= Smin + K_up_values[t+1] * R_ev_up_i[scenario_idx, t+1, n] / Eev
            #             )
            #     else:
            #         # 常规情况
            #         if Ta <= t < min(Td-1, T-1):
            #             self.model.addConstr(
            #                 soc[scenario_idx, t, n] <= Smax - K_dn_values[t+1] * R_ev_dn_i[scenario_idx, t+1, n] / Eev
            #             )
            #             self.model.addConstr(
            #                 soc[scenario_idx, t, n] >= Smin + K_up_values[t+1] * R_ev_up_i[scenario_idx, t+1, n] / Eev
            #             )
                


    def add_ev_fleet_aggregate_constraints(self, *,
                                        scenario_idx: int,
                                        T: int,
                                        P_ev_uc: dict,
                                        P_ev_cc: dict,
                                        P_ev0_uc: dict,
                                        P_ev0_cc: dict,
                                        P_ev_total: dict,
                                        P_ev0_total: dict,
                                        N_cc: int,
                                        N_uc: int,
                                        R_ev_dn_i: dict,
                                        R_ev_up_i: dict,
                                        P_ev0_cc_total: dict,
                                        R_ev_dn_cc_total: dict,
                                        R_ev_up_cc_total: dict) -> None:

        for t in range(T):
            # 实际充电功率约束
            self.model.addConstr(
                P_ev_total[scenario_idx, t] ==
                gp.quicksum(P_ev_uc[scenario_idx, t, n] for n in range(N_uc)) +
                gp.quicksum(P_ev_cc[scenario_idx, t, n] for n in range(N_cc))    #在cc约束里已限定cc功率等于充电功率减去放电功率
            )
            
            # 投标总和等于全局投标
            self.model.addConstr(
                P_ev0_total[t] == 
                gp.quicksum(P_ev0_uc[scenario_idx, t, n] for n in range(N_uc)) +
                gp.quicksum(P_ev0_cc[scenario_idx, t, n] for n in range(N_cc))
            )

            self.model.addConstr(
                P_ev0_cc_total[scenario_idx, t] ==
                gp.quicksum(P_ev0_cc[scenario_idx, t, n] for n in range(N_cc))
            )
            self.model.addConstr(
                R_ev_dn_cc_total[scenario_idx, t] ==
                gp.quicksum(R_ev_dn_i[scenario_idx, t, n] for n in range(N_cc))
            )
            self.model.addConstr(
                R_ev_up_cc_total[scenario_idx, t] ==
                gp.quicksum(R_ev_up_i[scenario_idx, t, n] for n in range(N_cc))
            )

    def add_es1_constraints(self, *,
                       T: int,
                       scenario_idx: int,
                       P_es1: dict,
                       P_es1_ch: dict,  #Charging power of ES in hour t in scenario w (MW)
                       P_es1_dis: dict, #Discharging power of ES in hour t in scenario w (MW)
                       E_es1: dict,
                       y_es1: dict,
                       R_es_up: dict, #Regulation up capacity bids of ES in hour t (MW)
                       R_es_dn: dict, #Regulation down capacity bids of ES in hour t (MW)
                       P_es0: dict,  # Energy bids (also PSP) of ES in hour t (MW)
                       agc_up: pd.Series,
                       agc_dn: pd.Series,
                       K_up_values: np.ndarray,  # 每个时段的上调频容量预留值
                       K_dn_values: np.ndarray,  # 每个时段的下调频容量预留值
                       P_es1_max: gp.Var,  #Maximum charging(discharging) power of ES1/ES2 (MW)
                       E_es1_max: gp.Var,  #Maximum energy stored in ES 1(MW)
                       E_es1_init: gp.Var,
                       eta_ch: float,
                       eta_dis: float,
                       delta_t: float,
                       ) -> None:
        """
        添加ES1相关约束
        区别于ES2 ES1类似于EV 直接参与调频任务 参与DAM套利
        """
        for t in range(T):
            # ES bids上下限  PSP
            self.model.addConstr(P_es0[t] >= -P_es1_max)
            self.model.addConstr(P_es0[t] <= P_es1_max)

            # 调频边界（基于 PSP 偏移量）
            self.model.addConstr(R_es_up[t] >= 0)
            self.model.addConstr(R_es_up[t] <= P_es0[t] + P_es1_max)
            self.model.addConstr(R_es_dn[t] >= 0)
            self.model.addConstr(R_es_dn[t] <= P_es1_max - P_es0[t])

            # ES1 实际功率等于 PSP + 调频响应
            self.model.addConstr(
                P_es1[scenario_idx, t] ==
                P_es0[t] - R_es_up[t] * agc_up[t] + R_es_dn[t] * agc_dn[t]
            )   #es1负责直接参与调频   所以  所有的之前上报的bids会在这里用掉一些
            

            # ES1 拆分充放电&充放电互斥
            self.model.addConstr(P_es1[scenario_idx, t] == P_es1_ch[scenario_idx, t] - P_es1_dis[scenario_idx, t])
            self.model.addConstr(P_es1_ch[scenario_idx, t] <= y_es1[scenario_idx, t] * P_es1_max)
            self.model.addConstr(P_es1_dis[scenario_idx, t] <= (1 - y_es1[scenario_idx, t]) * P_es1_max)
            self.model.addConstr(y_es1[scenario_idx, t] + (1 - y_es1[scenario_idx, t]) <= 1)


            # ES1 电量演化
            if t == 0:
                self.model.addConstr(E_es1[scenario_idx, 0] == E_es1_init)
            else:
                self.model.addConstr(
                    E_es1[scenario_idx, t] ==
                    E_es1[scenario_idx, t - 1] +
                    (eta_ch * P_es1_ch[scenario_idx, t] -
                    P_es1_dis[scenario_idx, t] / eta_dis) * delta_t
                )

            #容量预留逻辑
            # 动态 SOC 上限（考虑未来调频预留容量）case2这里没有用到 因为case2的ES1不参与调频
            # if t + 1 < T:
            #     self.model.addConstr(
            #         K_up_values[t+1] * R_es_up[t+1] <= E_es1[scenario_idx, t]
            #     )
            #     self.model.addConstr(
            #         E_es1[scenario_idx, t] <= E_es1_max - K_dn_values[t+1] * R_es_dn[t+1]
            #     )

    def add_es2_constraints(self, *,
                       T: int,
                       scenario_idx: int,
                       P_es2: dict,
                       P_es2_ch: dict,
                       P_es2_dis: dict,
                       E_es2: dict,
                       y_es2: dict,
                       P_es2_max: gp.Var,
                       E_es2_max: gp.Var,
                       E_es2_init: gp.Var,
                       eta_ch: float,
                       eta_dis: float,
                       delta_t: float,
                       P_es2_ch_i: dict = None,
                       P_es2_dis_i: dict = None,
                       N_cc: int = 0
                       ) -> None:
            """
            添加ES2相关约束
            区别于ES1 ES2不直接参与调频 作为了EV调频任务的backup 也不会在dam bids
            """

            for t in range(T):
                # ES2 能量限制
                self.model.addConstr(E_es2[scenario_idx, t] >= 0)
                self.model.addConstr(E_es2[scenario_idx, t] <= E_es2_max)

                # ES2 拆分充放电&充放电互斥
                self.model.addConstr(P_es2_ch[scenario_idx, t] >= 0)
                self.model.addConstr(P_es2_dis[scenario_idx, t] >= 0)
                self.model.addConstr(P_es2[scenario_idx, t] == P_es2_ch[scenario_idx, t] - P_es2_dis[scenario_idx, t])
                self.model.addConstr(P_es2_ch[scenario_idx, t] <= y_es2[scenario_idx, t] * P_es2_max)
                self.model.addConstr(P_es2_dis[scenario_idx, t] <= (1 - y_es2[scenario_idx, t]) * P_es2_max)
                self.model.addConstr(y_es2[scenario_idx, t] + (1 - y_es2[scenario_idx, t]) <= 1)
                
                # 添加ES2总功率与分配给各EV的功率之间的关联约束
                self.model.addConstr(
                    P_es2_ch[scenario_idx, t] == gp.quicksum(P_es2_ch_i[scenario_idx, t, n] for n in range(N_cc))
                )
                self.model.addConstr(
                    P_es2_dis[scenario_idx, t] == gp.quicksum(P_es2_dis_i[scenario_idx, t, n] for n in range(N_cc))
                )
                
                # ES2 电量演化
                if t == 0:
                    self.model.addConstr(E_es2[scenario_idx, 0] == E_es2_init)
                else:
                    self.model.addConstr(
                        E_es2[scenario_idx, t] ==
                        E_es2[scenario_idx, t - 1] +
                        (eta_ch * P_es2_ch[scenario_idx, t] -
                        P_es2_dis[scenario_idx, t] / eta_dis) * delta_t
                    )


    def add_es_total_constraints(self, *,
                       T: int,
                       scenario_idx: int,
                       P_es_ch: dict,
                       P_es_dis: dict,
                       P_es1: dict,
                       P_es2: dict,
                       E_es1: dict,
                       E_es2: dict,
                       y_es: dict,
                       P_es1_max: gp.Var,
                       P_es2_max: gp.Var,
                       P_es_max: float,
                       E_es1_max: gp.Var,
                       E_es2_max: gp.Var,
                       E_es_max: float,
                       E_es1_init: float,
                       E_es2_init: float,
                       E_es_init: float,
                       dod: gp.Var,
                       kappa: float,
                       gamma: float 
                       ) -> None:
        """
        添加ES整体的相关约束
        """
        for t in range(T):
        # 总功率 = 充电 - 放电
            self.model.addConstr(P_es1[scenario_idx, t] + P_es2[scenario_idx, t] ==
                                P_es_ch[scenario_idx, t] - P_es_dis[scenario_idx, t])

            # 最大功率和互斥充放电逻辑
            self.model.addConstr(P_es_ch[scenario_idx, t] <= y_es[scenario_idx, t] * P_es_max)
            self.model.addConstr(P_es_dis[scenario_idx, t] <= (1 - y_es[scenario_idx, t]) * P_es_max)
            self.model.addConstr(y_es[scenario_idx, t] + (1 - y_es[scenario_idx, t]) <= 1)

        # DOD约束（最大允许放电深度）
            self.model.addConstr(E_es1[scenario_idx, t] +E_es2[scenario_idx, t] >= dod * E_es_max)
            self.model.addConstr(E_es1[scenario_idx, t] +E_es2[scenario_idx, t] <= E_es_max)

        # 最终电量边界（允许 ±gamma 偏移）
        self.model.addConstr(
            E_es1[scenario_idx, T-1] + E_es2[scenario_idx, T-1] >= (1 - gamma) * E_es_init
        )
        self.model.addConstr(
            E_es1[scenario_idx, T-1] + E_es2[scenario_idx, T-1] <= (1 + gamma) * E_es_init
        )
        self.model.addConstr(P_es1_max + P_es2_max == P_es_max)
        self.model.addConstr(E_es1_max + E_es2_max == E_es_max)
        self.model.addConstr(P_es1_max == E_es1_max * kappa)
        self.model.addConstr(P_es2_max == E_es2_max * kappa)
        self.model.addConstr(P_es_max == E_es_max * kappa)
        self.model.addConstr(E_es1_init + E_es2_init == E_es_init)

    def add_es2_backup_constraints(self, *,
                                        T: int,
                                        scenario_idx: int,
                                        P_ev0_total: dict,         
                                        R_ev_up: dict,       
                                        R_ev_dn: dict,       
                                        agc_up: pd.Series,       
                                        agc_dn: pd.Series,             
                                        P_es2_ch: dict,     
                                        P_es2_dis: dict,
                                        P_es2_ch_i: dict,     
                                        P_es2_dis_i: dict,
                                        P_ev_total: dict,
                                        N_cc: int
                                        ) -> None:

        for t in range(T):
            # ES2总功率约束
            self.model.addConstr(
                    P_es2_ch[scenario_idx, t] == gp.quicksum(P_es2_ch_i[scenario_idx, t, n] for n in range(N_cc))
                )
            self.model.addConstr(
                    P_es2_dis[scenario_idx, t] == gp.quicksum(P_es2_dis_i[scenario_idx, t, n] for n in range(N_cc))
                )
            
            # 简单的功率平衡约束：EV调频目标 = EV实际功率 + ES2净功率  
            # P_ev0_total - agc_up * R_ev_up + agc_dn * R_ev_dn = P_ev_total + P_es2_ch - P_es2_dis
            self.model.addConstr(
                P_ev0_total[t] - agc_up[t] * R_ev_up[t] + agc_dn[t] * R_ev_dn[t] == 
                P_ev_total[scenario_idx, t] + P_es2_ch[scenario_idx, t] - P_es2_dis[scenario_idx, t],
                name=f"power_balance_backup[{scenario_idx},{t}]"
            )

    def add_cvar_constraints(self, *,
                            num_scenarios: int,
                            sigma: gp.Var,  
                            phi: Dict[int, gp.Var], 
                            f: List[gp.LinExpr],
                            pi: float,
                            beta: float                
                            ) -> None:
        """
        添加CVaR约束
        
        Args:
            num_scenarios: 场景数量
            sigma: VaR变量σ（利润的风险值）
            phi: 辅助变量 φ_w 的字典
            f: 每个场景对应的收益表达式 f_w
            pi: 每个场景的概率
            beta: 置信水平 (0到1之间)
        """
        for w in range(num_scenarios):
            # φ_w ≥ 0
            self.model.addConstr(phi[w] >= 0, name=f"phi_positive[{w}]")
            # φ_w ≥ σ - f_w
            self.model.addConstr(phi[w] >= sigma - f[w], name=f"phi_def[{w}]")




    # def add_battery_degradation_constraints(self, *,
    #                                         d: gp.Var,      # DoD ∈ (0, 0.9]
    #                                         L: gp.Var,      # cycle life
    #                                         N0: float,
    #                                         beta: float
    #                                         ) -> None:
    #     """
    #     添加电池退化约束：L = N0 / d^beta
    #     即：d^beta * L = N0
    #     """

    #     # 创建中间变量：d^beta
    #     d_pow = self.model.addVar(name="d_pow")

    #     # 添加幂约束：d_pow = d^beta
    #     self.model.addGenConstrPow(d, d_pow, beta, name="d_pow_constr")

    #     # 添加主约束：d_pow * L = N0
    #     self.model.addQConstr(d_pow * L == N0, name="cycle_dod")


    def add_all_constraints(self) -> None:
        self.add_uc_ev_constraints()
        self.add_cc_ev_constraints()
        self.add_ev_fleet_aggregate_constraints()
        self.add_es1_constraints()
        self.add_es2_constraints()
        self.add_es_total_constraints()
        self.add_es2_backup_constraints()
        self.add_cvar_constraints()
        # self.add_battery_degradation_constraints()
    