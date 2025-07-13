import gurobipy as gp
import numpy as np
import pandas as pd


class V2GConstraintsCase1:
    def __init__(self, model: gp.Model):
        """初始化约束管理器
        
        Args:
            model: Gurobi优化模型
        """
        self.model = model

    def add_uc_ev_constraints(self, *, 
                              ev_profiles: pd.DataFrame, 
                              T: int, 
                              P_ev_uc: dict, 
                              P_ev0_uc: dict,
                              scenario_idx: int, 
                              N_uc: int) -> None:
        uc_evs = ev_profiles[ev_profiles["ev_type"] == "uc"].reset_index(drop=True)
        for n in range(N_uc):
            row = uc_evs.iloc[n]
            Ta_i = row["arrival_time"]
            Sa_i = row["soc_arrival"]
            Sd_i = row["soc_departure"]
            Eev_i = row["battery_capacity"]
            Pmax = row["max_charge_power"]
            eta_ev = row["efficiency"]
            
            # 计算 Tk_i,w
            Tk_i = Ta_i + ((Sd_i - Sa_i) * Eev_i) / (Pmax * eta_ev)
            
            for t in range(T):
                if Ta_i <= t <= Tk_i:
                    self.model.addConstr(P_ev_uc[scenario_idx, t, n] == Pmax)
                    self.model.addConstr(P_ev0_uc[scenario_idx, t, n] >= 0)
                    self.model.addConstr(P_ev0_uc[scenario_idx, t, n] <= Pmax)
                else:
                    self.model.addConstr(P_ev_uc[scenario_idx, t, n] == 0)
                    self.model.addConstr(P_ev0_uc[scenario_idx, t, n] == 0)

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
                               K_dn: float,
                               N_cc: int) -> None:
        
        cc_evs = ev_profiles[ev_profiles["ev_type"] == "cc"].reset_index(drop=True)
        
        for t in range(T):
            self.model.addConstr(
                R_ev_up[scenario_idx, t] == gp.quicksum(R_ev_up_i[scenario_idx, t, n] for n in range(N_cc))
            )
            self.model.addConstr(
                R_ev_dn[scenario_idx, t] == gp.quicksum(R_ev_dn_i[scenario_idx, t, n] for n in range(N_cc))
            )
            for n in range(N_cc):
                row = cc_evs.iloc[n]
                Ta = int(row["arrival_time"])
                Td = int(row["departure_time"])
                Sa = row["soc_arrival"]
                Sd = row["soc_departure"]
                Smax = row["soc_max"]
                # Smin = row["soc_min"]
                Eev = row["battery_capacity"]
                Pmax = row["max_charge_power"]
                eta = row["efficiency"]
                if Ta <= t <= Td:
                    self.model.addConstr(P_ev0_cc[scenario_idx, t, n] >= 0)
                    self.model.addConstr(P_ev0_cc[scenario_idx, t, n] <= Pmax)
                    self.model.addConstr(R_ev_up_i[scenario_idx, t, n] >= 0)
                    self.model.addConstr(R_ev_up_i[scenario_idx, t, n] <= P_ev0_cc[scenario_idx, t, n])
                    self.model.addConstr(R_ev_dn_i[scenario_idx, t, n] >= 0)
                    self.model.addConstr(R_ev_dn_i[scenario_idx, t, n] <= Pmax - P_ev0_cc[scenario_idx, t, n])
                    self.model.addConstr(P_ev_cc[scenario_idx, t, n] >= 0)
                    self.model.addConstr(P_ev_cc[scenario_idx, t, n] <= Pmax)
                    self.model.addConstr(
                    soc[scenario_idx, t, n] == Sa +
                    gp.quicksum(P_ev_cc[scenario_idx, tau, n] * eta * delta_t / Eev for tau in range(Ta, t+1))
                )   
                    self.model.addConstr(soc[scenario_idx, Td, n] >= Sd)
                    self.model.addConstr(soc[scenario_idx, Td, n] <= Smax)
                    # self.model.addConstr(soc[scenario_idx, Td, n] >= Smin)
                else:
                    self.model.addConstr(R_ev_up_i[scenario_idx, t, n] == 0)
                    self.model.addConstr(R_ev_dn_i[scenario_idx, t, n] == 0)
                    self.model.addConstr(P_ev0_cc[scenario_idx, t, n] == 0)
                    self.model.addConstr(P_ev_cc[scenario_idx, t, n] == 0)
                
            for n in range(N_cc): #这里再循环一次是因为时间不同
                row = cc_evs.iloc[n]
                Ta = int(row["arrival_time"])
                Td = int(row["departure_time"])
                Eev = row["battery_capacity"]
                for t in range(Ta, min(Td-1,T-1)):
                    self.model.addConstr(
                    soc[scenario_idx, t, n] <= Smax - K_dn * R_ev_dn_i[scenario_idx, t+1, n] / Eev
                    ) # K_dn为常数，直接使用，不用时间下标 原论文用了 但我觉得即使是带下标 按论文的意思也是一个常数罢了 时时刻刻都一样

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
            self.model.addConstr(
                P_ev_total[scenario_idx, t] ==
                gp.quicksum(P_ev_uc[scenario_idx, t, n] for n in range(N_uc)) +
                gp.quicksum(P_ev_cc[scenario_idx, t, n] for n in range(N_cc))
            )
            self.model.addConstr(
                P_ev0_total[scenario_idx, t] ==
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
                       mu_es_ch: dict,
                       mu_es_dis: dict,
                       P_es_max: float,
                       E_es_max: float,
                       E_es_init: float,
                       eta_ch: float,
                       eta_dis: float,
                       delta_t: float,                                            
                       dod: gp.Var,
                       gamma: float 
                       ) -> None:
            """
            添加ES2相关约束
            区别于ES1 ES2不直接参与调频 作为了EV调频任务的backup 不会在dam bids
            """

            for t in range(T):
                # ES 能量限制  DOD约束（最大允许放电深度）
                self.model.addConstr(E_es[scenario_idx, t] >= 0)
                self.model.addConstr(E_es[scenario_idx, t] >= dod*E_es_max)
                self.model.addConstr(E_es[scenario_idx, t] <= E_es_max)

                # ES 拆分充放电&充放电互斥
                self.model.addConstr(P_es[scenario_idx, t] == P_es_ch[scenario_idx, t] - P_es_dis[scenario_idx, t])
                self.model.addConstr(P_es_ch[scenario_idx, t] <= mu_es_ch[scenario_idx, t] * P_es_max)
                self.model.addConstr(P_es_dis[scenario_idx, t] <= mu_es_dis[scenario_idx, t] * P_es_max)
                self.model.addConstr(mu_es_ch[scenario_idx, t] + mu_es_dis[scenario_idx, t] <= 1)
                
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
                E_es[scenario_idx, T-1] >= (1 - gamma) * E_es_init
            )
            self.model.addConstr(
                E_es[scenario_idx, T-1] <= (1 + gamma) * E_es_init
            )

    def add_es_backup_constraints(self, *,
                                        T: int,
                                        scenario_idx: int,
                                        P_ev_total: dict,         
                                        R_ev_up: dict,       
                                        R_ev_dn: dict,       
                                        agc_up: pd.Series,       
                                        agc_dn: pd.Series,       
                                        P_ev_uc: dict,       
                                        P_ev_cc: dict,       
                                        P_es_ch: dict,     
                                        P_es_dis: dict,
                                        N_cc: int,
                                        N_uc: int
                                        ) -> None:

        for t in range(T):
            lhs = (P_ev_total[scenario_idx, t]
                - agc_up[t] * R_ev_up[scenario_idx, t]
                + agc_dn[t] * R_ev_dn[scenario_idx, t])
            rhs = (
                gp.quicksum(P_ev_uc[scenario_idx, t, n] for n in range(N_uc)) +
                gp.quicksum(P_ev_cc[scenario_idx, t, n] for n in range(N_cc)) +
                P_es_ch[scenario_idx, t] - P_es_dis[scenario_idx, t]
            )  #这里建立了ES2 backup的逻辑
            self.model.addConstr(lhs == rhs)


    # def add_cvar_constraints(self, *,
    #                         num_scenarios: int,
    #                         sigma: gp.Var,  
    #                         phi: Dict[int, gp.Var], 
    #                         f: List[gp.LinExpr]               
    #                         ) -> None:
    #     """
    #     添加CVaR约束

    #         num_scenarios: 场景数量
    #         sigma: VaR变量σ 利润的风险值
    #         phi: 辅助变量 φ_w 的字典
    #         f: 每个场景对应的损失或负收益表达式 f_w
    #     """
    #     for w in range(num_scenarios):
    #         # φ_w ≥ σ - f_w
    #         self.model.addConstr(phi[w] >= sigma - f[w])
    #         # φ_w ≥ 0
    #         self.model.addConstr(phi[w] >= 0)


    def add_battery_degradation_constraints(self, *,
                                        d: gp.Var,      # DoD ∈ (0,0.9]
                                        L: gp.Var,      # cycle life
                                        N0: float = 5000,
                                        beta: float = 0.5
                                        ) -> None:

        # 范围约束
        self.model.addConstr(d >= 1e-4, name="dod_lb")
        self.model.addConstr(d <= 0.9,  name="dod_ub")

        # log 变量
        log_d = self.model.addVar(name="log_d")
        log_L = self.model.addVar(name="log_L")

        # Gurobi 自带对数锥生成
        self.model.addGenConstrLog(d, log_d, name="log_d_constr")
        self.model.addGenConstrLog(L, log_L, name="log_L_constr")

        # ln L + beta * ln d = ln N0
        self.model.addConstr(
            log_L + beta * log_d == np.log(N0),
            name="cycle_life_dod_relation"
        )



    def add_all_constraints(self) -> None:
        self.add_uc_ev_constraints()
        self.add_cc_ev_constraints()
        self.add_ev_fleet_aggregate_constraints()
        self.add_es_constraints()
        self.add_es_backup_constraints()
    #    self.add_cvar_constraints()
    #    self.add_battery_degradation_constraints()


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
                              P_ev_uc: dict, 
                              P_ev0_uc: dict,
                              scenario_idx: int, 
                              N_uc: int) -> None:
        """添加 uc ev 相关约束"""
        uc_evs = ev_profiles[ev_profiles["ev_type"] == "uc"].reset_index(drop=True)
        for n in range(N_uc):
            row = uc_evs.iloc[n]
            Ta_i = row["arrival_time"]
            Sa_i = row["soc_arrival"]
            Sd_i = row["soc_departure"]
            Eev_i = row["battery_capacity"]
            Pmax = row["max_charge_power"]
            eta_ev = row["efficiency"]
            
            # 计算 Tk_i,w
            Tk_i = Ta_i + ((Sd_i - Sa_i) * Eev_i) / (Pmax * eta_ev)
            
            for t in range(T):
                if Ta_i <= t <= Tk_i:
                    self.model.addConstr(P_ev_uc[scenario_idx, t, n] == Pmax)
                    self.model.addConstr(P_ev0_uc[scenario_idx, t, n] >= 0)
                    self.model.addConstr(P_ev0_uc[scenario_idx, t, n] <= Pmax)
                else:
                    self.model.addConstr(P_ev_uc[scenario_idx, t, n] == 0)
                    self.model.addConstr(P_ev0_uc[scenario_idx, t, n] == 0)

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
                               K_dn: float,
                               N_cc: int) -> None:
        cc_evs = ev_profiles[ev_profiles["ev_type"] == "cc"].reset_index(drop=True)
        
        for t in range(T):
            self.model.addConstr(
                R_ev_up[scenario_idx, t] == gp.quicksum(R_ev_up_i[scenario_idx, t, n] for n in range(N_cc))
            )
            self.model.addConstr(
                R_ev_dn[scenario_idx, t] == gp.quicksum(R_ev_dn_i[scenario_idx, t, n] for n in range(N_cc))
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
                if Ta <= t <= Td:
                    self.model.addConstr(P_ev0_cc[scenario_idx, t, n] >= 0)
                    self.model.addConstr(P_ev0_cc[scenario_idx, t, n] <= Pmax)
                    self.model.addConstr(R_ev_up_i[scenario_idx, t, n] >= 0)
                    self.model.addConstr(R_ev_up_i[scenario_idx, t, n] <= P_ev0_cc[scenario_idx, t, n])
                    self.model.addConstr(R_ev_dn_i[scenario_idx, t, n] >= 0)
                    self.model.addConstr(R_ev_dn_i[scenario_idx, t, n] <= Pmax - P_ev0_cc[scenario_idx, t, n])
                    self.model.addConstr(P_ev_cc[scenario_idx, t, n] >= 0)
                    self.model.addConstr(P_ev_cc[scenario_idx, t, n] <= Pmax)
                    self.model.addConstr(
                    soc[scenario_idx, t, n] == Sa +
                    gp.quicksum(P_ev_cc[scenario_idx, tau, n] * eta * delta_t / Eev for tau in range(Ta, t+1))
                )   
                    self.model.addConstr(soc[scenario_idx, Td, n] >= Sd)
                    self.model.addConstr(soc[scenario_idx, Td, n] <= Smax)
                    self.model.addConstr(soc[scenario_idx, Td, n] >= Smin)
                else:
                    self.model.addConstr(R_ev_up_i[scenario_idx, t, n] == 0)
                    self.model.addConstr(R_ev_dn_i[scenario_idx, t, n] == 0)
                    self.model.addConstr(P_ev0_cc[scenario_idx, t, n] == 0)
                    self.model.addConstr(P_ev_cc[scenario_idx, t, n] == 0)
                
            for n in range(N_cc): #这里再循环一次是因为时间不同
                row = cc_evs.iloc[n]
                Ta = int(row["arrival_time"])
                Td = int(row["departure_time"])
                Eev = row["battery_capacity"]
                for t in range(Ta, Td-1):
                    self.model.addConstr(
                        soc[scenario_idx, t, n] <= Smax - K_dn * R_ev_dn_i[scenario_idx, t+1, n] / Eev
                    ) # K_dn为常数，直接使用，不用时间下标 原论文用了 但我觉得即使是带下标 按论文的意思也是一个常数罢了 时时刻刻都一样

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
            self.model.addConstr(
                P_ev_total[scenario_idx, t] ==
                gp.quicksum(P_ev_uc[scenario_idx, t, n] for n in range(N_uc)) +
                gp.quicksum(P_ev_cc[scenario_idx, t, n] for n in range(N_cc))
            )
            self.model.addConstr(
                P_ev0_total[scenario_idx, t] ==
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
                       mu_es1_ch: dict,
                       mu_es1_dis: dict,
                       R_es_up: dict, #Regulation up capacity bids of ES in hour t (MW)
                       R_es_dn: dict, #Regulation down capacity bids of ES in hour t (MW)
                       P_es0: dict,  # Energy bids (also PSP) of ES in hour t (MW)
                       agc_up: pd.Series,
                       agc_dn: pd.Series,
                       K_up: float,
                       K_dn: float,
                       P_es1_max: gp.Var,  #Maximum charging(discharging) power of ES1/ES2 (MW)
                       E_es1_max: gp.Var,  #Maximum energy stored in ES 1(MW)
                       E_es1_init: float,
                       eta_ch: float,
                       eta_dis: float,
                       delta_t: float,
                       ) -> None:
        """
        添加ES1相关约束
        区别于ES2 ES1类似于EV 直接参与调频任务
        """
        for t in range(T):
            # ES bids上下限  PSP
            self.model.addConstr(P_es0[scenario_idx, t] >= -P_es1_max)
            self.model.addConstr(P_es0[scenario_idx, t] <= P_es1_max)

            # 调频边界（基于 PSP 偏移量）
            self.model.addConstr(R_es_up[scenario_idx, t] >= 0)
            self.model.addConstr(R_es_up[scenario_idx, t] <= P_es0[scenario_idx, t] + P_es1_max)
            self.model.addConstr(R_es_dn[scenario_idx, t] >= 0)
            self.model.addConstr(R_es_dn[scenario_idx, t] <= P_es1_max - P_es0[scenario_idx, t])

            # ES1 实际功率等于 PSP + 调频响应
            self.model.addConstr(
                P_es1[scenario_idx, t] ==
                P_es0[scenario_idx, t] - R_es_up[scenario_idx, t] * agc_up[t] + R_es_dn[scenario_idx, t] * agc_dn[t]
            )   #es1负责直接参与调频   所以  所有的之前上报的bids都在这里用掉 
                       
            # ES1 拆分充放电&充放电互斥
            self.model.addConstr(P_es1[scenario_idx, t] == P_es1_ch[scenario_idx, t] - P_es1_dis[scenario_idx, t])
            self.model.addConstr(P_es1_ch[scenario_idx, t] <= mu_es1_ch[scenario_idx, t] * P_es1_max)
            self.model.addConstr(P_es1_dis[scenario_idx, t] <= mu_es1_dis[scenario_idx, t] * P_es1_max)
            self.model.addConstr(mu_es1_ch[scenario_idx, t] + mu_es1_dis[scenario_idx, t] <= 1)


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

            # 动态 SOC 上限（考虑未来调频预留容量）
            if t + 1 < T:
                self.model.addConstr(
                    K_up * R_es_up[scenario_idx, t+1] <= E_es1[scenario_idx, t]
                )
                self.model.addConstr(
                    E_es1[scenario_idx, t] <= E_es1_max - K_dn * R_es_dn[scenario_idx, t+1]
            )

    def add_es2_constraints(self, *,
                       T: int,
                       scenario_idx: int,
                       P_es2: dict,
                       P_es2_ch: dict,
                       P_es2_dis: dict,
                       E_es2: dict,
                       mu_es2_ch: dict,
                       mu_es2_dis: dict,
                    #    R_es2_up: dict,
                    #    R_es2_dn: dict,
                    #    K_up: float,
                    #    K_dn: float,
                       P_es2_max: gp.Var,
                       E_es2_max: gp.Var,
                       E_es2_init: float,
                       eta_ch: float,
                       eta_dis: float,
                       delta_t: float
                       ) -> None:
            """
            添加ES2相关约束
            区别于ES1 ES2不直接参与调频 作为了EV调频任务的backup 不会在dam bids
            """

            for t in range(T):
                # ES2 能量限制
                self.model.addConstr(E_es2[scenario_idx, t] >= 0)
                self.model.addConstr(E_es2[scenario_idx, t] <= E_es2_max)

                # ES2 拆分充放电&充放电互斥
                self.model.addConstr(P_es2[scenario_idx, t] == P_es2_ch[scenario_idx, t] - P_es2_dis[scenario_idx, t])
                self.model.addConstr(P_es2_ch[scenario_idx, t] <= mu_es2_ch[scenario_idx, t] * P_es2_max)
                self.model.addConstr(P_es2_dis[scenario_idx, t] <= mu_es2_dis[scenario_idx, t] * P_es2_max)
                self.model.addConstr(mu_es2_ch[scenario_idx, t] + mu_es2_dis[scenario_idx, t] <= 1)
                
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

                # # 动态 SOC 上限（考虑未来调频预留容量）
                # if t + 1 < T:
                #     self.model.addConstr(
                #         K_up * R_es2_up[scenario_idx, t+1] <= E_es2[scenario_idx, t]
                #     )
                #     self.model.addConstr(
                #         E_es2[scenario_idx, t] <= E_es2_max - K_dn * R_es2_dn[scenario_idx, t+1]
                # )


    def add_es_total_constraints(self, *,
                       T: int,
                       scenario_idx: int,
                       P_es_ch: dict,
                       P_es_dis: dict,
                       P_es1: dict,
                       P_es2: dict,
                       E_es1: dict,
                       E_es2: dict,
                       mu_es_ch: dict,
                       mu_es_dis: dict,
                       P_es1_max: gp.Var,
                       P_es2_max: gp.Var,
                       P_es_max: float,
                       E_es1_max: gp.Var,
                       E_es2_max: gp.Var,
                       E_es_max: float,
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
            self.model.addConstr(P_es_ch[scenario_idx, t] <= mu_es_ch[scenario_idx, t] * P_es_max)
            self.model.addConstr(P_es_dis[scenario_idx, t] <= mu_es_dis[scenario_idx, t] * P_es_max)
            self.model.addConstr(mu_es_ch[scenario_idx, t] + mu_es_dis[scenario_idx, t] <= 1)

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

    def add_es2_backup_constraints(self, *,
                                        T: int,
                                        scenario_idx: int,
                                        P_ev_total: dict,         
                                        R_ev_up: dict,       
                                        R_ev_dn: dict,       
                                        agc_up: pd.Series,       
                                        agc_dn: pd.Series,       
                                        P_ev_uc: dict,       
                                        P_ev_cc: dict,       
                                        P_es2_ch: dict,     
                                        P_es2_dis: dict,
                                        N_cc: int,
                                        N_uc: int
                                        ) -> None:

        for t in range(T):
            lhs = (P_ev_total[scenario_idx, t]
                - agc_up[t] * R_ev_up[scenario_idx, t]
                + agc_dn[t] * R_ev_dn[scenario_idx, t])
            rhs = (
                gp.quicksum(P_ev_uc[scenario_idx, t, n] for n in range(N_uc)) +
                gp.quicksum(P_ev_cc[scenario_idx, t, n] for n in range(N_cc)) +
                P_es2_ch[scenario_idx, t] - P_es2_dis[scenario_idx, t]
            )  #区别于fllet自己的约束  这里建立了ES2 backup的逻辑
            self.model.addConstr(lhs == rhs)


    # def add_cvar_constraints(self, *,
    #                         num_scenarios: int,
    #                         sigma: gp.Var,  
    #                         phi: Dict[int, gp.Var], 
    #                         f: List[gp.LinExpr]               
    #                         ) -> None:
    #     """
    #     添加CVaR约束

    #         num_scenarios: 场景数量
    #         sigma: VaR变量σ 利润的风险值
    #         phi: 辅助变量 φ_w 的字典
    #         f: 每个场景对应的损失或负收益表达式 f_w
    #     """
    #     for w in range(num_scenarios):
    #         # φ_w ≥ σ - f_w
    #         self.model.addConstr(phi[w] >= sigma - f[w])
    #         # φ_w ≥ 0
    #         self.model.addConstr(phi[w] >= 0)


    def add_battery_degradation_constraints(self, *,
                                            d: gp.Var,  # DOD变量（Depth of Discharge，0~1）
                                            L: gp.Var,  # 电池循环寿命变量（life cycles）
                                            N0: float,  # 常数 按经验取个5000
                                            beta: float # 常数 通常取0.3-0.7
                                            ) -> None:
        # log(L) + beta * log(d) = log(N0)
        log_d = self.model.addVar(name="log_d")
        log_L = self.model.addVar(name="log_L")
        self.model.addGenConstrLog(d, log_d, name="log_d_constr")
        self.model.addGenConstrLog(L, log_L, name="log_L_constr")
        self.model.addConstr(log_L + beta * log_d == np.log(N0))
        self.model.addConstr(d >= 1e-4)
        self.model.addConstr(d <= 0.9)


    def add_all_constraints(self) -> None:
        self.add_uc_ev_constraints()
        self.add_cc_ev_constraints()
        self.add_ev_fleet_aggregate_constraints()
        self.add_es1_constraints()
        self.add_es2_constraints()
        self.add_es_total_constraints()
        self.add_es2_backup_constraints()
    #    self.add_cvar_constraints()
        self.add_battery_degradation_constraints()
