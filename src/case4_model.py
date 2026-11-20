import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import os
from typing import Dict, List
from constraints import V2GConstraintsCase4

class V2GOptimizationModelCase4:
    def __init__(
        self,
        case: int,
        reduced_ev_scenarios: List[pd.DataFrame],  #传入ev的数据
        reduced_price_scenarios: List[pd.DataFrame],  #传入价格数据：dam rtm价格和投标价、激活价
        reduced_agc_scenarios: List[pd.DataFrame],  #传入agc数据
        capacity_reserves: pd.DataFrame,  # 每个时段的容量预留值DataFrame(timeslot, K_up, K_dn)
        beta: float,  #CVaR变量之一
        alpha: float,  #CVaR变量之一
        P_es_max: float = 0.09,  # ES最大充放电功率1.6（MW）
        E_es_max: float = 0.18,  # ES最大能量容量3.2（MWh）
        E_es_init: float = 0.09,  # ES初始能量1.0（MWh）
        dod_max: float = 0.9,    # 最大允许DOD
        gamma: float = 0.3,      # 最终能量与初始能量偏差容忍率
        T: int = 96,
        kappa: float = 0.5,  #等于最大充放电功率除以最大储能
        delta_t: float = 0.25,  #0.25h = 15min
        beta_ES: float = 0.58, #ES degredation公式中的系数 一般取0.3-0.7
        N0: float = 10921.8, #ES degredation公式中的系数 经验取值
        C_es: float = 1e5, #ES置换费用 - 调整为更现实的现代储能成本
        eta_ch = 0.95, #ES充电功率
        eta_dis = 0.95 #ES放电功率
    ):

        self.case = case
        self.reduced_ev_scenarios = reduced_ev_scenarios
        self.reduced_price_scenarios = reduced_price_scenarios
        self.reduced_agc_scenarios = reduced_agc_scenarios
        self.capacity_reserves = capacity_reserves
        self.num_scenarios = len(reduced_ev_scenarios)
        self.beta = beta
        self.alpha = alpha
        self.P_es_max = P_es_max
        self.E_es_max = E_es_max
        self.E_es_init = E_es_init
        self.dod_max = dod_max
        self.gamma = gamma
        self.T = T
        self.kappa = kappa
        self.delta_t = delta_t
        self.beta_ES = beta_ES
        self.N0 = N0
        self.C_es = C_es
        self.eta_ch = eta_ch
        self.eta_dis = eta_dis
        self.model = None
        self.results = {}

    def build_model(self):
        self.model = gp.Model("V2G_Optimization_Case4")
        constraints = V2GConstraintsCase4(self.model)

        # CVaR相关变量
        sigma = self.model.addVar(name="sigma")
        phi = self.model.addVars(self.num_scenarios, name="phi")

        # 电池退化相关变量
        dod = self.model.addVar(lb=1e-4, ub=0.9, name="DOD")
        L = self.model.addVar(lb=1, name="N_cycle")

        # 定义主决策变量 - 场景无关变量
        P_ev0_total = self.model.addVars(self.T, lb=-1e6, ub=1e6, name="P_ev0_total") # 能源投标 - 场景无关
        R_ev_up = self.model.addVars(self.T, lb=0, ub=1e6, name="R_ev_up") # 上调频容量投标 - 场景无关 
        R_ev_dn = self.model.addVars(self.T, lb=0, ub=1e6, name="R_ev_dn") # 下调频容量投标 - 场景无关
        R_es_up = self.model.addVars(self.T, lb = 0, ub = self.P_es_max, name="R_es_up") # Regulation up capacity bids of ES in hour t (MW)
        R_es_dn = self.model.addVars(self.T, lb=0, ub=self.P_es_max, name="R_es_dn") # Regulation down capacity bids of ES in hour t (MW)
        P_es0 = self.model.addVars(self.T, lb = - self.P_es_max, ub=self.P_es_max, name="P_es0") # Energy bids (also PSP) of ES in hour t (MW)
        E_es1_max = self.model.addVar(lb=0, ub=self.E_es_max, name="E_es1_max") # Maximum energy stored in ES1 (MW)
        E_es2_max = self.model.addVar(lb=0, ub=self.E_es_max, name="E_es2_max") # Maximum energy stored in ES2 (MW)
        E_es1_init  = self.model.addVar(lb=0, ub=self.E_es_max, name="E_es1_init") # Initial energy stored in ES1 (MW)
        E_es2_init = self.model.addVar(lb=0, ub=self.E_es_max, name="E_es2_init") # Initial energy stored in ES2 (MW)
        P_es1_max = self.model.addVar(lb=0, ub=self.P_es_max, name="P_es1_max") # Maximum charging(discharging) power of ES1 (MW)
        P_es2_max = self.model.addVar(lb=0, ub=self.P_es_max, name="P_es2_max") # Maximum charging(discharging) power of ES2 (MW)
        
        # 场景相关的其他变量 一维/二维
        E_es1 = self.model.addVars(self.num_scenarios, self.T, lb=0, ub=self.E_es_max, name="E_es1") # Energy stored in ES1 in hour t in scenario w (MW)
        E_es2 = self.model.addVars(self.num_scenarios, self.T, lb=0, ub=self.E_es_max, name="E_es2") # Energy stored in ES2 in hour t in scenario w (MW)
        f_w = self.model.addVars(self.num_scenarios, lb=-1e6, ub=1e6, name="f_w") # Expected net profit of EV aggregator in scenario w (€)
        F_ev_buy = self.model.addVars(self.num_scenarios, self.T, lb=0, ub=1e6, name="F_ev_buy") # DAM energy cost of EVA in hour t in scenario w (€)
        F_ev_sell = self.model.addVars(self.num_scenarios, self.T, lb=0, ub=1e6, name="F_ev_sell") # Income of EVA in hour t in scenario w (€)
        F_ev_charging = self.model.addVars(self.num_scenarios, self.T, lb=-1e6, ub=1e6, name="F_ev_charging") # Profit from charging the EV fleets in hour t in scenario w (€)
        F_ev_cap = self.model.addVars(self.num_scenarios, self.T, lb=0, ub=1e6, name="F_ev_cap") # Regulation capacity income of EVs in hour t in scenario w (€)
        F_es_cap = self.model.addVars(self.num_scenarios, self.T, lb=0, ub=1e6, name="F_es_cap") # Regulation capacity income of ES in hour t in scenario w (€)
        F_ev_mil = self.model.addVars(self.num_scenarios, self.T, lb=0, ub=1e6, name="F_ev_mil") # Regulation mileage income of EVs in hour t in scenario w (€)
        F_es_mil = self.model.addVars(self.num_scenarios, self.T, lb=0, ub=1e6, name="F_es_mil") # Regulation mileage income of ES in hour t in scenario w (€)
        F_ev_deploy = self.model.addVars(self.num_scenarios, self.T, lb=-1e6, ub=1e6, name="F_ev_deploy") # RTM energy cost of EVs for deploying regulation in hour t in scenario w (€)
        F_es_deploy = self.model.addVars(self.num_scenarios, self.T, lb=-1e6, ub=1e6, name="F_es_deploy") # RTM energy cost of ES for deploying regulation in hour t in scenario w (€)
        F_es_arb = self.model.addVars(self.num_scenarios, self.T, lb=-1e6, ub=1e6, name="F_es_arb") # ES income from arbitrage in DAM in hour t in scenario w (€)
        F_es_deg = self.model.addVars(self.num_scenarios, self.T, lb=0, ub=1e6, name="F_es_deg") # Degradation cost of ES in hour t in scenario w (€)
        P_ev_total = self.model.addVars(self.num_scenarios, self.T, lb=-1e6, ub=1e6, name="P_ev_total")  # Total charging(+)/discharging(-) power of EVs
        P_es_ch = self.model.addVars(self.num_scenarios, self.T, lb=0, ub=self.P_es_max, name="P_es_ch") # Charging power of ES in hour t in scenario w (MW)
        P_es_dis = self.model.addVars(self.num_scenarios, self.T, lb=0, ub=self.P_es_max, name="P_es_dis") # Discharging power of ES in hour t in scenario w (MW)
        P_es1 = self.model.addVars(self.num_scenarios, self.T, lb=-1e6, ub=self.P_es_max, name="P_es1") # Expected power of ES1 in hour t in scenario w (MW)
        P_es2 = self.model.addVars(self.num_scenarios, self.T, lb=-1e6, ub=self.P_es_max, name="P_es2") # Expected power of ES2 in hour t in scenario w (MW)
        P_es1_ch = self.model.addVars(self.num_scenarios, self.T, lb=0, ub=self.P_es_max, name="P_es1_ch") # Charging component of expected power of ES1 in hour t in scenario w (MW)
        P_es1_dis = self.model.addVars(self.num_scenarios, self.T, lb=0, ub=self.P_es_max, name="P_es1_dis") # Discharging component of expected power of ES1 in hour t in scenario w (MW)
        P_es2_ch = self.model.addVars(self.num_scenarios, self.T, lb=0, ub=self.P_es_max, name="P_es2_ch") # Charging component of expected power of ES2 in hour t in scenario w (MW)
        P_es2_dis = self.model.addVars(self.num_scenarios, self.T, lb=0, ub=self.P_es_max, name="P_es2_dis") # Disharging component of expected power of ES2 in hour t in scenario w (MW)
        # mu_es_ch = self.model.addVars(self.num_scenarios, self.T, vtype=GRB.BINARY, name="mu_es_ch") # Binary variable for charging power of entire ES in hour t in scenario w
        # mu_es_dis = self.model.addVars(self.num_scenarios, self.T, vtype=GRB.BINARY, name="mu_es_dis") # Binary variable for discharging power of entire ES in hour t in scenario w
        # mu_es1_ch = self.model.addVars(self.num_scenarios, self.T, vtype=GRB.BINARY, name="mu_es1_ch") # Binary variable for expected charging power of ES1 in hour t in scenario w
        # mu_es1_dis = self.model.addVars(self.num_scenarios, self.T, vtype=GRB.BINARY, name="mu_es1_dis") # Binary variable for expected discharging power of ES1 in hour t in scenario w
        # mu_es2_ch = self.model.addVars(self.num_scenarios, self.T, vtype=GRB.BINARY, name="mu_es2_ch") # Binary variable for expected charging power of ES2 in hour t in scenario w
        # mu_es2_dis = self.model.addVars(self.num_scenarios, self.T, vtype=GRB.BINARY, name="mu_es2_dis") # Binary variable for expected discharging power of ES2 in hour t in scenario w
        y_es  = self.model.addVars(self.num_scenarios, self.T, vtype=GRB.BINARY, name="y_es")    # 总 ES 层面
        y_es1 = self.model.addVars(self.num_scenarios, self.T, vtype=GRB.BINARY, name="y_es1")   # ES1
        y_es2 = self.model.addVars(self.num_scenarios, self.T, vtype=GRB.BINARY, name="y_es2")

        #调试
        P_ev0_cc_total = self.model.addVars(self.num_scenarios, self.T, lb=-1e6, ub=1e6, name="P_ev0_cc_total")
        R_ev_dn_cc_total = self.model.addVars(self.num_scenarios, self.T, lb=0.0, name="R_ev_dn_cc_total")
        R_ev_up_cc_total = self.model.addVars(self.num_scenarios, self.T, lb=0.0, name="R_ev_up_cc_total")

        # 预计算在场数组 presence_cc / presence_uc，形状 [W, T]
        self.presence_cc = np.zeros((self.num_scenarios, self.T), dtype=int)
        self.presence_uc = np.zeros((self.num_scenarios, self.T), dtype=int)
        for w in range(self.num_scenarios):
            ev_df = self.reduced_ev_scenarios[w]
            for _, row in ev_df.iterrows():
                Ta = int(row.get('arrival_time', 0))
                Td = int(row.get('departure_time', 0))
                is_overnight = (row.get('charging_type') == 'night')
                if row.get('ev_type') == 'cc':
                    target = self.presence_cc[w]
                else:
                    target = self.presence_uc[w]
                if is_overnight:
                    actual_td = Td - self.T
                    for t in range(self.T):
                        if t >= Ta or t < max(0, actual_td):
                            target[t] += 1
                else:
                    for t in range(self.T):
                        if Ta <= t < min(Td, self.T):
                            target[t] += 1

        # 动态生成EV相关变量keys  三维变量
        keys_cc = []
        keys_uc = []
        for w in range(self.num_scenarios):
            ev_profiles = self.reduced_ev_scenarios[w]
            N_cc = sum(ev_profiles['ev_type'] == 'cc')
            N_uc = sum(ev_profiles['ev_type'] == 'uc')
            for t in range(self.T):
                for n in range(N_cc):
                    keys_cc.append((w, t, n))
                for n in range(N_uc):
                    keys_uc.append((w, t, n))
        # 用keys方式创建变量
        P_ev_cc = self.model.addVars(keys_cc, lb=-0.022, ub=0.022, name="P_ev_cc") # Net power of controllable EV i (charge+:discharge-)
        P_ev_uc = self.model.addVars(keys_uc, lb=0, ub=0.022, name="P_ev_uc") # Charging power of uc individual EV i in hour t in scenario w (MW)
        P_ev0_cc = self.model.addVars(keys_cc, lb=-1e6, ub=1e6, name="P_ev0_cc") # PSP baseline (will be bounded by per-vehicle Pmax in constraints)
        P_ev0_uc = self.model.addVars(keys_uc, lb=0, ub=1e6, name="P_ev0_uc") # PSP of uc individual EV i in hour t in scenario w (MW)
        R_ev_up_i = self.model.addVars(keys_cc, lb=0, ub=22e-3, name="R_ev_up_i") # Upward regulation capacity provided by controllable individual EV i in hour t in scenario w (MW)
        R_ev_dn_i = self.model.addVars(keys_cc, lb=0, ub=22e-3, name="R_ev_dn_i") # Downward regulation capacity provided by controllable individual EV i in hour t in scenario w (MW)
        soc_cc = self.model.addVars(keys_cc, lb=0, ub=1, name="soc_cc") # SOC of individual cc EV i in hour t in scenario w
        soc_uc = self.model.addVars(keys_uc, lb=0, ub=1, name="soc_uc") # SOC of individual cc EV i in hour t in scenario w
        
        # Binary mutual exclusion per (w,t,n)
        # mu_ev_ch = self.model.addVars(keys_cc, vtype=GRB.BINARY, name="mu_ev_ch")
        # mu_ev_dis = self.model.addVars(keys_cc, vtype=GRB.BINARY, name="mu_ev_dis")
        y_ev = self.model.addVars(keys_cc, vtype=GRB.BINARY, name="y_ev")  # 每辆可控EV逐时段

        # Add controllable EV charge/discharge component variables for V2G
        P_ev_cc_ch = self.model.addVars(keys_cc, lb=0, name="P_ev_cc_ch")
        P_ev_cc_dis = self.model.addVars(keys_cc, lb=0, name="P_ev_cc_dis")
        # Effective power after efficiency piecewise
        P_ev_cc_ch_eff = self.model.addVars(keys_cc, lb=0, name="P_ev_cc_ch_eff")
        P_ev_cc_dis_eff = self.model.addVars(keys_cc, lb=0, name="P_ev_cc_dis_eff")


        # 添加ES2对每个可控EV的充放电功率变量
        P_es2_ch_i = self.model.addVars(keys_cc, lb=0, ub=self.P_es_max, name="P_es2_ch_i") # ES2 charging power allocated to controllable EV i in hour t in scenario w (MW)
        P_es2_dis_i = self.model.addVars(keys_cc, lb=0, ub=self.P_es_max, name="P_es2_dis_i") # ES2 discharging power allocated to controllable EV i in hour t in scenario w (MW)
        
        # 6. 添加约束
        d_star = 0.8  # 代表性 DoD，可做成超参数
        lambda_deg = self.C_es / (2.0 * self.eta_ch * self.eta_dis * self.N0 * (d_star ** (1.0 + self.beta_ES)))
        # constraints.add_battery_degradation_constraints(
        #     d = dod,
        #     L = L, 
        #     N0 = self.N0,  
        #     beta = self.beta_ES 
        #     )

        for w in range(self.num_scenarios):
            ev_profiles = self.reduced_ev_scenarios[w]
            price_scenario = self.reduced_price_scenarios[w]
            agc_scenario = self.reduced_agc_scenarios[w]
            cc_evs = ev_profiles[ev_profiles['ev_type'] == 'cc'].reset_index(drop=True)
            uc_evs = ev_profiles[ev_profiles['ev_type'] == 'uc'].reset_index(drop=True)
            N_cc = len(cc_evs)
            N_uc = len(uc_evs)
            
           # 不可控EV约束
            constraints.add_uc_ev_constraints(
                ev_profiles=ev_profiles,
                T=self.T,
                delta_t=self.delta_t,
                P_ev_uc=P_ev_uc,
                P_ev0_uc=P_ev0_uc,
                scenario_idx=w,
                soc = soc_uc,
                N_uc=N_uc
            )
            
            # 可控EV约束 - 使用时段特定的K_up和K_dn值
            constraints.add_cc_ev_constraints(
                ev_profiles=ev_profiles,
                delta_t=self.delta_t,
                T=self.T,
                scenario_idx=w,
                P_ev_cc=P_ev_cc,
                P_ev_cc_ch=P_ev_cc_ch,
                P_ev_cc_dis=P_ev_cc_dis,
                P_ev_cc_ch_eff=P_ev_cc_ch_eff,
                P_ev_cc_dis_eff=P_ev_cc_dis_eff,
                soc=soc_cc,
                P_ev0_cc=P_ev0_cc,
                R_ev_up_i=R_ev_up_i,
                R_ev_dn_i=R_ev_dn_i,
                R_ev_up=R_ev_up,
                R_ev_dn=R_ev_dn,
                K_up_values=self.capacity_reserves['K_up'].values,
                K_dn_values=self.capacity_reserves['K_dn'].values,
                N_cc=N_cc,
                y_ev=y_ev,
            )
            
            
            #EV fleet约束 - 使用场景无关的P_ev0_total
            constraints.add_ev_fleet_aggregate_constraints(
                scenario_idx=w,
                T=self.T,
                P_ev_uc=P_ev_uc,
                P_ev_cc=P_ev_cc,
                P_ev0_uc=P_ev0_uc,
                P_ev0_cc=P_ev0_cc,
                P_ev_total=P_ev_total,
                P_ev0_total=P_ev0_total,
                N_cc=N_cc,
                N_uc=N_uc,
                R_ev_dn_i=R_ev_dn_i,
                R_ev_up_i=R_ev_up_i,
                P_ev0_cc_total=P_ev0_cc_total,
                R_ev_dn_cc_total=R_ev_dn_cc_total,
                R_ev_up_cc_total=R_ev_up_cc_total
            )


            # ES1约束
            constraints.add_es1_constraints(
                T=self.T,
                scenario_idx=w,
                P_es1=P_es1,
                P_es1_ch=P_es1_ch,
                P_es1_dis=P_es1_dis,
                E_es1=E_es1,
                y_es1=y_es1,
                R_es_up=R_es_up,
                R_es_dn=R_es_dn,
                P_es0=P_es0,
                agc_up=agc_scenario['agc_up'],
                agc_dn=agc_scenario['agc_dn'],
                K_up_values=self.capacity_reserves['K_up'].values,
                K_dn_values=self.capacity_reserves['K_dn'].values,
                P_es1_max=P_es1_max,
                E_es1_max=E_es1_max,
                E_es1_init=E_es1_init,
                eta_ch=self.eta_ch,
                eta_dis=self.eta_dis,
                delta_t=self.delta_t
            )
            
            # ES2约束
            constraints.add_es2_constraints(
                T=self.T,
                scenario_idx=w,
                P_es2=P_es2,
                P_es2_ch=P_es2_ch,
                P_es2_dis=P_es2_dis,
                E_es2=E_es2,
                y_es2=y_es2,
                P_es2_max=P_es2_max,
                E_es2_max=E_es2_max,
                E_es2_init=E_es2_init,
                eta_ch=self.eta_ch,
                eta_dis=self.eta_dis,
                delta_t=self.delta_t,
                P_es2_ch_i=P_es2_ch_i,
                P_es2_dis_i=P_es2_dis_i,
                N_cc=N_cc
            )
            
            # ES整体约束
            constraints.add_es_total_constraints(
                T=self.T,
                scenario_idx=w,
                P_es_ch=P_es_ch,
                P_es_dis=P_es_dis,
                P_es1=P_es1,
                P_es2=P_es2,
                E_es1=E_es1,
                E_es2=E_es2,
                y_es=y_es,
                P_es1_max=P_es1_max,
                P_es2_max=P_es2_max,
                P_es_max=self.P_es_max,
                E_es1_max=E_es1_max,
                E_es2_max=E_es2_max,
                E_es_max=self.E_es_max,
                E_es1_init=E_es1_init,
                E_es2_init=E_es2_init,
                E_es_init=self.E_es_init,
                dod=dod,
                kappa=self.kappa,
                gamma=self.gamma
            )
            
            # ES2 backup约束
            constraints.add_es2_backup_constraints(
                T=self.T,
                scenario_idx=w,
                P_ev0_total=P_ev0_total,
                R_ev_up=R_ev_up,
                R_ev_dn=R_ev_dn,
                agc_up=agc_scenario['agc_up'],
                agc_dn=agc_scenario['agc_dn'],
                P_es2_ch=P_es2_ch,
                P_es2_dis=P_es2_dis,
                P_es2_ch_i=P_es2_ch_i,
                P_es2_dis_i=P_es2_dis_i,
                P_ev_total=P_ev_total,
                N_cc=N_cc
            )

            # 两车一桩功率约束
            constraints.add_charge_station_constraints(
                ev_profiles=ev_profiles,
                scenario_idx=w,
                T=self.T,
                P_ev_cc_ch=P_ev_cc_ch,
                P_ev_cc_dis=P_ev_cc_dis,
            )
            

        # 目标函数
        pi = 1.0 / self.num_scenarios  #参考的论文里的piw 代表每个场景发生的权重 这里按概率均等
        scenario_revenues = []  # 用于CVaR计算
        
        for w in range(self.num_scenarios):
            ev_profiles = self.reduced_ev_scenarios[w]
            cc_evs = ev_profiles[ev_profiles['ev_type'] == 'cc'].reset_index(drop=True)
            uc_evs = ev_profiles[ev_profiles['ev_type'] == 'uc'].reset_index(drop=True)
            N_cc = len(cc_evs)
            N_uc = len(uc_evs)
            price_scenario = self.reduced_price_scenarios[w]
            agc_scenario = self.reduced_agc_scenarios[w]

            expr = gp.LinExpr()
            for t in range(self.T):
                # EV收益 
                # 所有EV dam买电成本
                self.model.addConstr(
                    F_ev_buy[w, t] == P_ev0_total[t] * price_scenario['dam_prices'].iloc[t] * self.delta_t
                )
                # EVA卖电收入（分开cc和uc因为他们充电价格不一样）
                # Only charge component is billed; discharging does not incur charging fee
                self.model.addConstr(
                    F_ev_sell[w, t] == (
                        gp.quicksum(P_ev_cc_ch[w, t, n] * cc_evs.iloc[n]['charging_price'] for n in range(N_cc)) + 
                        gp.quicksum(P_ev_uc[w, t, n] * uc_evs.iloc[n]['charging_price'] for n in range(N_uc))
                    ) * self.delta_t
                )
                # EV充电收益
                self.model.addConstr(
                    F_ev_charging[w, t] == F_ev_sell[w, t] - F_ev_buy[w, t]
                )
                # EV调频投标收益
                self.model.addConstr(
                    F_ev_cap[w, t] == (R_ev_up[t] * price_scenario['afrr_up_cap_prices'].iloc[t] +
                                       R_ev_dn[t] * price_scenario['afrr_dn_cap_prices'].iloc[t]) * self.delta_t
                )
                # EV调频激活收益
                self.model.addConstr(
                    F_ev_mil[w, t] == (R_ev_up[t] * price_scenario['mileage_multiplier_up'].iloc[t] + 
                                       R_ev_dn[t] * price_scenario['mileage_multiplier_dn'].iloc[t]) * 
                                       price_scenario['balancing_prices'].iloc[t] * self.delta_t
                )
                # EV deploy cost (调频过程中买电的钱)
                self.model.addConstr(
                    F_ev_deploy[w, t] == (-R_ev_up[t] * agc_scenario['agc_up'].iloc[t] + R_ev_dn[t] * agc_scenario['agc_dn'].iloc[t]) * price_scenario['rtm_prices'].iloc[t] * self.delta_t
                )

                # ES收益 
                # ES套利收益(低价买入)
                self.model.addConstr(
                    F_es_arb[w, t] == -P_es0[t] * price_scenario['dam_prices'].iloc[t] * self.delta_t
                )
                # ES调频投标收益
                self.model.addConstr(
                    F_es_cap[w, t] == (R_es_up[t] * price_scenario['afrr_up_cap_prices'].iloc[t] +
                            R_es_dn[t] * price_scenario['afrr_dn_cap_prices'].iloc[t]) * self.delta_t
                )
                # ES调频激活收益
                self.model.addConstr(
                    F_es_mil[w, t] == (R_es_up[t] * price_scenario['mileage_multiplier_up'].iloc[t] + 
                                       R_es_dn[t] * price_scenario['mileage_multiplier_dn'].iloc[t]) * price_scenario['balancing_prices'].iloc[t] * self.delta_t
                )

                #ES deploy cost
                self.model.addConstr(
                    F_es_deploy[w, t] == (-R_es_up[t] * agc_scenario['agc_up'].iloc[t] + R_es_dn[t] * agc_scenario['agc_dn'].iloc[t]) * price_scenario['rtm_prices'].iloc[t] * self.delta_t
                )

                # 储能系统退化成本（支出项，需从总收益中扣除）
                # 按照原文公式：F_es_deg = C(P_es_ch + P_es_dis) / (2L*d*η_ch*η_dis)
                self.model.addConstr(
                    F_es_deg[w, t] == lambda_deg * (P_es_ch[w, t] + P_es_dis[w, t]),
                    name=f"deg_linear[{w},{t}]"
                )

                expr += (F_ev_charging[w, t]  + F_ev_cap[w, t] + F_ev_mil[w, t] - F_ev_deploy[w, t] - F_es_deg[w, t] )
            self.model.addConstr(f_w[w] == expr)
            scenario_revenues.append(f_w[w])  # 添加到场景收益列表

        # 添加CVaR约束
        constraints.add_cvar_constraints(
            num_scenarios=self.num_scenarios,
            sigma=sigma,
            phi=phi,
            f=scenario_revenues,
            pi=pi,
            beta=self.beta
        )
        
        # 设置带CVaR的目标函数
        cvar_expr = sigma - (1/(1-self.alpha)) * gp.quicksum(pi * phi[w] for w in range(self.num_scenarios))
        expected_profit = gp.quicksum(pi * f_w[w] for w in range(self.num_scenarios))

        # 目标函数：(1-beta)*期望收益 + beta*CVaR
        self.model.setObjective(
            (1-self.beta) * expected_profit + self.beta * cvar_expr,
            GRB.MAXIMIZE
        )

    def solve(self):
        self.build_model()
        # 添加Gurobi参数设置，用于诊断
        self.model.setParam('NonConvex', 2)   # 保险；A 已线性化但留着无害
        self.model.setParam('Presolve', 2)
        self.model.setParam('Heuristics', 0.5)
        self.model.setParam('RINS', 20)
        self.model.setParam('MIPFocus', 1)     # 先找可行解
        self.model.setParam('MIPGap', 0.05)    # 先放宽；跑通后再收紧
        # self.model.setParam('TimeLimit', 7200) # 可按需调整
        self.model.setParam('NodefileStart', 1.0)
        self.model.setParam('Threads', 16)
        self.model.setParam('Cuts', 3)
        self.model.optimize()
        
        # 写出标准LP文件（始终可写）
        self.model.write("model.lp")
        
        # 先根据状态分支，再安全访问 SolCount
        if self.model.status in [GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT]:
            try:
                if getattr(self.model, "SolCount", 0) and getattr(self.model, "SolCount", 0) > 0:
                    self.model.write("model.sol")
            except Exception:
                pass
            if self.model.status == GRB.OPTIMAL:
                return self.get_results()
        
        # 如果不可行，写出IIS文件
        if self.model.status == GRB.INFEASIBLE:
            print("模型不可行，正在写出IIS约束...")
            self.model.computeIIS()
            self.model.write("model.ilp")
            print("IIS已写入model.ilp")
        elif self.model.status == GRB.INF_OR_UNBD:
            print("模型不可行或无界，尝试进一步诊断...")
            self.model.setParam("InfUnbdInfo", 1)
            self.model.optimize()
            self.model.write("model.lp")
            if self.model.status == GRB.INFEASIBLE:
                print("模型不可行，正在写出IIS约束...")
                self.model.computeIIS()
                self.model.write("model.ilp")
                print("IIS已写入model.ilp")
            else:
                raise Exception("优化问题无解（无界）")
        else:
            raise Exception("优化问题无解")
            
    def get_results(self) -> Dict:
        # 计算各类收益和成本的总和
        ev_cap_revenue = 0
        ev_mil_revenue = 0
        ev_charging_revenue = 0
        ev_dam_cost = 0
        ev_deploy_cost = 0
        es_arb_revenue = 0
        es_cap_revenue = 0
        es_mil_revenue = 0
        es_deploy_cost = 0
        es_deg_cost = 0
        
        # 对所有场景进行求和
        pi = 1.0 / self.num_scenarios  # 每个场景的权重
        for w in range(self.num_scenarios):
            for t in range(self.T):
                ev_cap_revenue += pi * self.model.getVarByName(f"F_ev_cap[{w},{t}]").X
                ev_mil_revenue += pi * self.model.getVarByName(f"F_ev_mil[{w},{t}]").X
                ev_charging_revenue += pi * self.model.getVarByName(f"F_ev_sell[{w},{t}]").X
                ev_dam_cost += pi * self.model.getVarByName(f"F_ev_buy[{w},{t}]").X
                ev_deploy_cost += pi * self.model.getVarByName(f"F_ev_deploy[{w},{t}]").X
                es_arb_revenue += pi * self.model.getVarByName(f"F_es_arb[{w},{t}]").X
                es_cap_revenue += pi * self.model.getVarByName(f"F_es_cap[{w},{t}]").X
                es_mil_revenue += pi * self.model.getVarByName(f"F_es_mil[{w},{t}]").X
                es_deploy_cost += pi * self.model.getVarByName(f"F_es_deploy[{w},{t}]").X
                es_deg_cost += pi * self.model.getVarByName(f"F_es_deg[{w},{t}]").X
        
        # 获取CVaR值
        sigma_value = self.model.getVarByName("sigma").X
        cvar_value = sigma_value - (1/(1-self.alpha)) * sum(pi * self.model.getVarByName(f"phi[{w}]").X for w in range(self.num_scenarios))
        
        # 计算期望收益
        expected_profit = sum(pi * self.model.getVarByName(f"f_w[{w}]").X for w in range(self.num_scenarios))
        
        results = {
            "objective_value": self.model.objVal,
            "ev_cap_revenue": ev_cap_revenue,  # EV调频容量收入
            "ev_mil_revenue": ev_mil_revenue,  # EV调频里程收入
            "ev_charging_revenue": ev_charging_revenue,  # EV充电收入
            "ev_dam_cost": ev_dam_cost,  # 日前市场能源成本
            "ev_deploy_cost": ev_deploy_cost,  # 调频部署成本
            "es_cap_revenue": es_cap_revenue,  # ES调频容量收入
            "es_mil_revenue": es_mil_revenue,  # ES调频里程收入
            "es_arb_revenue": es_arb_revenue,  # ES能量市场套利收入
            "es_deploy_cost": es_deploy_cost,  # ES调频部署成本
            "es_deg_cost": es_deg_cost,  # ES退化成本
            "cvar_value": cvar_value,  # CVaR值
            "expected_profit": expected_profit,  # 期望收益
        }
        
        # 保存能源和调频投标决策
        energy_ev_bids = {}
        energy_es_bids = {}
        reg_up_ev_bids = {}
        reg_dn_ev_bids = {}
        reg_up_es_bids = {}
        reg_dn_es_bids = {}
        
        for t in range(self.T):
            # 使用场景无关的投标变量
            energy_ev_bids[t] = self.model.getVarByName(f"P_ev0_total[{t}]").X
            energy_es_bids[t] = self.model.getVarByName(f"P_es0[{t}]").X
            reg_up_ev_bids[t] = self.model.getVarByName(f"R_ev_up[{t}]").X
            reg_dn_ev_bids[t] = self.model.getVarByName(f"R_ev_dn[{t}]").X
            reg_up_es_bids[t] = self.model.getVarByName(f"R_es_up[{t}]").X
            reg_dn_es_bids[t] = self.model.getVarByName(f"R_es_dn[{t}]").X
            
        results["energy_ev_bids"] = energy_ev_bids
        results["energy_es_bids"] = energy_es_bids
        results["reg_up_ev_bids"] = reg_up_ev_bids
        results["reg_dn_ev_bids"] = reg_dn_ev_bids
        results["reg_up_es_bids"] = reg_up_es_bids
        results["reg_dn_es_bids"] = reg_dn_es_bids
        
        # 储能分配情况
        results["P_es1_max"] = self.model.getVarByName("P_es1_max").X
        results["P_es2_max"] = self.model.getVarByName("P_es2_max").X
        results["E_es1_max"] = self.model.getVarByName("E_es1_max").X
        results["E_es2_max"] = self.model.getVarByName("E_es2_max").X
        
        # 场景0：输出所有夜间充电的可控EV在“晚上到达 → 次日早上离开”期间的出价 P_ev0_cc
        # try:
        #     w = 0
        #     ev_profiles = self.reduced_ev_scenarios[w]
        #     cc_evs = ev_profiles[ev_profiles['ev_type'] == 'cc'].reset_index(drop=True)
        #     night_cc_records = []
        #     for n in range(len(cc_evs)):
        #         row = cc_evs.iloc[n]
        #         if str(row.get('charging_type', '')) != 'day':
        #             continue
        #         Ta = int(row['arrival_time'])
        #         Td = int(row['departure_time'])  # 跨天，>95
        #         actual_td = Td - 96
        #         # 夜间段：t in [Ta..95]，早晨段：t in [0..actual_td]
        #         timeslots = list(range(Ta, self.T)) + list(range(0, actual_td + 1))
        #         bid_series = {}
        #         for t in timeslots:
        #             var = self.model.getVarByName(f"P_ev0_cc[{w},{t},{n}]")
        #             bid_series[t] = var.X if var is not None else None
        #         night_cc_records.append({
        #             'ev_id': int(row['ev_id']) if 'ev_id' in row else n,
        #             'n_index': n,
        #             'arrival_slot': Ta,
        #             'departure_slot_next_day': actual_td,
        #             'bids': bid_series
        #         })
        #     results["night_cc_ev_bids_s0"] = night_cc_records

        #     # 写出为CSV（长表：每行一个时间槽）
        #     output_dir = "plots"
        #     try:
        #         if not os.path.exists(output_dir):
        #             os.makedirs(output_dir)
        #     except Exception:
        #         pass

        #     rows = []
        #     for rec in night_cc_records:
        #         ev_id = rec.get('ev_id')
        #         n_idx = rec.get('n_index')
        #         Ta = rec.get('arrival_slot')
        #         Td_next = rec.get('departure_slot_next_day')
        #         for t, val in rec.get('bids', {}).items():
        #             rows.append({
        #                 'scenario': 0,
        #                 'ev_id': ev_id,
        #                 'n_index': n_idx,
        #                 'arrival_slot': Ta,
        #                 'departure_slot_next_day': Td_next,
        #                 'timeslot': t,
        #                 'P_ev0_cc_MW': val
        #             })
        #     df_csv = pd.DataFrame(rows)
        #     csv_path = os.path.join(output_dir, "cc_night_bids_scenario0.csv")
        #     try:
        #         df_csv.to_csv(csv_path, index=False, encoding='utf-8-sig')
        #     except Exception:
        #         # 文件写出失败不影响主流程
        #         pass
        # except Exception as _:
        #     # 提取失败不影响其它结果
        #     results["night_cc_ev_bids_s0"] = []

        return results