import gurobipy as gp
from gurobipy import GRB
import pandas as pd
from typing import Dict, List
from constraints import V2GConstraints

class V2GOptimizationModel:
    def __init__(
        self,
        reduced_ev_scenarios: List[pd.DataFrame],  #传入ev的数据
        reduced_price_scenarios: List[pd.DataFrame],  #传入价格数据：dam rtm价格和投标价、激活价
        reduced_agc_scenarios: List[pd.DataFrame],  #传入agc数据
        K_dn: float,  #容量预留 固定值
        K_up: float,
        P_es_max: float = 1.6,  # ES最大充放电功率1.6（MW）
        E_es_max: float = 3.2,  # ES最大能量容量3.2（MWh）
        E_es_init: float = 3.2,  # ES初始能量3.2（MWh）
        E_es1_init: float = 1.0,  # ES初始能量1.0（MWh）
        E_es2_init: float = 2.2,  # ES初始能量2.2（MWh）
        dod_max: float = 0.9,    # 最大允许DOD
        gamma: float = 0.3,      # 最终能量与初始能量偏差容忍率
        T: int = 96,
        beta: float = 0.95,  #CVaR变量之一
        alpha: float = 0.5,  #CVaR变量之一
        kappa: float = 0.5,  #等于最大充放电功率除以最大储能
        delta_t: float = 0.25,  #0.25h = 15min
        beta_ES: float = 0.5, #ES degredation公式中的系数 一般取0.3-0.7
        N0: float = 5000, #ES degredation公式中的系数 经验取值
        C_es: float = 3e5, #ES置换费用
        eta_ch = 0.95, #ES充电功率
        eta_dis = 0.95 #ES放电功率
    ):
        self.reduced_ev_scenarios = reduced_ev_scenarios
        self.reduced_price_scenarios = reduced_price_scenarios
        self.reduced_agc_scenarios = reduced_agc_scenarios
        self.num_scenarios = len(reduced_ev_scenarios)
        self.T = T
        self.beta = beta
        self.alpha = alpha
        self.delta_t = delta_t
        self.model = None
        self.variables = {}
        self.K_dn = K_dn
        self.K_up = K_up
        self.P_es_max = P_es_max
        self.E_es_max = E_es_max
        self.E_es_init = E_es_init
        self.E_es1_init = E_es1_init
        self.E_es2_init = E_es2_init
        self.dod_max = dod_max
        self.gamma = gamma
        self.kappa = kappa
        self.beta_ES = beta_ES
        self.N0 = N0
        self.C_es = C_es
        self.eta_ch = eta_ch
        self.eta_dis = eta_dis

    def build_model(self):
        self.model = gp.Model("V2G_Optimization")
        constraints = V2GConstraints(self.model)

        # 定义主决策变量  全局/一维/二维变量
        E_es1_max = self.model.addVar(name="E_es1_max") # Maximum energy stored in ES1 (MW)
        E_es2_max = self.model.addVar(name="E_es2_max") # Maximum energy stored in ES2 (MW)
        f_w = self.model.addVars(self.num_scenarios, name="f_w") # Expected net profit of EV aggregator in scenario w (€)
        F_ev_buy = self.model.addVars(self.num_scenarios, self.T, name="F_ev_buy") # DAM energy cost of EVA in hour t in scenario w (€)
        F_ev_sell = self.model.addVars(self.num_scenarios, self.T, name="F_ev_sell") # Income of EVA in hour t in scenario w (€)
        F_ev_charging = self.model.addVars(self.num_scenarios, self.T, name="F_ev_charging") # Profit from charging the EV fleets in hour t in scenario w (€)
        F_ev_cap = self.model.addVars(self.num_scenarios, self.T, name="F_ev_cap") # Regulation capacity income of EVs in hour t in scenario w (€)
        F_es_cap = self.model.addVars(self.num_scenarios, self.T, name="F_es_cap") # Regulation capacity income of ES in hour t in scenario w (€)
        F_ev_mil = self.model.addVars(self.num_scenarios, self.T, name="F_ev_mil") # Regulation mileage income of EVs in hour t in scenario w (€)
        F_es_mil = self.model.addVars(self.num_scenarios, self.T, name="F_es_mil") # Regulation mileage income of ES in hour t in scenario w (€)
        F_ev_deploy = self.model.addVars(self.num_scenarios, self.T, name="F_ev_deploy") # RTM energy cost of EVs for deploying regulation in hour t in scenario w (€)
        F_es_deploy = self.model.addVars(self.num_scenarios, self.T, name="F_es_deploy") # RTM energy cost of ES for deploying regulation in hour t in scenario w (€)
        F_es_arb = self.model.addVars(self.num_scenarios, self.T, name="F_es_arb") # ES income from arbitrage in DAM in hour t in scenario w (€)
        F_es_deg = self.model.addVars(self.num_scenarios, self.T, name="F_es_deg") # Degradation cost of ES in hour t in scenario w (€)
        E_es1 = self.model.addVars(self.num_scenarios, self.T, name="E_es1") # Energy stored in ES1 in hour t in scenario w (MW)
        E_es2 = self.model.addVars(self.num_scenarios, self.T, name="E_es2") # Energy stored in ES2 in hour t in scenario w (MW)
        P_ev_total = self.model.addVars(self.num_scenarios, self.T, name="P_ev_total")  # Total charging power of EVs in hour t in scenario w (MW)
        P_ev0_total = self.model.addVars(self.num_scenarios, self.T, name="P_ev0_total") #Energy bids (also the preferred dispatch set point (PSP)) of EVs in hour t in scenario w(MW)
        P_es0 = self.model.addVars(self.num_scenarios, self.T, name="P_es0") # Energy bids (also PSP) of ES in hour t in scenario w(MW)
        P_es_ch = self.model.addVars(self.num_scenarios, self.T, name="P_es_ch") # Charging power of ES in hour t in scenario w (MW)
        P_es_dis = self.model.addVars(self.num_scenarios, self.T, name="P_es_dis") # Discharging power of ES in hour t in scenario w (MW)
        P_es1_max = self.model.addVar(name="P_es1_max") # Maximum charging(discharging) power of ES1 (MW)
        P_es2_max = self.model.addVar(name="P_es2_max") # Maximum charging(discharging) power of ES2 (MW)
        P_es1 = self.model.addVars(self.num_scenarios, self.T, name="P_es1") # Expected power of ES1 in hour t in scenario w (MW)
        P_es2 = self.model.addVars(self.num_scenarios, self.T, name="P_es2") # Expected power of ES2 in hour t in scenario w (MW)
        P_es1_ch = self.model.addVars(self.num_scenarios, self.T, name="P_es1_ch") # Charging component of expected power of ES1 in hour t in scenario w (MW)
        P_es1_dis = self.model.addVars(self.num_scenarios, self.T, name="P_es1_dis") # Discharging component of expected power of ES1 in hour t in scenario w (MW)
        P_es2_ch = self.model.addVars(self.num_scenarios, self.T, name="P_es2_ch") # Charging component of expected power of ES2 in hour t in scenario w (MW)
        P_es2_dis = self.model.addVars(self.num_scenarios, self.T, name="P_es2_dis") # Disharging component of expected power of ES2 in hour t in scenario w (MW)
        R_ev_up = self.model.addVars(self.num_scenarios, self.T, name="R_ev_up") # Regulation up capacity bids of ES in hour t in scenario w(MW)
        R_ev_dn = self.model.addVars(self.num_scenarios, self.T, name="R_ev_dn") # Regulation down capacity bids of ES in hour t in scenario w(MW)
        R_es_up = self.model.addVars(self.num_scenarios, self.T, name="R_es_up") # Regulation up capacity bids of ES in hour t in scenario w(MW)
        R_es_dn = self.model.addVars(self.num_scenarios, self.T, name="R_es_dn") # Regulation down capacity bids of ES in hour t in scenario w(MW)
        mu_es_ch = self.model.addVars(self.num_scenarios, self.T, name="mu_es_ch") # Binary variable for charging power of entire ES in hour t in scenario w
        mu_es_dis = self.model.addVars(self.num_scenarios, self.T, name="mu_es_dis") # Binary variable for discharging power of entire ES in hour t in scenario w
        mu_es1_ch = self.model.addVars(self.num_scenarios, self.T, name="mu_es1_ch") # Binary variable for expected charging power of ES1 in hour t in scenario w
        mu_es1_dis = self.model.addVars(self.num_scenarios, self.T, name="mu_es1_dis") # Binary variable for expected discharging power of ES1 in hour t in scenario w
        mu_es2_ch = self.model.addVars(self.num_scenarios, self.T, name="mu_es2_ch") # Binary variable for expected charging power of ES2 in hour t in scenario w
        mu_es2_dis = self.model.addVars(self.num_scenarios, self.T, name="mu_es2_dis") # Binary variable for expected discharging power of ES2 in hour t in scenario w

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
        P_ev_cc = self.model.addVars(keys_cc, name="P_ev_cc") # Charging power of cc individual EV i in hour t in scenario w (MW)
        P_ev_uc = self.model.addVars(keys_uc, name="P_ev_uc") # Charging power of uc individual EV i in hour t in scenario w (MW)
        P_ev0_cc = self.model.addVars(keys_cc, name="P_ev0_cc") # PSP of cc individual EV i in hour t in scenario w (MW)
        P_ev0_uc = self.model.addVars(keys_uc, name="P_ev0_uc") #P SP of ucc individual EV i in hour t in scenario w (MW)
        R_ev_up_i = self.model.addVars(keys_cc, name="R_ev_up_i") # Upward regulation capacity provided by controllable individual EV i in hour t in scenario w (MW)
        R_ev_dn_i = self.model.addVars(keys_cc, name="R_ev_dn_i") # Downward regulation capacity provided by controllable individual EV i in hour t in scenario w (MW)
        soc = self.model.addVars(keys_cc, name="soc") # SOC of individual EV i in hour t in scenario w


        # CVaR相关变量
        # sigma = self.model.addVar(name="sigma")
        # phi = self.model.addVars(self.num_scenarios, name="phi")

        # 电池退化相关变量
        dod = self.model.addVar(lb=1e-4, ub=0.9, name="DOD")
        L = self.model.addVar(lb=1, name="N_cycle")


        # 6. 添加约束
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
                P_ev_uc=P_ev_uc,
                scenario_idx=w,
                N_uc=N_uc
            )
            # 可控EV约束
            constraints.add_cc_ev_constraints(
                ev_profiles=ev_profiles,
                delta_t=self.delta_t,
                T=self.T,
                scenario_idx=w,
                P_ev_cc=P_ev_cc,
                soc=soc,
                P_ev0_cc=P_ev0_cc,
                R_ev_up_i=R_ev_up_i,
                R_ev_dn_i=R_ev_dn_i,
                R_ev_up=R_ev_up,
                R_ev_dn=R_ev_dn,
                K_dn=self.K_dn,
                N_cc=N_cc
            )
            #EV fleet约束
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
                N_uc=N_uc
            )
            # ES1约束
            constraints.add_es1_constraints(
                T=self.T,
                scenario_idx=w,
                P_es1=P_es1,
                P_es1_ch=P_es1_ch,
                P_es1_dis=P_es1_dis,
                E_es1=E_es1,
                mu_es1_ch=mu_es1_ch,
                mu_es1_dis=mu_es1_dis,
                R_es_up=R_es_up,
                R_es_dn=R_es_dn,
                P_es0=P_es0,
                agc_up=agc_scenario['agc_up'],
                agc_dn=agc_scenario['agc_dn'],
                K_up=self.K_up,
                K_dn=self.K_dn,
                P_es1_max=P_es1_max,
                E_es1_max=E_es1_max,
                E_es1_init=self.E_es1_init,
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
                mu_es2_ch=mu_es2_ch,
                mu_es2_dis=mu_es2_dis,
                P_es2_max=P_es2_max,
                E_es2_max=E_es2_max,
                E_es2_init=self.E_es2_init,
                eta_ch=self.eta_ch,
                eta_dis=self.eta_dis,
                delta_t=self.delta_t
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
                mu_es_ch=mu_es_ch,
                mu_es_dis=mu_es_dis,
                P_es1_max=P_es1_max,
                P_es2_max=P_es2_max,
                P_es_max=self.P_es_max,
                E_es1_max=E_es1_max,
                E_es2_max=E_es2_max,
                E_es_max=self.E_es_max,
                E_es_init=self.E_es_init,
                dod=dod,
                kappa=self.kappa,
                gamma=self.gamma
            )
            # 能量平衡约束
            constraints.add_es2_backup_constraints(
                T=self.T,
                scenario_idx=w,
                P_ev_total=P_ev_total,
                R_ev_up=R_ev_up,
                R_ev_dn=R_ev_dn,
                agc_up=agc_scenario['agc_up'],
                agc_dn=agc_scenario['agc_dn'],
                P_ev_uc=P_ev_uc,
                P_ev_cc=P_ev_cc,
                P_es2_ch=P_es2_ch,
                P_es2_dis=P_es2_dis,
                N_cc=N_cc,
                N_uc=N_uc
            )
            #电池退化约束
            constraints.add_battery_degradation_constraints(
                                            d = dod,
                                            L = L, 
                                            N0 = self.N0,  
                                            beta = self.beta_ES 
                                            )
            # CVaR约束
        #     constraints.add_cvar_constraints(
        #     num_scenarios=self.num_scenarios,
        #     sigma=sigma,
        #     phi=phi,
        #     f=scenario_revenues
        # )



        # 目标函数
        pi = 1.0 / self.num_scenarios  #参考的论文里的piw 代表每个场景发生的权重 这里按概率均等
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
                    F_ev_buy[w, t] == P_ev0_total[w, t] * price_scenario['dam_prices'].iloc[t] * self.delta_t
                )
                # EVA卖电收入（分开cc和uc因为他们充电价格不一样）
                self.model.addConstr(
                    F_ev_sell[w, t] == (
                        gp.quicksum(P_ev_cc[w, t, n] * cc_evs.iloc[n]['charging_price'] for n in range(N_cc)) +
                        gp.quicksum(P_ev_uc[w, t, n] * uc_evs.iloc[n]['charging_price'] for n in range(N_uc))
                    ) * self.delta_t
                )
                # EV充电收益
                self.model.addConstr(
                    F_ev_charging[w, t] == F_ev_sell[w, t] - F_ev_buy[w, t]
                )
                # EV调频投标收益
                self.model.addConstr(
                    F_ev_cap[w, t] == (R_ev_up[w, t] * price_scenario['afrr_up_cap_prices'].iloc[t] +
                                       R_ev_dn[w, t] * price_scenario['afrr_dn_cap_prices'].iloc[t]) * self.delta_t
                )
                # EV调频激活收益
                self.model.addConstr(
                    F_ev_mil[w, t] == (R_ev_up[w, t] + R_ev_dn[w, t]) * price_scenario['balancing_prices'].iloc[t] * self.delta_t
                )
                # EV deploy cost (调频过程中买电的钱)
                self.model.addConstr(
                    F_ev_deploy[w, t] == (-R_ev_up[w, t] * agc_scenario['agc_up'].iloc[t] + R_ev_dn[w, t] * agc_scenario['agc_dn'].iloc[t]) * price_scenario['rtm_prices'].iloc[t] * self.delta_t
                )

                # ES收益
                # ES套利收益(低价买入)
                self.model.addConstr(
                    F_es_arb[w, t] == - P_es0[w, t] * price_scenario['dam_prices'].iloc[t] * self.delta_t
                )
                # ES调频投标收益
                self.model.addConstr(
                    F_es_cap[w, t] == (R_es_up[w, t] * price_scenario['afrr_up_cap_prices'].iloc[t] +
                                       R_es_dn[w, t] * price_scenario['afrr_dn_cap_prices'].iloc[t]) * self.delta_t
                )
                # ES调频激活收益
                self.model.addConstr(
                    F_es_mil[w, t] == (R_es_up[w, t] + R_es_dn[w, t]) * price_scenario['balancing_prices'].iloc[t] * self.delta_t
                )

                #ES deploy cost
                self.model.addConstr(
                    F_es_deploy[w, t] == (-R_es_up[w, t] * agc_scenario['agc_up'].iloc[t] + R_es_dn[w, t] * agc_scenario['agc_dn'].iloc[t]) * price_scenario['rtm_prices'].iloc[t] * self.delta_t
                )

                # 储能系统退化成本（支出项，需从总收益中扣除）
                self.model.addConstr(
                    F_es_deg[w, t] == self.C_es * (P_es_ch[w, t] + P_es_dis[w, t]) / (2 * L * dod * self.eta_ch * self.eta_dis)
                )

                expr += (F_ev_charging[w, t] + F_ev_cap[w, t] + F_ev_mil[w, t] - F_ev_deploy[w, t] +
                         F_es_arb[w, t] + F_es_cap[w, t] + F_es_mil[w, t] - F_es_deploy[w, t] - F_es_deg[w, t])
            self.model.addConstr(f_w[w] == expr)

        self.model.setObjective(
            gp.quicksum(pi * f_w[w] for w in range(self.num_scenarios)),
            GRB.MAXIMIZE
        )

    def solve(self):
        self.build_model()
        self.model.optimize()
        if self.model.status == GRB.OPTIMAL:
            return self.get_results()
        elif self.model.status == GRB.INFEASIBLE:
            print("模型不可行，正在写出IIS约束...")
            self.model.computeIIS()
            self.model.write("model.ilp")
            print("IIS已写入model.ilp，请用Gurobi自带工具或文本编辑器查看。")
            raise Exception("优化问题无解（不可行）")
        elif self.model.status == GRB.INF_OR_UNBD:
            print("模型不可行或无界，尝试进一步诊断...")
            self.model.setParam("InfUnbdInfo", 1)
            self.model.optimize()
            if self.model.status == GRB.INFEASIBLE:
                print("模型不可行，正在写出IIS约束...")
                self.model.computeIIS()
                self.model.write("model.ilp")
                print("IIS已写入model.ilp，请用Gurobi自带工具或文本编辑器查看。")
                raise Exception("优化问题无解（不可行）")
            else:
                raise Exception("优化问题无解（无界）")
        else:
            raise Exception("优化问题无解")
            
    def get_results(self) -> Dict:
        results = {
            "objective_value": self.model.objVal,
            
        }
        return results 