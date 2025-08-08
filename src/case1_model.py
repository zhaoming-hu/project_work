import gurobipy as gp
from gurobipy import GRB
import pandas as pd
from typing import Dict, List
from constraints import V2GConstraintsCase1

class V2GOptimizationModelCase1:
    def __init__(
        self,
        case: int,
        reduced_ev_scenarios: List[pd.DataFrame],  #传入ev的数据
        reduced_price_scenarios: List[pd.DataFrame],  #传入价格数据：dam rtm价格和投标价、激活价
        reduced_agc_scenarios: List[pd.DataFrame],  #传入agc数据
        capacity_reserves: pd.DataFrame,  # 每个时段的容量预留值DataFrame(timeslot, K_up, K_dn)
        beta: float,  #CVaR变量之一 在主函数中自己定义
        alpha: float,  #CVaR变量之一 在主函数中自己定义
        P_es_max: float = 1.6,  # ES最大充放电功率1.6 MW
        E_es_max: float = 3.2,  # ES最大能量容量3.2 MWh
        E_es_init: float = 1.0,  # ES初始能量1.0 MWh
        dod_max: float = 0.9,    # 最大允许DOD
        gamma: float = 0.3,      # 最终能量与初始能量偏差容忍率
        T: int = 96,
        kappa: float = 0.5,  #等于最大充放电功率除以最大储能
        delta_t: float = 0.25,  #0.25h = 15min
        beta_ES: float = 0.58, #ES degradation公式中的系数 拟合值取了0.58
        N0: float = 10921.8, #ES degradation公式中的系数 经验取值
        C_es: float = 1e5, #ES置换费用
        eta_ch = 0.95, #ES充电效率
        eta_dis = 0.95 #ES放电效率
    ):
        self.case = case
        self.reduced_ev_scenarios = reduced_ev_scenarios
        self.reduced_price_scenarios = reduced_price_scenarios
        self.reduced_agc_scenarios = reduced_agc_scenarios
        self.capacity_reserves = capacity_reserves
        self.num_scenarios = len(reduced_ev_scenarios)
        self.T = T
        self.delta_t = delta_t
        self.model = None
        self.variables = {}
        self.P_es_max = P_es_max
        self.E_es_max = E_es_max
        self.E_es_init = E_es_init
        self.dod_max = dod_max
        self.gamma = gamma
        self.kappa = kappa
        self.beta_ES = beta_ES
        self.N0 = N0
        self.C_es = C_es
        self.eta_ch = eta_ch
        self.eta_dis = eta_dis
        self.beta = beta
        self.alpha = alpha

    def build_model(self):
        self.model = gp.Model("V2G_Optimization")
        constraints = V2GConstraintsCase1(self.model)
        
        # CVaR相关变量
        sigma = self.model.addVar(name="sigma")
        phi = self.model.addVars(self.num_scenarios, name="phi")

        # 电池退化相关变量
        dod = self.model.addVar(lb=1e-6, ub=0.9, name="DOD")
        L = self.model.addVar(lb=5000, ub=1e6, name="N_cycle")  # 与约束保持一致

        # 定义主决策变量 - 场景无关的投标决策变量
        P_ev0_total = self.model.addVars(self.T, lb=-1e6, ub=1e6, name="P_ev0_total") # 能源投标 - 场景无关
        R_ev_up = self.model.addVars(self.T, lb=0, ub=1e6, name="R_ev_up") # ev上调频容量投标 - 场景无关 
        R_ev_dn = self.model.addVars(self.T, lb=0, ub=1e6, name="R_ev_dn") # ev下调频容量投标 - 场景无关


        # 场景相关的其他变量 一维/二维
        f_w = self.model.addVars(self.num_scenarios, lb=-1e6, ub=1e6, name="f_w") # Expected net profit of EV aggregator in scenario w (€)
        F_ev_buy = self.model.addVars(self.num_scenarios, self.T, lb=0, ub=1e6, name="F_ev_buy") # DAM energy cost of EVA in hour t in scenario w (€)
        F_ev_sell = self.model.addVars(self.num_scenarios, self.T, lb=0, ub=1e6, name="F_ev_sell") # Income of EVA in hour t in scenario w (€)
        F_ev_charging = self.model.addVars(self.num_scenarios, self.T, lb=-1e6, ub=1e6, name="F_ev_charging") # Profit from charging the EV fleets in hour t in scenario w (€)
        F_ev_cap = self.model.addVars(self.num_scenarios, self.T, lb=0, ub=1e6, name="F_ev_cap") # Regulation capacity income of EVs in hour t in scenario w (€)
        F_ev_mil = self.model.addVars(self.num_scenarios, self.T, lb=0, ub=1e6, name="F_ev_mil") # Regulation mileage income of EVs in hour t in scenario w (€)
        F_ev_deploy = self.model.addVars(self.num_scenarios, self.T, lb=-1e6, ub=1e6, name="F_ev_deploy") # RTM energy cost of EVs for deploying regulation in hour t in scenario w (€)
        F_es_deg = self.model.addVars(self.num_scenarios, self.T, lb=0, ub=1e6, name="F_es_deg") # Degradation cost of ES in hour t in scenario w (€)
        E_es = self.model.addVars(self.num_scenarios, self.T, lb=0, ub=3.2, name="E_es") # Energy stored in ES in hour t in scenario w (MW)
        P_ev_total = self.model.addVars(self.num_scenarios, self.T, lb=0, ub=1e6, name="P_ev_total")  # Total charging power of EVs in hour t in scenario w (MW)
        P_es_ch = self.model.addVars(self.num_scenarios, self.T, lb=0, ub=1.6, name="P_es_ch") # Charging power of ES in hour t in scenario w (MW)
        P_es_dis = self.model.addVars(self.num_scenarios, self.T, lb=0, ub=1.6, name="P_es_dis") # Discharging power of ES in hour t in scenario w (MW)
        P_es = self.model.addVars(self.num_scenarios, self.T, lb=-1.6, ub=1.6, name="P_es") # Expected power of ES in hour t in scenario w (MW)
        mu_es_ch = self.model.addVars(self.num_scenarios, self.T, vtype=GRB.BINARY, name="mu_es_ch") # Binary variable for charging power of entire ES in hour t in scenario w
        mu_es_dis = self.model.addVars(self.num_scenarios, self.T, vtype=GRB.BINARY, name="mu_es_dis") # Binary variable for discharging power of entire ES in hour t in scenario w

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
        P_ev_cc = self.model.addVars(keys_cc, lb=0, ub=0.0066, name="P_ev_cc") # Charging power of cc individual EV i in hour t in scenario w (MW)
        P_ev_uc = self.model.addVars(keys_uc, lb=0, ub=0.0066, name="P_ev_uc") # Charging power of uc individual EV i in hour t in scenario w (MW)
        P_ev0_cc = self.model.addVars(keys_cc, lb=0, ub=1e6, name="P_ev0_cc") # PSP of cc individual EV i in hour t in scenario w (MW)
        P_ev0_uc = self.model.addVars(keys_uc, lb=0, ub=1e6, name="P_ev0_uc") # PSP of uc individual EV i in hour t in scenario w (MW)
        R_ev_up_i = self.model.addVars(keys_cc, lb=0, ub=6.6e-3, name="R_ev_up_i") # Upward regulation capacity provided by controllable individual EV i in hour t in scenario w (MW)
        R_ev_dn_i = self.model.addVars(keys_cc, lb=0, ub=6.6e-3, name="R_ev_dn_i") # Downward regulation capacity provided by controllable individual EV i in hour t in scenario w (MW)
        soc = self.model.addVars(keys_cc, lb=0, ub=1, name="soc") # SOC of individual EV i in hour t in scenario w
        
        # 新增：ES2对每个可控EV的充放电功率变量  在case1里 不区分ES1/2 全是ES2的功能体现 所以写成了P_es_ch_i和P_es_dis_i
        P_es_ch_i = self.model.addVars(keys_cc, lb=0, ub=1.6, name="P_es_ch_i") # ES2 charging power allocated to controllable EV i in hour t in scenario w (MW)
        P_es_dis_i = self.model.addVars(keys_cc, lb=0, ub=1.6, name="P_es_dis_i") # ES2 discharging power allocated to controllable EV i in hour t in scenario w (MW)
              
        # 6. 添加约束

        # 电池退化约束 
        constraints.add_battery_degradation_constraints(
                                        d = dod,
                                        L = L, 
                                        N0 = self.N0,  
                                        beta = self.beta_ES 
                                        )
        
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
                P_ev0_uc=P_ev0_uc,
                scenario_idx=w,
                N_uc=N_uc
            )
            
            # 可控EV约束 - 使用时段特定的K_up和K_dn值
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
                K_up_values=self.capacity_reserves['K_up'].values,
                K_dn_values=self.capacity_reserves['K_dn'].values,
                N_cc=N_cc
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
                N_uc=N_uc
            )
                
            # ES2约束
            constraints.add_es_constraints(
                T=self.T,
                scenario_idx=w,
                P_es=P_es,
                P_es_ch=P_es_ch,
                P_es_dis=P_es_dis,
                E_es=E_es,
                mu_es_ch=mu_es_ch,
                mu_es_dis=mu_es_dis,
                P_es_max=self.P_es_max,
                E_es_max=self.E_es_max,
                E_es_init=self.E_es_init,
                eta_ch=self.eta_ch,
                eta_dis=self.eta_dis,
                delta_t=self.delta_t,
                dod=dod,
                gamma=self.gamma,
                kappa=self.kappa
            )
            
            # es2_backup 约束 - 传递时段特定的AGC激活值
            constraints.add_es_backup_constraints(
                T=self.T,
                scenario_idx=w,
                P_ev0_total=P_ev0_total,         
                R_ev_up=R_ev_up,       
                R_ev_dn=R_ev_dn,       
                agc_up=agc_scenario['agc_up'],       
                agc_dn=agc_scenario['agc_dn'],       
                P_es_ch=P_es_ch,     
                P_es_dis=P_es_dis,
                P_es_ch_i=P_es_ch_i,     
                P_es_dis_i=P_es_dis_i,
                P_ev_total=P_ev_total,
                N_cc=N_cc
            )


        
        # 场景收益列表 - 用于CVaR计算
        scenario_revenues = []
        pi = 1.0 / self.num_scenarios  # 参考的论文里的piw 代表每个场景发生的权重 这里按概率均等

        # 目标函数
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
                # 所有EV dam买电成本 - 使用场景无关的P_ev0_total
                self.model.addConstr(
                    F_ev_buy[w, t] == P_ev0_total[t] * price_scenario['dam_prices'].iloc[t] * self.delta_t
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
                # EV调频投标收益 - 使用场景无关的R_ev_up/R_ev_dn
                self.model.addConstr(
                    F_ev_cap[w, t] == (R_ev_up[t] * price_scenario['afrr_up_cap_prices'].iloc[t] +
                                       R_ev_dn[t] * price_scenario['afrr_dn_cap_prices'].iloc[t]) * self.delta_t
                )
                # EV调频激活收益 - 使用场景无关的R_ev_up/R_ev_dn
                self.model.addConstr(
                    F_ev_mil[w, t] == (R_ev_up[t] * price_scenario['mileage_multiplier_up'].iloc[t]
                                       + R_ev_dn[t] * price_scenario['mileage_multiplier_dn'].iloc[t]) * price_scenario['balancing_prices'].iloc[t] * self.delta_t
                )
                # EV deploy cost (调频过程中买电的钱) - 使用场景无关的R_ev_up/R_ev_dn
                self.model.addConstr(
                    F_ev_deploy[w, t] == (-R_ev_up[t] * agc_scenario['agc_up'].iloc[t] + R_ev_dn[t] * agc_scenario['agc_dn'].iloc[t]) * price_scenario['rtm_prices'].iloc[t] * self.delta_t
                )

                # 储能系统退化成本（支出项，需从总收益中扣除）
                # 按照原文公式：F_es_deg = C(P_es_ch + P_es_dis) / (2L*d*η_ch*η_dis)
                self.model.addConstr(
                    F_es_deg[w, t] == self.C_es * (P_es_ch[w, t] + P_es_dis[w, t]) / 
                    (2 * L * dod * self.eta_ch * self.eta_dis)
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
        self.model.setParam('NumericFocus', 3)  # 提高数值精度
        self.model.setParam('FeasibilityTol', 1e-6)  # 可行性容忍度
        self.model.setParam('IntFeasTol', 1e-6)  # 整数可行性容忍度
        self.model.setParam('MarkowitzTol', 0.01)  # 增加数值稳定性
        self.model.setParam('Method', 2)  # 求解方法设为barrier
        self.model.setParam('Crossover', 0)  # 关闭crossover以提高求解效率
        self.model.optimize()
        
        # 无论什么情况都写出标准LP文件
        # self.model.computeIIS()
        self.model.write("model.lp")
        # 如果不可行，写出IIS文件
        if self.model.status == GRB.INFEASIBLE:
            print("模型不可行，正在写出IIS约束...")
            self.model.computeIIS()
            self.model.write("model.ilp")
            print("IIS已写入model.ilp")
        elif self.model.status == GRB.OPTIMAL:
            return self.get_results()
        elif self.model.status == GRB.INF_OR_UNBD:
            print("模型不可行或无界，尝试进一步诊断...")
            self.model.setParam("InfUnbdInfo", 1)
            self.model.optimize()
            self.model.write("model.lp")
            if self.model.status== GRB.INFEASIBLE:
                print("模型不可行，正在写出IIS约束...")
                self.model.computeIIS()
                print("IIS已写入model.ilp")
            else:
                self.model.setParam("DualReductions", 0)
                self.model.optimize()
                print(self.model.status)
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
        es_deg_cost = 0
        
        # 对所有场景进行求和
        pi = 1.0 / self.num_scenarios  # 每个场景的权重
        for w in range(self.num_scenarios):
            for t in range(self.T):
                ev_cap_revenue += pi * self.model.getVarByName(f"F_ev_cap[{w},{t}]").X
                ev_mil_revenue += pi * self.model.getVarByName(f"F_ev_mil[{w},{t}]").X
                ev_charging_revenue += pi * self.model.getVarByName(f"F_ev_charging[{w},{t}]").X
                ev_dam_cost += pi * self.model.getVarByName(f"F_ev_buy[{w},{t}]").X
                ev_deploy_cost += pi * self.model.getVarByName(f"F_ev_deploy[{w},{t}]").X
                es_deg_cost += pi * self.model.getVarByName(f"F_es_deg[{w},{t}]").X

        
        # 获取CVaR值
        sigma_value = self.model.getVarByName("sigma").X
        cvar_value = sigma_value - (1/(1-self.alpha)) * sum(pi * self.model.getVarByName(f"phi[{w}]").X for w in range(self.num_scenarios))
        
        # 计算期望收益
        expected_profit = sum(pi * self.model.getVarByName(f"f_w[{w}]").X for w in range(self.num_scenarios))
        
        # Case 1没有ES的regulation相关收益，设为0
        es_cap_revenue = 0
        es_mil_revenue = 0
        es_arb_revenue = 0
        es_deploy_cost = 0
        
        results = {
            "objective_value": self.model.objVal,
            "ev_cap_revenue": ev_cap_revenue,  # EV调频容量收入
            "ev_mil_revenue": ev_mil_revenue,  # EV调频里程收入
            "ev_charging_revenue": ev_charging_revenue,  # EV充电收入
            "ev_dam_cost": ev_dam_cost,  # EV买电成本
            "ev_deploy_cost": ev_deploy_cost,  # 调频部署成本
            "es_cap_revenue": es_cap_revenue,  # ES调频容量收入
            "es_mil_revenue": es_mil_revenue,  # ES调频里程收入
            "es_arb_revenue": es_arb_revenue,  # ES能量市场套利收入
            "es_deploy_cost": es_deploy_cost,  # ES调频部署成本
            "es_deg_cost": es_deg_cost,  # ES退化成本
            "cvar_value": cvar_value,  # CVaR值
            "expected_profit": expected_profit,  # 期望收益
        }
        
        # 保存场景无关的投标决策
        energy_bids = {}
        reg_up_bids = {}
        reg_dn_bids = {}
        for t in range(self.T):
            energy_bids[t] = self.model.getVarByName(f"P_ev0_total[{t}]").X
            reg_up_bids[t] = self.model.getVarByName(f"R_ev_up[{t}]").X
            reg_dn_bids[t] = self.model.getVarByName(f"R_ev_dn[{t}]").X
            print(f"t: {t}, P_ev0_total: {energy_bids[t]}")
            # agc_scenario = self.reduced_agc_scenarios[w]
            # print(f"t: {t}, P_regulation: {reg_dn_bids[t] * agc_scenario['agc_dn'].iloc[t] - reg_up_bids[t] * agc_scenario['agc_up'].iloc[t]}")

        # 输出场景0的EV充电功率
        w = 0  # 场景0
        # print("\n输出场景0的EV充电功率:")
        # for t in range(self.T):
        #     print(f"t: {t}, P_ev_total[t]: {self.model.getVarByName(f'P_ev_total[{w},{t}]').X}")

        # print("\n输出场景0的ES充放电功率:")
        # for t in range(self.T):
        #     print(f"t: {t}, P_es_ch{t}: {self.model.getVarByName(f'P_es_ch[{w},{t}]').X} , P_es_dis[t]: {self.model.getVarByName(f'P_es_dis[{w},{t}]').X}") 

            
        results["energy_ev_bids"] = energy_bids
        results["reg_up_ev_bids"] = reg_up_bids
        results["reg_dn_ev_bids"] = reg_dn_bids
        
        return results
