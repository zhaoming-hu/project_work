import gurobipy as gp
from gurobipy import GRB
import pandas as pd
from typing import Dict, List
from constraints import V2GConstraints

class V2GOptimizationModel:
    def __init__(
        self,
        reduced_ev_scenarios: List[pd.DataFrame],  #传入ev的数据
        reduced_price_scenarios: List[Dict[str, pd.Series]],  #传入价格数据：dam rtm价格和投标价、激活价
        reduced_agc_scenarios: List[pd.DataFrame],  #传入agc数据
        K_dn: float,  #容量预留 固定值
        K_up: float,
        P_es_max: float = 1.6,  # ES最大充放电功率1.6（MW）
        E_es_max: float = 3.2,  # ES最大能量容量3.2（MWh）
        E_es_init: float = 1,  # ES初始能量1（MWh）
        dod_max: float = 0.9,    # 最大允许DOD
        gamma: float = 0.3,      # 最终能量与初始能量偏差容忍率
        T: int = 96,
        beta: float = 0.95,
        alpha: float = 0.5,
        delta_t: float = 0.25  #0.25h = 15min
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
        self.variables = {}  #先这么写 用来放置中间决策变量
        self.K_dn = K_dn
        self.K_up = K_up
        self.P_es_max = P_es_max
        self.E_es_max = E_es_max
        self.E_es_init = E_es_init
        self.dod_max = dod_max
        self.gamma = gamma

    def build_model(self):
        """
        这里建模的思路主要是：1.定义变量  2. 添加约束 
        """
        self.model = gp.Model("V2G_Optimization")
        constraints = V2GConstraints(self.model)

        # 1. 定义主决策变量
        P_ev0 = self.model.addVars(self.num_scenarios, self.T, name="P_ev0")   #所有ev的dam energy bids
        P_es0 = self.model.addVars(self.num_scenarios, self.T, name="P_es0")   #所有es的dam energy bids
        R_ev_up = self.model.addVars(self.num_scenarios, self.T, name="R_ev_up")  #所有ev的上调频容量bids
        R_ev_dn = self.model.addVars(self.num_scenarios, self.T, name="R_ev_dn")  #所有ev的下调频容量bids
        R_es_up = self.model.addVars(self.num_scenarios, self.T, name="R_es_up")  #es的上调频容量bids(es1负责)
        R_es_dn = self.model.addVars(self.num_scenarios, self.T, name="R_es_dn")  #es的下调频容量bids(es1负责)

        # 2. 定义场景相关变量（主要是先创建空字典 便于后续定义）
        # P_ev_uc[w, t, n]: 不可控EV在场景w、时刻t、第n辆车的实际充电功率（MW）
        # P_ev_cc[w, t, n]: 可控EV在场景w、时刻t、第n辆车的实际充电功率（MW）
        # P_ev0_uc[w, t, n]: 不可控EV在场景w、时刻t、第n辆车的日前能量投标（MW）
        # P_ev0_cc[w, t, n]: 可控EV在场景w、时刻t、第n辆车的日前能量投标（MW）
        P_ev_uc, P_ev_cc, P_ev0_uc, P_ev0_cc = {}, {}, {}, {} 
        # R_ev_up_i[w, t, n]: 可控EV在场景w、时刻t、第n辆车的上调频容量（MW）
        # R_ev_dn_i[w, t, n]: 可控EV在场景w、时刻t、第n辆车的下调频容量（MW）
        # soc[w, t, n]: EV在场景w、时刻t、第n辆车的荷电状态（SOC）
        R_ev_up_i, R_ev_dn_i, soc = {}, {}, {}
        # P_es[w, t]: 储能系统在场景w、时刻t的总净功率（MW）
        P_es, P_es_ch, P_es_dis, E_es = {}, {}, {}, {}
        # P_es1[w, t]: ES1子系统在场景w、时刻t的净功率（MW）
        P_es1, P_es2, P_es1_ch, P_es1_dis, E_es1, E_es2 = {}, {}, {}, {}, {}, {}
        # mu_es_ch[w, t]: 储能系统在场景w、时刻t的充电辅助变量（如充电开关/比例）
        # mu_es_dis[w, t]: 储能系统在场景w、时刻t的放电辅助变量（如放电开关/比例）
        # P_es2_ch[w, t]: ES2在场景w、时刻t的充电功率（MW）
        # P_es2_dis[w, t]: ES2在场景w、时刻t的放电功率（MW）
        mu_es_ch, mu_es_dis, P_es2_ch, P_es2_dis = {}, {}, {}, {}

        # 3. CVaR相关变量
        sigma = self.model.addVar(name="sigma")
        phi = self.model.addVars(self.num_scenarios, name="phi")

        # 4. 电池退化相关变量
        dod = self.model.addVar(name="dod")
        life_cycle = self.model.addVar(name="life_cycle")

        # 电池更换成本（3×10^5 $/MWh）
        C_es = 3e5
        # 电池寿命循环次数
        L_es = 4000  # 典型锂电池寿命
        # 最大允许DOD
        d_es = self.dod_max  # 0.9
        # 充放电效率
        eta_ch = 0.95
        eta_dis = 0.95

        # 5. 按场景、时间、车辆定义变量
        for w in range(self.num_scenarios):      #每个场景作为最基础的研究范围 所以先从场景开始循环
            ev_profiles = self.reduced_ev_scenarios[w]
            N = len(ev_profiles)
            for t in range(self.T):
                # ES变量
                # P_es[w, t]: es在场景w、时刻t的总净功率（MW）
                P_es[w, t] = self.model.addVar(name=f"P_es_{w}_{t}")
                # P_es_ch[w, t]: es在场景w、时刻t的充电功率（MW）
                P_es_ch[w, t] = self.model.addVar(name=f"P_es_ch_{w}_{t}")
                # P_es_dis[w, t]: es在场景w、时刻t的放电功率（MW）
                P_es_dis[w, t] = self.model.addVar(name=f"P_es_dis_{w}_{t}")
                # E_es[w, t]: es在场景w、时刻t的能量状态（MWh）
                E_es[w, t] = self.model.addVar(name=f"E_es_{w}_{t}")
                # P_es1[w, t]: ES1子系统在场景w、时刻t的净功率（MW）
                P_es1[w, t] = self.model.addVar(name=f"P_es1_{w}_{t}")
                # P_es2[w, t]: ES2子系统在场景w、时刻t的净功率（MW）
                P_es2[w, t] = self.model.addVar(name=f"P_es2_{w}_{t}")
                # P_es1_ch[w, t]: ES1在场景w、时刻t的充电功率（MW）
                P_es1_ch[w, t] = self.model.addVar(name=f"P_es1_ch_{w}_{t}")
                # P_es1_dis[w, t]: ES1在场景w、时刻t的放电功率（MW）
                P_es1_dis[w, t] = self.model.addVar(name=f"P_es1_dis_{w}_{t}")
                # E_es1[w, t]: ES1在场景w、时刻t的能量状态（MWh）
                E_es1[w, t] = self.model.addVar(name=f"E_es1_{w}_{t}")
                # E_es2[w, t]: ES2在场景w、时刻t的能量状态（MWh）
                E_es2[w, t] = self.model.addVar(name=f"E_es2_{w}_{t}")
                # mu_es_ch[w, t]: 储能系统在场景w、时刻t的充电辅助变量（如充电开关/比例）
                mu_es_ch[w, t] = self.model.addVar(name=f"mu_es_ch_{w}_{t}")
                # mu_es_dis[w, t]: 储能系统在场景w、时刻t的放电辅助变量（如放电开关/比例）
                mu_es_dis[w, t] = self.model.addVar(name=f"mu_es_dis_{w}_{t}")
                # P_es2_ch[w, t]: ES2在场景w、时刻t的充电功率（MW）
                P_es2_ch[w, t] = self.model.addVar(name=f"P_es2_ch_{w}_{t}")
                # P_es2_dis[w, t]: ES2在场景w、时刻t的放电功率（MW）
                P_es2_dis[w, t] = self.model.addVar(name=f"P_es2_dis_{w}_{t}")
                for n in range(N):
                    # P_ev_uc[w, t, n]: 不可控EV在场景w、时刻t、第n辆车的实际充电功率（MW）
                    P_ev_uc[w, t, n] = self.model.addVar(name=f"P_ev_uc_{w}_{t}_{n}")
                    # P_ev_cc[w, t, n]: 可控EV在场景w、时刻t、第n辆车的实际充电功率（MW）
                    P_ev_cc[w, t, n] = self.model.addVar(name=f"P_ev_cc_{w}_{t}_{n}")
                    # P_ev0_uc[w, t, n]: 不可控EV在场景w、时刻t、第n辆车的日前能量投标（MW）
                    P_ev0_uc[w, t, n] = self.model.addVar(name=f"P_ev0_uc_{w}_{t}_{n}")
                    # P_ev0_cc[w, t, n]: 可控EV在场景w、时刻t、第n辆车的日前能量投标（MW）
                    P_ev0_cc[w, t, n] = self.model.addVar(name=f"P_ev0_cc_{w}_{t}_{n}")
                    # R_ev_up_i[w, t, n]: 可控EV在场景w、时刻t、第n辆车的上调频容量（MW）
                    R_ev_up_i[w, t, n] = self.model.addVar(name=f"R_ev_up_i_{w}_{t}_{n}")
                    # R_ev_dn_i[w, t, n]: 可控EV在场景w、时刻t、第n辆车的下调频容量（MW）
                    R_ev_dn_i[w, t, n] = self.model.addVar(name=f"R_ev_dn_i_{w}_{t}_{n}")
                    # soc[w, t, n]: EV在场景w、时刻t、第n辆车的荷电状态（SOC）
                    soc[w, t, n] = self.model.addVar(name=f"soc_{w}_{t}_{n}")

        # 5.1 定义总最大功率和能量，并添加等式约束
        P_es_max = self.model.addVar(name="P_es_max")
        E_es_max = self.model.addVar(name="E_es_max")
        self.model.addConstr(P_es_max == self.P_es_max)
        self.model.addConstr(E_es_max == self.E_es_max)

        # 6. 添加约束
        for w in range(self.num_scenarios):
            ev_profiles = self.reduced_ev_scenarios[w]
            price_scenario = self.reduced_price_scenarios[w]
            agc_scenario = self.reduced_agc_scenarios[w]
            constraints.add_cc_ev_constraints(
                ev_profiles=ev_profiles,
                T=self.T,
                delta_t=self.delta_t,
                P_ev_cc=P_ev_cc,
                soc=soc,
                scenario_idx=w,
                P_ev0_cc=P_ev0_cc,
                R_ev_up_i=R_ev_up_i,
                R_ev_dn_i=R_ev_dn_i,
                R_ev_up=R_ev_up,
                R_ev_dn=R_ev_dn,
                K_dn=self.K_dn
            )
            constraints.add_uc_ev_constraints(
                ev_profiles=ev_profiles,
                T=self.T,
                P_ev_uc=P_ev_uc,
                scenario_idx=w
            )
            constraints.add_ev_fleet_aggregate_constraints(
                ev_profiles=ev_profiles,
                scenario_idx=w,
                T=self.T,
                P_ev_uc=P_ev_uc,
                P_ev_cc=P_ev_cc,
                P_ev0_uc=P_ev0_uc,
                P_ev0_cc=P_ev0_cc,
                P_ev_total={},
                P_ev0_total={}
            )
            constraints.add_es_constraints(
                T=self.T,
                scenario_idx=w,
                P_es=P_es,
                P_es_ch=P_es_ch,
                P_es_dis=P_es_dis,
                E_es=E_es,
                P_es1=P_es1,
                P_es2=P_es2,
                P_es1_ch=P_es1_ch,
                P_es1_dis=P_es1_dis,
                E_es1=E_es1,
                E_es2=E_es2,
                mu_es_ch=mu_es_ch,
                mu_es_dis=mu_es_dis,
                R_es_up=R_es_up,
                R_es_dn=R_es_dn,
                P_es0=P_es0,
                agc_up=agc_scenario['agc_up'],
                agc_dn=agc_scenario['agc_dn'],
                K_up=self.K_up,
                K_dn=self.K_dn,
                P_es_max=self.P_es_max,
                E_es_max=self.E_es_max,
                E_es_init=self.E_es_init,
                dod_max=self.dod_max,
                eta_ch=0.95,
                eta_dis=0.95,
                delta_t=self.delta_t,
                kappa=1.0,
                gamma=self.gamma
            )
            constraints.add_energy_balance_constraints(
                T=self.T,
                ev_profiles=ev_profiles,
                scenario_idx=w,
                P_ev0=P_ev0,
                P_es0=P_es0,
                R_ev_up=R_ev_up,
                R_ev_dn=R_ev_dn,
                agc_up=agc_scenario['agc_up'],
                agc_dn=agc_scenario['agc_dn'],
                P_ev_uc=P_ev_uc,
                P_ev_cc=P_ev_cc,
                P_es2_ch=P_es2_ch,
                P_es2_dis=P_es2_dis
            )

        # 7. 目标函数（期望利润）
        scenario_revenues = []
        pi = 1.0 / self.num_scenarios  #参考的论文里的piw 代表每个场景发生的权重 这里按概率均等
        for w in range(self.num_scenarios):
            ev_profiles = self.reduced_ev_scenarios[w]
            price_scenario = self.reduced_price_scenarios[w]
            agc_scenario = self.reduced_agc_scenarios[w]
            N = len(ev_profiles)
            revenue_w = gp.LinExpr()
            for t in range(self.T):
                
                # EV收益
                # 所有EV dam买电成本
                F_ev_buy = P_ev0[w, t] * price_scenario['dam_prices'].iloc[t] * self.delta_t
                # EV市场卖电收入
                F_ev_sell = gp.quicksum(
                    (P_ev_cc[w, t, n] + P_ev_uc[w, t, n]) * ev_profiles.iloc[n]['charging_price'] * self.delta_t
                    for n in range(N)
                )
                # EV充电收益
                F_ev_charging = F_ev_sell - F_ev_buy
                # EV调频投标收益
                F_ev_cap = (R_ev_up[w, t] * price_scenario['afrr_up_prices'].iloc[t] +
                            R_ev_dn[w, t] * price_scenario['afrr_dn_prices'].iloc[t]) * self.delta_t
                # EV调频激活收益
                F_ev_mil = (R_ev_up[w, t] + R_ev_dn[w, t]) * price_scenario['balancing_prices'].iloc[t] * self.delta_t  #这里先忽略了performance score和 mileage multiplier  因为都是1
                
                #EV deploy cost
                F_ev_deploy = (R_ev_up[w, t] * agc_scenario['agc_up'].iloc[t] - R_ev_dn[w, t] * agc_scenario['agc_dn'].iloc[t]) * price_scenario['rtm_prices'].iloc[t] * self.delta_t

                # ES收益
                # ES套利收益
                F_es_arb = - P_es0[w, t] * price_scenario['dam_prices'].iloc[t] * self.delta_t
                # ES调频投标收益
                F_es_cap = (R_es_up[w, t] * price_scenario['afrr_up_prices'].iloc[t] +
                            R_es_dn[w, t] * price_scenario['afrr_dn_prices'].iloc[t]) * self.delta_t
                # ES调频激活收益
                F_es_mil = (R_es_up[w, t] + R_es_dn[w, t]) * price_scenario['balancing_prices'].iloc[t] * self.delta_t

                #ES deploy cost
                F_es_deploy = (R_es_up[w, t] * agc_scenario['agc_up'].iloc[t] - R_es_dn[w, t] * agc_scenario['agc_dn'].iloc[t]) * price_scenario['rtm_prices'].iloc[t] * self.delta_t

                # 储能系统退化成本（支出项，需从总收益中扣除）
                F_es_deg = C_es * (P_es_ch[w, t] + P_es_dis[w, t]) / (2 * L_es * d_es * eta_ch * eta_dis)

                revenue_w += F_ev_charging + F_ev_cap + F_ev_mil - F_ev_deploy + F_es_arb + F_es_cap + F_es_mil - F_es_deploy - F_es_deg
            scenario_revenues.append(revenue_w)

        # CVaR约束
        constraints.add_cvar_constraints(
            num_scenarios=self.num_scenarios,
            sigma=sigma,
            phi=phi,
            f=scenario_revenues
        )

        # 目标函数：max (1-beta) * E[f_w] + beta * CVaR
        expectation = gp.quicksum(pi * scenario_revenues[w] for w in range(self.num_scenarios))
        cvar_term = sigma - (1/(1-self.alpha)) * gp.quicksum(pi * phi[w] for w in range(self.num_scenarios))
        self.model.setObjective((1-self.beta)*expectation + self.beta*cvar_term, GRB.MAXIMIZE)
        
    def solve(self):
        self.build_model()
        self.model.optimize()
        if self.model.status == GRB.OPTIMAL:
            return self.get_results()
        else:
            raise Exception("优化问题无解")
            
    def get_results(self) -> Dict:
        results = {
            "objective_value": self.model.objVal,
            
        }
        return results 