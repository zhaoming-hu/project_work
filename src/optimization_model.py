import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from data_loader import DataLoader
from constraints import V2GConstraints
from scenario_generator import ScenarioGenerator

class V2GOptimizationModel:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.model = None
        self.T = None  # 时间步数
        self.N = None  # 电动汽车数量
        self.beta = 0.95  # CVaR置信水平
        self.alpha = 0.5  # 风险厌恶系数
        self.scenario_generator = ScenarioGenerator()
        
    def _create_model(self):
        """创建Gurobi优化模型"""
        self.model = gp.Model("V2G_Optimization")
        
    def _add_variables(self):
        """添加决策变量"""
        # 日前市场投标变量
        self.P_ev0 = self.model.addVars(self.T, name="P_ev0")  # EV能量投标
        self.P_es0 = self.model.addVars(self.T, name="P_es0")  # ES能量投标
        self.R_ev_up = self.model.addVars(self.T, name="R_ev_up")  # EV上调容量
        self.R_ev_dn = self.model.addVars(self.T, name="R_ev_dn")  # EV下调容量
        self.R_es_up = self.model.addVars(self.T, name="R_es_up")  # ES上调容量
        self.R_es_dn = self.model.addVars(self.T, name="R_es_dn")  # ES下调容量
        
        # 实时市场调度变量
        self.P_ev = self.model.addVars(self.T, self.N, name="P_ev")  # EV实际充电功率
        self.P_es = self.model.addVars(self.T, name="P_es")  # ES实际充放电功率
        self.P_es1 = self.model.addVars(self.T, name="P_es1")  # ES1功率（调频）
        self.P_es2 = self.model.addVars(self.T, name="P_es2")  # ES2功率（补偿）
        
        # 二进制变量
        self.u_ev = self.model.addVars(self.T, self.N, vtype=GRB.BINARY, name="u_ev")  # EV充电状态
        self.u_es = self.model.addVars(self.T, vtype=GRB.BINARY, name="u_es")  # ES充电状态
        
        # CVaR相关变量
        self.eta = self.model.addVar(name="eta")
        self.s = self.model.addVars(self.T, name="s")  # 辅助变量
        
        # ES能量状态变量
        self.E_es = self.model.addVars(self.T, name="E_es")  # ES能量状态
        
        # EV SOC变量
        self.soc = self.model.addVars(self.T, self.N, name="soc")  # EV SOC状态
        
    def _set_objective(self):
        """设置目标函数"""
        # 日前市场收益
        dam_revenue = (
            sum(self.P_ev0[t] * self.dam_prices[t] for t in range(self.T)) +
            sum(self.P_es0[t] * self.dam_prices[t] for t in range(self.T)) +
            sum(self.R_ev_up[t] * self.afrr_up_prices[t] for t in range(self.T)) +
            sum(self.R_ev_dn[t] * self.afrr_dn_prices[t] for t in range(self.T)) +
            sum(self.R_es_up[t] * self.afrr_up_prices[t] for t in range(self.T)) +
            sum(self.R_es_dn[t] * self.afrr_dn_prices[t] for t in range(self.T))
        )
        
        # CVaR项
        cvar_term = self.eta - (1/(1-self.beta)) * sum(self.s[t] for t in range(self.T))
        
        # 总目标函数
        self.model.setObjective(dam_revenue + self.alpha * cvar_term, GRB.MAXIMIZE)
        
    def _generate_scenarios(self):
        """生成场景"""
        # 加载基础数据
        ev_profiles = self.data_loader.load_ev_profiles()
        dam_prices = self.data_loader.load_dam_price()['price']
        rtm_prices = self.data_loader.load_rtm_price()['price']
        afrr_prices = self.data_loader.load_capacity_price()
        agc_signals = self.data_loader.load_agc_signal()
        
        # 生成场景
        ev_scenarios = self.scenario_generator.generate_ev_scenarios(
            ev_profiles=ev_profiles,
            T=self.T
        )
        
        price_scenarios = self.scenario_generator.generate_price_scenarios(
            dam_prices=dam_prices,
            rtm_prices=rtm_prices,
            afrr_up_prices=afrr_prices['afrr_up_cap_price'],
            afrr_dn_prices=afrr_prices['afrr_dn_cap_price'],
            T=self.T
        )
        
        agc_scenarios = self.scenario_generator.generate_agc_scenarios(
            agc_signals=agc_signals,
            T=self.T
        )
        
        # 缩减场景
        self.reduced_ev_scenarios, self.reduced_price_scenarios, self.reduced_agc_scenarios = \
            self.scenario_generator.reduce_scenarios(
                ev_scenarios=ev_scenarios,
                price_scenarios=price_scenarios,
                agc_scenarios=agc_scenarios
            )
        
    def solve(self):
        """求解优化问题"""
        self._create_model()
        self._add_variables()
        self._generate_scenarios()
        
        # 准备参数字典
        params = {
            "T": self.T,
            "N": self.N,
            "P_ev": self.P_ev,
            "P_es": self.P_es,
            "P_es1": self.P_es1,
            "P_es2": self.P_es2,
            "P_ev0": self.P_ev0,
            "P_es0": self.P_es0,
            "R_ev_up": self.R_ev_up,
            "R_ev_dn": self.R_ev_dn,
            "R_es_up": self.R_es_up,
            "R_es_dn": self.R_es_dn,
            "u_ev": self.u_ev,
            "u_es": self.u_es,
            "eta": self.eta,
            "s": self.s,
            "E_es": self.E_es,
            "soc": self.soc,
            "ev_profiles": self.reduced_ev_scenarios[0],  # 使用第一个场景作为基础
            "dam_prices": self.reduced_price_scenarios[0]['dam_prices'],
            "rtm_prices": self.reduced_price_scenarios[0]['rtm_prices'],
            "afrr_up_prices": self.reduced_price_scenarios[0]['afrr_up_prices'],
            "afrr_dn_prices": self.reduced_price_scenarios[0]['afrr_dn_prices'],
            "agc_signals": self.reduced_agc_scenarios[0],
            "P_es_max": 10.0,  # 这些参数需要根据实际情况设置
            "E_es_max": 40.0,
            "eta_es": 0.95,
            "beta": self.beta,
            "revenue_scenarios": [self._calculate_scenario_revenue(i) for i in range(len(self.reduced_price_scenarios))]
        }
        
        # 创建约束管理器并添加约束
        constraints = V2GConstraints(self.model, params)
        constraints.add_all_constraints()
        
        # 设置目标函数
        self._set_objective()
        
        # 求解
        self.model.optimize()
        
        if self.model.status == GRB.OPTIMAL:
            return self._get_results()
        else:
            raise Exception("优化问题无解")
            
    def _calculate_scenario_revenue(self, scenario_idx: int) -> float:
        """计算给定场景的收益
        
        Args:
            scenario_idx: 场景索引
            
        Returns:
            float: 场景收益
        """
        price_scenario = self.reduced_price_scenarios[scenario_idx]
        revenue = 0
        
        # 计算日前市场收益
        for t in range(self.T):
            revenue += (
                self.P_ev0[t].x * price_scenario['dam_prices'][t] +
                self.P_es0[t].x * price_scenario['dam_prices'][t] +
                self.R_ev_up[t].x * price_scenario['afrr_up_prices'][t] +
                self.R_ev_dn[t].x * price_scenario['afrr_dn_prices'][t] +
                self.R_es_up[t].x * price_scenario['afrr_up_prices'][t] +
                self.R_es_dn[t].x * price_scenario['afrr_dn_prices'][t]
            )
            
        return revenue
            
    def _get_results(self) -> Dict:
        """获取优化结果"""
        return {
            "P_ev0": [self.P_ev0[t].x for t in range(self.T)],
            "P_es0": [self.P_es0[t].x for t in range(self.T)],
            "R_ev_up": [self.R_ev_up[t].x for t in range(self.T)],
            "R_ev_dn": [self.R_ev_dn[t].x for t in range(self.T)],
            "R_es_up": [self.R_es_up[t].x for t in range(self.T)],
            "R_es_dn": [self.R_es_dn[t].x for t in range(self.T)],
            "objective_value": self.model.objVal
        } 