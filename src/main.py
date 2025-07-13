import pandas as pd
from data_loader import DataLoader
from scenario_generator import ScenarioGenerator
from optimization_model import V2GOptimizationModelCase1
from optimization_model import V2GOptimizationModelCase3

def main():
    # 1. 加载基础数据
    data_loader = DataLoader(data_dir="../data")
    ev_profiles = data_loader.load_ev_profiles(num_evs=2, discount=0.2, charging_price=180, seed=42)
    rtm_price = data_loader.load_rtm_price()
    dam_price = data_loader.load_dam_price()
    agc_signal = data_loader.load_agc_signal()
    capacity_price = data_loader.load_capacity_price()
    balancing_price = data_loader.load_balancing_energy_and_price()

    # 3. 生成初始场景
    scenario_gen = ScenarioGenerator(num_scenarios=4, num_clusters=2, seed=42)
    ev_scenarios = scenario_gen.generate_ev_scenarios(ev_profiles=ev_profiles)
    price_scenarios = scenario_gen.generate_price_scenarios(
        dam_prices=dam_price,
        rtm_prices=rtm_price,
        capacity_price=capacity_price,
        balancing_prices=balancing_price,
        T=96
    )
    agc_scenarios = scenario_gen.generate_agc_scenarios(agc_signals=agc_signal, T=96)[0]
    K_dn = scenario_gen.generate_agc_scenarios(agc_signals=agc_signal, T=96)[1]
    K_up = scenario_gen.generate_agc_scenarios(agc_signals=agc_signal, T=96)[2]

    # 4. K-means缩减场景
    reduced_ev_scenarios, reduced_price_scenarios, reduced_agc_scenarios = scenario_gen.reduce_scenarios(
        ev_scenarios=ev_scenarios,
        price_scenarios=price_scenarios,
        agc_scenarios=agc_scenarios
    )

    # 5. 构建并求解优化模型
    model = V2GOptimizationModelCase1(
        reduced_ev_scenarios=reduced_ev_scenarios,
        reduced_price_scenarios=reduced_price_scenarios,
        reduced_agc_scenarios=reduced_agc_scenarios,
        K_dn=K_dn,
        K_up=K_up

    )
    results = model.solve()

    # 6. 输出结果
    print("优化目标值：", results["objective_value"])
    

if __name__ == "__main__":
    main()