import pandas as pd
from data_loader import DataLoader
from scenario_generator import ScenarioGenerator
from optimization_model import V2GOptimizationModel

def main():
    # 1. 加载基础数据
    data_loader = DataLoader(data_dir="data")
    ev_profiles = data_loader.load_ev_profiles(num_evs=400, discount=0.2, charging_price=180, seed=42)
    dam_price = data_loader.load_dam_price()
    rtm_price = data_loader.load_rtm_price()
    agc_signal, K_up, K_dn = data_loader.load_agc_signal()
    capacity_price = data_loader.load_capacity_price()
    balancing_price = data_loader.load_balancing_energy_and_price()

    # 2. 统一时间索引   
    T = 96  # 24小时*4=96个15分钟
    dam_prices = dam_price['price'].iloc[:T].reset_index(drop=True)
    rtm_prices = rtm_price['Price'].iloc[:T].reset_index(drop=True)
    afrr_up_prices = capacity_price['afrr_up_cap_price'].iloc[:T].reset_index(drop=True)
    afrr_dn_prices = capacity_price['afrr_dn_cap_price'].iloc[:T].reset_index(drop=True)
    balancing_prices = balancing_price['balancing_price'].iloc[:T].reset_index(drop=True)
    agc_signals = agc_signal.iloc[:T].reset_index(drop=True)

    # 3. 生成初始场景
    scenario_gen = ScenarioGenerator(num_scenarios=1000, num_clusters=48, seed=42)
    ev_scenarios = scenario_gen.generate_ev_scenarios(ev_profiles=ev_profiles, T=T)
    price_scenarios = scenario_gen.generate_price_scenarios(
        dam_prices=dam_prices,
        rtm_prices=rtm_prices,
        afrr_up_prices=afrr_up_prices,
        afrr_dn_prices=afrr_dn_prices,
        balancing_prices=balancing_prices,
        T=T
    )
    agc_scenarios = scenario_gen.generate_agc_scenarios(agc_signals=agc_signals, T=T)

    # 4. K-means缩减场景
    reduced_ev_scenarios, reduced_price_scenarios, reduced_agc_scenarios = scenario_gen.reduce_scenarios(
        ev_scenarios=ev_scenarios,
        price_scenarios=price_scenarios,
        agc_scenarios=agc_scenarios
    )

    # 5. 构建并求解优化模型
    model = V2GOptimizationModel(
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