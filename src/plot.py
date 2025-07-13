import matplotlib.pyplot as plt
import pandas as pd
from data_loader import DataLoader
from scenario_generator import ScenarioGenerator
import numpy as np

def plotting():
    # 1. 加载基础数据
    data_loader = DataLoader(data_dir="../data")
    ev_profiles = data_loader.load_ev_profiles(num_evs=5, discount=0.2, charging_price=180, seed=42)
    rtm_price = data_loader.load_rtm_price()
    dam_price = data_loader.load_dam_price()
    agc_signal = data_loader.load_agc_signal()
    capacity_price = data_loader.load_capacity_price()
    balancing_price = data_loader.load_balancing_energy_and_price()

    # 2. 生成初始场景
    scenario_gen = ScenarioGenerator(num_scenarios=10, num_clusters=2, seed=42)
    ev_scenarios = scenario_gen.generate_ev_scenarios(ev_profiles=ev_profiles)
    price_scenarios = scenario_gen.generate_price_scenarios(
        dam_prices=dam_price,
        rtm_prices=rtm_price,
        capacity_price=capacity_price,
        balancing_prices=balancing_price,
        T=96
    )
    agc_scenarios_all, _, _ = scenario_gen.generate_agc_scenarios(
        agc_signals=agc_signal,
        T=96
    )

    # 3. 场景缩减（聚类降维）
    reduced_ev_scenarios, reduced_price_scenarios, reduced_agc_scenarios = scenario_gen.reduce_scenarios(
        ev_scenarios=ev_scenarios,
        price_scenarios=price_scenarios,
        agc_scenarios=agc_scenarios_all
    )

    # 4. 分别绘图
    for i, (agc_df, ev_df) in enumerate(zip(reduced_agc_scenarios, reduced_ev_scenarios)):
        T = 96

        # # --- 画 AGC ---
        # plt.figure(figsize=(12, 5))
        # plt.plot(agc_df['agc_up'], label='AGC Up')
        # plt.plot(agc_df['agc_dn'], label='AGC Down')
        # plt.xlabel('Time Slot (15-min steps)')
        # plt.ylabel('AGC Signal (positive)')
        # plt.title(f'Reduced AGC Scenario #{i}')
        # plt.legend()
        # plt.grid(True)
        # plt.tight_layout()
        # plt.savefig(f"agc_plot_scenario_{i}.png")
        # plt.show()
        #
        # # --- 画 EV 到达/离开时间为时间步上的水平线 ---
        # plt.figure(figsize=(12, 5))
        # for idx, row in ev_df.iterrows():
        #     arrival_step = int(row['arrival_time'] * 4)
        #     departure_step = int(row['departure_time'] * 4)
        #     plt.hlines(row['arrival_time'], 0, T-1, colors='blue', linestyles='--', label='Arrival Time' if idx == 0 else "")
        #     plt.hlines(row['departure_time'], 0, T-1, colors='orange', linestyles='--', label='Departure Time' if idx == 0 else "")
        # plt.xlabel('Time Step (0–95)')
        # plt.ylabel('Time [h]')
        # plt.title(f'EV Arrival & Departure Times (Scenario #{i})')
        # plt.legend()
        # plt.grid(True)
        # plt.tight_layout()
        # plt.savefig(f"ev_time_line_scenario_{i}.png")
        # plt.show()
        #
        # # --- 画 EV SOC 到达/离开为时间步上的水平线 ---
        # plt.figure(figsize=(12, 5))
        # for idx, row in ev_df.iterrows():
        #     plt.hlines(row['soc_arrival'], 0, T-1, colors='green', linestyles='-', label='SOC Arrival' if idx == 0 else "")
        #     plt.hlines(row['soc_departure'], 0, T-1, colors='red', linestyles='--', label='SOC Departure' if idx == 0 else "")
        # plt.xlabel('Time Step (0–95)')
        # plt.ylabel('SOC [0–1]')
        # plt.title(f'EV SOC Levels (Scenario #{i})')
        # plt.legend()
        # plt.grid(True)
        # plt.tight_layout()
        # plt.savefig(f"ev_soc_line_scenario_{i}.png")
        # plt.show()

        # --- 画价格曲线 ---
        price_df = reduced_price_scenarios[i]
        plt.figure(figsize=(12, 6))
        plt.plot(price_df['dam_prices'], label='DAM Price')
        plt.plot(price_df['rtm_prices'], label='RTM Price')
        plt.plot(price_df['afrr_up_cap_prices'], label='aFRR Up Capacity Price')
        plt.plot(price_df['afrr_dn_cap_prices'], label='aFRR Down Capacity Price')
        plt.plot(price_df['balancing_prices'], label='Balancing Energy Price')
        plt.xlabel('Time Step (0–95)')
        plt.ylabel('Price (€/MWh)')
        plt.title(f'Market Prices Over Time (Scenario #{i})')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"price_plot_scenario_{i}.png")
        plt.show()



if __name__ == "__main__":
    plotting()
