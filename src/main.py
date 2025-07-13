import pandas as pd
import os
from data_loader import DataLoader
from scenario_generator import ScenarioGenerator
from optimization_model import V2GOptimizationModelCase1
from optimization_model import V2GOptimizationModelCase3
from plot import plot_ev_bids_and_price

def main():
    # 1. 加载基础数据
    data_loader = DataLoader(data_dir="../data")
    ev_profiles = data_loader.load_ev_profiles(num_evs=6, discount=0.2, charging_price=180, seed=42)       
    rtm_price = data_loader.load_rtm_price()
    dam_price = data_loader.load_dam_price()
    agc_signal = data_loader.load_agc_signal()
    capacity_price = data_loader.load_capacity_price()
    balancing_price = data_loader.load_balancing_energy_and_price()

    # 2. 生成初始场景，增加场景数量
    scenario_gen = ScenarioGenerator(num_scenarios=8, num_clusters=4, seed=42)
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

    # 3. K-means缩减场景
    reduced_ev_scenarios, reduced_price_scenarios, reduced_agc_scenarios = scenario_gen.reduce_scenarios(
        ev_scenarios=ev_scenarios,
        price_scenarios=price_scenarios,
        agc_scenarios=agc_scenarios
    )
    
    # 输出缩减后场景的信息
    print("\n===== K-Means 缩减后场景信息 =====")
    print(f"场景数量: {len(reduced_ev_scenarios)}")
    
    for i, scenario in enumerate(reduced_ev_scenarios):
        day_count = sum(scenario['charging_type'] == 'day')
        night_count = sum(scenario['charging_type'] == 'night')
        print(f"场景 {i}: 白天充电EV数量: {day_count}, 控制类型: {scenario[scenario['charging_type'] == 'day']['ev_type'].value_counts().to_dict()}")
        print(f"场景 {i}: 夜间充电EV数量: {night_count}, 控制类型: {scenario[scenario['charging_type'] == 'night']['ev_type'].value_counts().to_dict()}")
    
    # 4. 构建并求解优化模型
    try:
        model = V2GOptimizationModelCase1(
            reduced_ev_scenarios=reduced_ev_scenarios,
            reduced_price_scenarios=reduced_price_scenarios,
            reduced_agc_scenarios=reduced_agc_scenarios,
            K_dn=K_dn,
            K_up=K_up
        )
        results = model.solve()

        # 5. 输出结果
        if results:
            print("\n===== 优化结果 =====")
            print("优化目标值：", results["objective_value"])
            print("\n===== 收益和成本明细 =====")
            print(f"EV调频容量收入: {results['ev_cap_revenue']:.2f} €")
            print(f"EV调频里程收入: {results['ev_mil_revenue']:.2f} €")
            print(f"EV充电收入: {results['ev_charging_revenue']:.2f} €")
            print(f"日前市场能源成本: {results['ev_dam_cost']:.2f} €")
            print(f"调频部署成本: {results['ev_deploy_cost']:.2f} €")
            print(f"储能退化成本: {results['es_deg_cost']:.2f} €")
            
            # 计算净收益和比例
            total_revenue = results['ev_cap_revenue'] + results['ev_mil_revenue'] + results['ev_charging_revenue']
            total_cost = results['ev_dam_cost'] + results['ev_deploy_cost'] + results['es_deg_cost']
            print(f"\n总收入: {total_revenue:.2f} €")
            print(f"总成本: {total_cost:.2f} €")
            print(f"净收益: {total_revenue - total_cost:.2f} €")
            
            # 绘制EV bids和能源价格图像
            try:
                print("\n正在绘制EV bids和能源价格图像...")
                output_dir = "plots"
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                # 确保dam_price是数值型数据
                dam_price_numeric = dam_price.copy()
                if 'timeslot' in dam_price_numeric.columns and not pd.api.types.is_numeric_dtype(dam_price_numeric['timeslot']):
                    # 如果timeslot不是数字类型，转换为小时数(0-23)
                    try:
                        # 尝试提取小时数
                        dam_price_numeric['hour'] = pd.to_numeric(dam_price_numeric['timeslot']) % 24
                    except:
                        # 如果无法转换，创建一个序号列
                        dam_price_numeric['hour'] = range(len(dam_price_numeric))
                
                plot_ev_bids_and_price(
                    model, 
                    dam_price_numeric, 
                    output_path=os.path.join(output_dir, "ev_bids_and_price.png")
                )
                print(f"图像已保存到 {os.path.join(output_dir, 'ev_bids_and_price.png')}")
            except Exception as e:
                print(f"\n绘图过程中出现错误：{e}")
        else:
            print("\n===== 优化结果 =====")
            print("优化模型无法找到可行解")
    except Exception as e:
        print(f"\n优化过程中出现错误：{e}")



if __name__ == "__main__":
    main()