import pandas as pd
import os
from data_loader import DataLoader
from scenario_generator import ScenarioGenerator
from optimization_model import V2GOptimizationModelCase1
from optimization_model import V2GOptimizationModelCase2
from optimization_model import V2GOptimizationModelCase3
from optimization_model import V2GOptimizationModelCase4
from plot import plot_ev_bids_and_price
from plot import plot_ev_regulation_bids

def main():
    # 1. 加载基础数据
    data_loader = DataLoader(data_dir="../data")
    ev_profiles = data_loader.load_ev_profiles(num_evs=400, discount=0.2, charging_price=180, seed=42, use_timeslot=True)       
    rtm_price = data_loader.load_rtm_price()
    dam_price = data_loader.load_dam_price()
    agc_signal = data_loader.load_agc_signal()
    capacity_price = data_loader.load_capacity_price()
    balancing_price = data_loader.load_balancing_energy_and_price()

    # 2. 生成初始场景，增加场景数量
    scenario_gen = ScenarioGenerator(num_scenarios=10, num_clusters=4, seed=42)
    ev_scenarios = scenario_gen.generate_ev_scenarios(ev_profiles=ev_profiles)    
    price_scenarios = scenario_gen.generate_price_scenarios(
        dam_prices=dam_price,
        rtm_prices=rtm_price,
        capacity_price=capacity_price,
        balancing_prices=balancing_price,
        T=96
    )
    # 获取AGC场景和每个时间段的容量预留值
    agc_scenarios, capacity_reserves = scenario_gen.generate_agc_scenarios(
        agc_signals=agc_signal, 
        T=96
    )

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
        # 设置CVaR参数
        beta = 0.95  # 期望收益和CVaR的权重系数 越大越保守
        alpha = 0.5  # 置信水平 代表利润大于sigma的概率
        
        # Case1模型，取消下面的注释 
        # model = V2GOptimizationModelCase1(
        #     reduced_ev_scenarios=reduced_ev_scenarios,
        #     reduced_price_scenarios=reduced_price_scenarios,
        #     reduced_agc_scenarios=reduced_agc_scenarios,
        #     capacity_reserves=capacity_reserves,  # 传递每个时段的容量预留值
        #     beta=beta,  # 传递CVaR置信水平
        #     alpha=alpha  # 传递CVaR权重系数
        # )

        # Case2模型，取消下面的注释 
        # model = V2GOptimizationModelCase2(
        #     reduced_ev_scenarios=reduced_ev_scenarios,
        #     reduced_price_scenarios=reduced_price_scenarios,
        #     reduced_agc_scenarios=reduced_agc_scenarios,
        #     capacity_reserves=capacity_reserves,  # 传递每个时段的容量预留值
        #     beta=beta,  # 传递CVaR置信水平
        #     alpha=alpha  # 传递CVaR权重系数
        # )
        
        # Case3模型，取消下面的注释
        model = V2GOptimizationModelCase3(
            reduced_ev_scenarios=reduced_ev_scenarios,
            reduced_price_scenarios=reduced_price_scenarios,
            reduced_agc_scenarios=reduced_agc_scenarios,
            capacity_reserves=capacity_reserves,  # 传递每个时段的容量预留值
            beta=beta,
            alpha=alpha
        )

        # Case4模型，取消下面的注释
        model = V2GOptimizationModelCase4(
            reduced_ev_scenarios=reduced_ev_scenarios,
            reduced_price_scenarios=reduced_price_scenarios,
            reduced_agc_scenarios=reduced_agc_scenarios,
            capacity_reserves=capacity_reserves,  # 传递每个时段的容量预留值
            beta=beta,
            alpha=alpha
        )
        
        print("正在构建和求解模型...")
        try:
            results = model.solve()
            print("===== 优化结果 =====")
            print(f"目标函数值: {results['objective_value']:.2f}")
            print(f"期望收益: {results['expected_profit']:.2f}")
            print(f"CVaR值: {results['cvar_value']:.2f}")
            print("\n----- EV收益与成本 -----")
            print(f"EV调频容量收入: {results['ev_cap_revenue']:.2f}")
            print(f"EV调频里程收入: {results['ev_mil_revenue']:.2f}")
            print(f"EV充电收入: {results['ev_charging_revenue']:.2f}")
            print(f"EV日前市场能源成本: {results['ev_dam_cost']:.2f}")
            print(f"EV调频部署成本: {results['ev_deploy_cost']:.2f}")
            
            print("\n----- ES收益与成本 -----")
            print(f"ES调频容量收入: {results.get('es_cap_revenue', 0):.2f}")
            print(f"ES调频里程收入: {results.get('es_mil_revenue', 0):.2f}")
            print(f"ES能量市场套利收入: {results.get('es_arb_revenue', 0):.2f}")
            print(f"ES调频部署成本: {results.get('es_deploy_cost', 0):.2f}")
            print(f"ES退化成本: {results['es_deg_cost']:.2f}")
            
            if 'P_es1_max' in results:
                print("\n----- 储能分配 -----")
                print(f"ES1最大功率: {results['P_es1_max']:.2f} MW")
                print(f"ES2最大功率: {results['P_es2_max']:.2f} MW")
                print(f"ES1最大容量: {results['E_es1_max']:.2f} MWh")
                print(f"ES2最大容量: {results['E_es2_max']:.2f} MWh")
            
        except Exception as solve_error:
            print(f"模型求解过程中出现错误：{solve_error}")
            import traceback
            traceback.print_exc()
            results = None
            
            # 创建输出目录
            output_dir = "plots"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # 绘制EV bids和能源价格图像
            try:
                print("\n正在绘制EV bids和能源价格图像...")
                plot_ev_bids_and_price(
                    model, 
                    dam_price,
                    output_path=os.path.join(output_dir, "ev_bids_and_price.png")
                )
                print(f"图像已保存到 {os.path.join(output_dir, 'ev_bids_and_price.png')}")
                
                # 绘制EV调频投标图
                print("\n正在绘制EV调频投标图像...")
                plot_ev_regulation_bids(
                    model,
                    output_path=os.path.join(output_dir, "ev_regulation_bids.png"),
                    case_name="Case I"
                )
                print(f"图像已保存到 {os.path.join(output_dir, 'ev_regulation_bids.png')}")
            except Exception as e:
                print(f"\n绘图过程中出现错误：{e}")

    except Exception as e:
        print(f"\n优化过程中出现错误：{e}")


if __name__ == "__main__":
    main()