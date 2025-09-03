import pandas as pd
import os
import numpy as np
import random
import gc
from data_loader import DataLoader
from scenario_generator import ScenarioGenerator
from case1_model import V2GOptimizationModelCase1
from case2_model import V2GOptimizationModelCase2
from case3_model import V2GOptimizationModelCase3
from case4_model import V2GOptimizationModelCase4
from plot import plot_ev_bids_and_price
from plot import plot_ev_regulation_bids
from plot import plot_es_bids_and_price
from plot import plot_es_regulation_bids
from plot import plot_es_energy_change
from plot import plot_ev_reg_bids_and_capacity_price
from plot import plot_agc_signal_both


def get_case_selection():
    """获取用户选择的case"""
    while True:
        print("\n===== V2G优化模型案例选择 =====")
        print("1. Case 1: EV充电和调频 ES backup")
        print("2. Case 2: EV充电和调频 + ES1套利 ES2 backup")
        print("3. Case 3: EV充电和调频 + ES1套利和调频 ES2 backup")
        print("4. Case 4: EV充电和调频 + ES1套利和调频 ES2 backup 无容量预留")
        print("请选择要运行的案例 (1-4):")
        
        try:
            choice = int(input().strip())
            if choice in [1, 2, 3, 4]:
                return choice
            else:
                print("无效选择，请输入1-4之间的数字")
        except ValueError:
            print("输入无效，请输入数字")

def create_model(case, reduced_ev_scenarios, reduced_price_scenarios, reduced_agc_scenarios, capacity_reserves, beta, alpha):
    """根据case创建对应的模型"""
    model_classes = {
        1: V2GOptimizationModelCase1,
        2: V2GOptimizationModelCase2,
        3: V2GOptimizationModelCase3,
        4: V2GOptimizationModelCase4
    }
    
    if case not in model_classes:
        raise ValueError(f"不支持的案例: {case}. 支持的案例: {list(model_classes.keys())}")
    
    model_class = model_classes[case]
    try:
        return model_class(
            case=case,
            reduced_ev_scenarios=reduced_ev_scenarios,
            reduced_price_scenarios=reduced_price_scenarios,
            reduced_agc_scenarios=reduced_agc_scenarios,
            capacity_reserves=capacity_reserves,
            beta=beta,
            alpha=alpha
        )
    except Exception as e:
        raise RuntimeError(f"创建 Case {case} 模型时出错: {str(e)}")

def main():
    # 1. 获取用户选择的case
    selected_case = get_case_selection()
    print(f"\n您选择了 Case {selected_case}")
    
    # 2. 加载基础数据
    print("\n正在加载基础数据...")
    try:
        seed = 42
        data_loader = DataLoader(data_dir="../data")
        # 由场景生成器统一生成EV基准数据（使用同一seed）
        scenario_gen = ScenarioGenerator(num_scenarios=100, num_clusters=2, seed=seed)
        ev_profiles = scenario_gen.generate_base_ev_profiles(num_evs=10, discount=0.2, charging_price=180, use_timeslot=True)
        rtm_price = data_loader.load_rtm_price()
        dam_price = data_loader.load_dam_price()
        agc_signal = data_loader.load_agc_signal()
        capacity_price = data_loader.load_capacity_price()
        balancing_price = data_loader.load_balancing_energy_and_price()
        mileage_multiplier = data_loader.load_multiplier()
        print("基础数据加载完成")
    except Exception as e:
        print(f"加载基础数据时出错: {e}")
        print("请检查数据文件是否存在于 ../data 目录中")
        return

    # 3. 生成初始场景，增加场景数量
    print("正在生成场景...")
    try:
        ev_scenarios = scenario_gen.generate_ev_scenarios(ev_profiles=ev_profiles)    
        price_scenarios = scenario_gen.generate_price_scenarios(
            dam_prices=dam_price,
            rtm_prices=rtm_price,
            capacity_price=capacity_price,
            balancing_prices=balancing_price,
            mileage_multiplier=mileage_multiplier,
            T=96
        )
        # 获取AGC场景和每个时间段的容量预留值
        agc_scenarios, capacity_reserves = scenario_gen.generate_agc_scenarios(
            agc_signals=agc_signal, 
            T=96
        )
        print("场景生成完成")

    except Exception as e:
        print(f"生成场景时出错: {e}")
        return

    # 4. K-means缩减场景
    print("正在缩减场景...")
    try:
        reduced_ev_scenarios, reduced_price_scenarios, reduced_agc_scenarios = scenario_gen.reduce_scenarios(
            ev_scenarios=ev_scenarios,
            price_scenarios=price_scenarios,
            agc_scenarios=agc_scenarios
        )
        print("场景缩减完成")
    except Exception as e:
        print(f"缩减场景时出错: {e}")
        return

    # 打印缩减后场景0的AGC信号和容量预留K_dn，K_up
    # try:
    #     if len(reduced_agc_scenarios) > 0:
    #         print("\n===== 场景0的AGC信号和容量预留K_dn K_up（缩减后） =====")
    #         agc0 = reduced_agc_scenarios[0]
    #         print(f"长度: {len(agc0)}")
    #         print("agc_up:", agc0['agc_up'].tolist())
    #         print("agc_dn:", agc0['agc_dn'].tolist())
    #         print("K_up:", capacity_reserves['K_up'].tolist())
    #         print("K_dn:", capacity_reserves['K_dn'].tolist())
    #     else:
    #         print("未找到缩减后的AGC场景，无法打印场景0的AGC信号")
    # except Exception as e:
    #     print(f"打印AGC信号时出错: {e}")
    
    # 输出缩减后场景的信息
    print("\n===== K-Means 缩减后场景信息 =====")
    print(f"场景数量: {len(reduced_ev_scenarios)}")
    
    for i, scenario in enumerate(reduced_ev_scenarios):
        day_count = sum(scenario['charging_type'] == 'day')
        night_count = sum(scenario['charging_type'] == 'night')
        print(f"场景 {i}: 白天充电EV数量: {day_count}, 控制类型: {scenario[scenario['charging_type'] == 'day']['ev_type'].value_counts().to_dict()}")
        print(f"场景 {i}: 夜间充电EV数量: {night_count}, 控制类型: {scenario[scenario['charging_type'] == 'night']['ev_type'].value_counts().to_dict()}")
    
    # 5. 构建并求解优化模型
    try:
        # 设置CVaR参数
        beta = 0.9 # 期望收益和CVaR的权重系数 越大越保守
        alpha = 0.5  # 置信水平 代表利润大于sigma的概率

        # 创建对应的模型
        print(f"\n正在创建 Case {selected_case} 模型...")
        model = create_model(
            case=selected_case,
            reduced_ev_scenarios=reduced_ev_scenarios,
            reduced_price_scenarios=reduced_price_scenarios,
            reduced_agc_scenarios=reduced_agc_scenarios,
            capacity_reserves=capacity_reserves,
            beta=beta,
            alpha=alpha
        )
        print("模型创建成功")
        
        print("正在构建和求解模型...")
        results = model.solve()
        
        if results is None:
            print("模型求解失败，未返回结果")
            return
            
        print("===== 优化结果 =====")
        print(f"目标函数值: {results['objective_value']:.2f}")
        print(f"期望收益: {results['expected_profit']:.2f}")
        print(f"CVaR值: {results['cvar_value']:.2f}")
        
        # 根据case显示相关结果
        # 所有case都有EV
        print("\n----- EV收益与成本 -----")
        print(f"EV调频容量收入: {results.get('ev_cap_revenue', 0):.2f}")
        print(f"EV调频里程收入: {results.get('ev_mil_revenue', 0):.2f}")
        print(f"EV充电收入: {results.get('ev_charging_revenue', 0):.2f}")
        print(f"EV买电成本: {results.get('ev_dam_cost', 0):.2f}")
        print(f"EV调频部署成本: {results.get('ev_deploy_cost', 0):.2f}")
        

        # print("-----------场景0 cc类EV的SOC-----------")
        # if 'soc_values' in results:
        #     soc_values = results['soc_values']
        #     if soc_values:
        #         # 获取cc类EV的数量
        #         scenario_0_ev = reduced_ev_scenarios[0]
        #         cc_evs = scenario_0_ev[scenario_0_ev['ev_type'] == 'cc'].reset_index(drop=True)
                
        #         for n in range(len(cc_evs)):
        #             soc_list = [f"{soc_values.get((0, t, n), 0):.3f}" for t in range(96)]
        #             print(f"cc_EV{n}: {soc_list}")
        #     else:
        #         print("SOC数据为空")
        # else:
        #     print("未找到SOC数据")

        # Case 2,3,4有ES
        if selected_case in [2, 3, 4]:
            print("\n----- ES收益与成本 -----")
            print(f"ES调频容量收入: {results.get('es_cap_revenue', 0):.2f}")
            print(f"ES调频里程收入: {results.get('es_mil_revenue', 0):.2f}")
            print(f"ES能量市场套利收入: {results.get('es_arb_revenue', 0):.2f}")
            print(f"ES调频部署成本: {results.get('es_deploy_cost', 0):.2f}")
            print(f"ES退化成本: {results.get('es_deg_cost', 0):.2f}")
        
        # 创建输出目录
        output_dir = "plots"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 绘制图像
        try:
            # 绘制EV相关图像 (所有case都有EV)
            print("\n正在绘制EV bids和能源价格图像...")
            plot_ev_bids_and_price(
                model, 
                dam_price,
                output_path=os.path.join(output_dir, "ev_bids_and_price.png")
            )
            print(f"图像已保存到 {os.path.join(output_dir, 'ev_bids_and_price.png')}")

            print("\n正在绘制EV调频投标和容量价格图像...")
            plot_ev_reg_bids_and_capacity_price(
                model,
                capacity_price,
                output_path=os.path.join(output_dir, "ev_regulation_bids_and_capacity_price.png"),
                case_name=f"Case {selected_case}"
            )
            print(f"图像已保存到 {os.path.join(output_dir, 'ev_regulation_bids_and_capacity_price.png')}")
            
            # 绘制ES套利图像 (Case 2, 3有ES套利)
            if selected_case in [2, 3]:
                print("\n正在绘制ES bids和能源价格图像...")
                plot_es_bids_and_price(
                    model,
                    dam_price,
                    output_path=os.path.join(output_dir, "es_bids_and_price.png")
                )
                print(f"图像已保存到 {os.path.join(output_dir, 'es_bids_and_price.png')}")

            if selected_case in [2, 3, 4]:
                print("\n正在绘制ES energy变化图像...")
                plot_es_energy_change(
                    model,
                    dam_price,
                    capacity_price,
                    output_path=os.path.join(output_dir, "es_energy_change.png"),
                    case_name=f"Case {selected_case}"
                )
                print(f"图像已保存到 {os.path.join(output_dir, 'es_energy_change.png')}")

            # 绘制ES调频图像 (Case 3有ES调频)
            if selected_case == 3:
                print("\n正在绘制ES调频投标图像...")
                plot_es_regulation_bids(
                    model,
                    output_path=os.path.join(output_dir, "es_regulation_bids.png"),
                    case_name=f"Case {selected_case}"
                )
                print(f"图像已保存到 {os.path.join(output_dir, 'es_regulation_bids.png')}")
        except Exception as e:
            print(f"\n绘图过程中出现错误：{e}")

        # # 绘制AGC信号图（仅保存上下合并图，使用缩减后的第一个AGC场景）
        # try:
        #     if len(reduced_agc_scenarios) > 0:
        #         print("\n正在绘制AGC信号图...")
        #         plot_agc_signal_both(
        #             reduced_agc_scenarios[0],
        #             output_path=os.path.join(output_dir, "agc_up_down_signal.png"),
        #             title='AGC Up & Down'
        #         )
        #         print("AGC图像已保存到 agc_up_down_signal.png")
        # except Exception as e:
        #     print(f"绘制AGC信号图时出现错误：{e}")


        if 'P_es1_max' in results:
            print("\n----- 储能分配 -----")
            print(f"ES1最大功率: {results['P_es1_max']:.6f} MW")
            print(f"ES2最大功率: {results['P_es2_max']:.6f} MW")
            print(f"ES1最大容量: {results['E_es1_max']:.6f} MWh")
            print(f"ES2最大容量: {results['E_es2_max']:.6f} MWh")
            
    except Exception as solve_error:
        print(f"模型求解过程中出现错误：{solve_error}")
        import traceback
        traceback.print_exc()
    finally:
    
        try:
            #关闭所有 Matplotlib 图像句柄，释放绘图缓存
            import matplotlib.pyplot as plt
            plt.close('all')
        except Exception:
            pass
        try:
            #释放 Gurobi 模型内存
            if 'model' in locals() and getattr(model, 'model', None) is not None:
                model.model.dispose()
                model.model = None
        except Exception as e:
            print(f"清理 Gurobi 模型时出错：{e}")
        gc.collect()

if __name__ == "__main__":
    main()