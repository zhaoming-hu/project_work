#  Risk-averse frequency regulation strategy of electric vehicle aggregator considering multiple uncertainties

本文主要介绍各个脚本及运行指南

---

## main脚本
主要包含两个文件：main.py 和 main_test.py 对于case 1 2 3 请直接运行main.py, case4请运行main_test.py

## 模型脚本
包含四个脚本，分别是case1_model.py, case2_model.py, case3_model.py, case4_model.py

## 约束脚本
constraints.py，不同case的约束用class进行了区分

## 数据导入及生成脚本
data_loader.py 这个脚本主要用于导入外部csv文件数据。scenario_genarator.py脚本主要用于处理数据及生成场景，包含两个class，第一个class用于case1,2,3，另一个class用于case4
