import math
import random

import numpy as np
import pandas as pd
import folium
import os
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 添加这条可以让图形显示中文

#####对一系列装卸货点进行适当的路径规划，在满足约束条件（客户需求、车辆载重和容积、车型、车辆行驶里程、配送时间窗、配送中心数量等限制）
# 和目标最优化（路程最短、成本最低、使用车辆数最少、配送时间最快等）下，将客户的配送需求从配送中心送达客户点，或从客户点送回配送中心。

# def calDistance(CityCoordinates):
#     '''
#     计算城市间距离
#     输入：CityCoordinates-城市坐标；
#     输出：城市间距离矩阵-dis_matrix
#     '''
#     dis_matrix = pd.DataFrame(data=None, columns=range(len(CityCoordinates)), index=range(len(CityCoordinates)))
#     for i in range(len(CityCoordinates)):
#         xi, yi = CityCoordinates[i][0], CityCoordinates[i][1]
#         for j in range(len(CityCoordinates)):
#             xj, yj = CityCoordinates[j][0], CityCoordinates[j][1]
#             dis_matrix.iloc[i, j] = round(math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2), 2)
#     return dis_matrix

def car_type_data(car_type):
    # 车辆参数
    CAPACITY = 0  # 车辆最大容量
    C0 = 0  # 车辆启动成本
    C1 = 0  # 车辆单位距离行驶成本
    Z0 = 0  # 装货效率 单位 方/min
    Z1 = 0  #卸货效率
    if car_type == 4.2:
        CAPACITY = 16
        C0 = 200
        C1 = 1.686
        Z0 = 0.227
        Z1 = 0.544
    if car_type == 5.2:
        CAPACITY = 23
        C0 = 220.8
        C1 = 2.314
        Z0 = 0.326
        Z1 = 0.782
    if car_type == 6.8:
        CAPACITY = 38
        C0 = 268.9666667
        C1 = 3
        Z0 = 0.538
        Z1 = 1.292
    return CAPACITY, C0, C1, Z0, Z1


def greedy(CityCoordinates, dis_matrix):
    '''
    贪婪策略构造初始解,初始化时将VRP简化为TSP进行构造。
    输入：CityCoordinates-节点坐标,dis_matrix-距离矩阵
    输出：初始解-line
    '''
    # 修改dis_matrix以适应求解需要
    dis_matrix = dis_matrix.astype('float64')
    for i in range(len(CityCoordinates)):
        dis_matrix.loc[i, i] = math.pow(10, 10)
    dis_matrix.loc[:, 0] = math.pow(10, 10)  # 0不在编码内
    line = []  # 初始化
    now_city = random.randint(1, len(CityCoordinates) - 1)  # 随机生成出发城市
    line.append(now_city)  # 添加当前城市到路径
    dis_matrix.loc[:, now_city] = math.pow(10, 10)  # 更新距离矩阵，已经过城市不再被取出
    for i in range(1, len(CityCoordinates) - 1):
        next_city = dis_matrix.loc[now_city, :].idxmin()  # 距离最近的城市
        line.append(next_city)  # 添加进路径
        dis_matrix.loc[:, next_city] = math.pow(10, 10)  # 更新距离矩阵
        now_city = next_city  # 更新当前城市

    return line


def calFitness(birdPop, Demand, dis_matrix, DISTABCE, V):
    '''
    贪婪策略分配车辆（解码），计算路径距离（评价函数）
    输入：birdPop-路径，Demand-客户需求,dis_matrix-城市间距离矩阵，CAPACITY-车辆最大载重,DISTABCE-车辆最大行驶距离,C0-车辆启动成本,C1-车辆单位距离行驶成本；
    输出：birdPop_car-分车后路径,fits-适应度
    '''
    birdPop_car, fits , birdPop_car_type, birdPop_car_dis = [], [], [], []  # 初始化
    times, moneys = [], []
    for j in range(len(birdPop)):
        bird = birdPop[j]
        lines = []  # 存储线路分车
        line = [0]  # 每辆车服务客户点
        car_types = []  # 存储线路分车的车型
        dis_sum = []  # 线路上的路径距离
        time_car = []  # 线路上各路径所需时间
        dis, d = 0, 0  # 当前客户距离前一个客户的距离、当前客户需求量
        i = 0  # 指向配送中心
        time_point = 0
        car_type = 0  # 所需车型
        # 车辆参数
        CAPACITY = 0  # 车辆最大容量
        C0 = 0  # 车辆启动成本
        C1 = 0  # 车辆单位距离行驶成本
        Z0 = 0  # 装货效率 单位 方/min
        Z1 = 0  # 卸货效率
        money = []  # 分车后的费用
        while i < len(bird):

            if (Demand[bird[i]] <= 16) & (dis == 0):
                car_type = 4.2

            elif (Demand[bird[i]] <= 16) & (dis != 0) & (car_type == 4.2):
                car_type = 4.2

            elif (Demand[bird[i]] <= 16) & (dis != 0) & (car_type == 5.2):
                car_type = 5.2

            else:
                if Demand[bird[i]] > 16:
                    if (car_type == 4.2) & (dis != 0):
                        dis += dis_matrix.loc[line[-1], 0]  # 当前车辆装满
                        time_point += dis_matrix.loc[line[-1], 0] / V
                        time_point += d / Z0 / 60
                        line.append(0)
                        dis_sum.append(dis)
                        lines.append(line)
                        car_types.append(car_type)
                        time_car.append(round(time_point, 3))
                        money.append(round(C1 * dis + C0, 1))
                        # 下一辆车
                        dis, d = 0, 0
                        time_point = 0
                        line = [0]
                        car_type = 5.2

                    elif (car_type == 4.2) & (dis == 0):
                        dis, d = 0, 0
                        time_point = 0
                        line = [0]
                        car_type = 5.2

                    else:
                        car_type = 5.2

            if car_type == 4.2:
                CAPACITY, C0, C1, Z0, Z1 = car_type_data(car_type)
                if line == [0]:  # 车辆未分配客户点
                    dis += dis_matrix.loc[0, bird[i]]  # 记录距离
                    line.append(bird[i])  # 为客户点分车
                    d += Demand[bird[i]]  # 记录需求量
                    time_point += dis_matrix.loc[0, bird[i]] / V
                    time_point += Demand[bird[i]] / Z1 / 60
                    i += 1  # 指向下一个客户点
                else:  # 已分配客户点则需判断车辆载重和行驶距离
                    if (dis_matrix.loc[line[-1], bird[i]] + dis_matrix.loc[bird[i], 0] + dis <= DISTABCE) & (
                            d + Demand[bird[i]] <= CAPACITY):
                        dis += dis_matrix.loc[line[-1], bird[i]]
                        line.append(bird[i])
                        d += Demand[bird[i]]
                        time_point += dis_matrix.loc[line[-1], bird[i]] / V
                        time_point += Demand[bird[i]] / Z1 / 60
                        i += 1
                    else:
                        dis += dis_matrix.loc[line[-1], 0]  # 当前车辆装满
                        time_point += dis_matrix.loc[line[-1], 0] / V
                        time_point += d / Z0 / 60
                        line.append(0)
                        dis_sum.append(dis)
                        lines.append(line)
                        car_types.append(car_type)
                        time_car.append(round(time_point, 3))
                        money.append(round(C1 * dis + C0, 1))
                        # 下一辆车
                        dis, d = 0, 0
                        time_point = 0
                        line = [0]
            if car_type == 5.2:
                CAPACITY, C0, C1, Z0, Z1 = car_type_data(car_type)
                if line == [0]:  # 车辆未分配客户点
                    dis += dis_matrix.loc[0, bird[i]]  # 记录距离
                    line.append(bird[i])  # 为客户点分车
                    d += Demand[bird[i]]  # 记录需求量
                    time_point += dis_matrix.loc[0, bird[i]] / V
                    time_point += Demand[bird[i]] / Z1 / 60
                    i += 1  # 指向下一个客户点
                else:  # 已分配客户点则需判断车辆载重和行驶距离
                    if (dis_matrix.loc[line[-1], bird[i]] + dis_matrix.loc[bird[i], 0] + dis <= DISTABCE) & (
                            d + Demand[bird[i]] <= CAPACITY):
                        dis += dis_matrix.loc[line[-1], bird[i]]
                        line.append(bird[i])
                        d += Demand[bird[i]]
                        time_point += dis_matrix.loc[line[-1], bird[i]] / V
                        time_point += Demand[bird[i]] / Z1 / 60
                        i += 1
                    else:
                        dis += dis_matrix.loc[line[-1], 0]  # 当前车辆装满
                        time_point += dis_matrix.loc[line[-1], 0] / V
                        time_point += d / Z0 / 60
                        line.append(0)
                        dis_sum.append(dis)
                        lines.append(line)
                        car_types.append(car_type)
                        time_car.append(round(time_point, 3))
                        money.append(round(C1 * dis + C0, 1))
                        # 下一辆车
                        dis, d = 0, 0
                        time_point = 0
                        line = [0]

        # 最后一辆车
        dis += dis_matrix.loc[line[-1], 0]
        time_point += dis_matrix.loc[line[-1], 0] / V
        time_point += d / Z0 / 60
        line.append(0)
        dis_sum.append(round(dis, 3))
        lines.append(line)
        car_types.append(car_type)
        time_car.append(round(time_point, 3))
        money.append(round(C1 * dis + C0, 1))

        birdPop_car.append(lines)
        fits.append(round(sum(money), 1))
        birdPop_car_type.append(car_types)
        birdPop_car_dis.append(dis_sum)
        times.append(time_car)
        moneys.append(money)

    return birdPop_car, fits, birdPop_car_type, birdPop_car_dis, times, moneys


def crossover(bird, pLine, gLine, w, c1, c2):
    '''
    采用顺序交叉方式；交叉的parent1为粒子本身，分别以w/(w+c1+c2),c1/(w+c1+c2),c2/(w+c1+c2)
    的概率接受粒子本身逆序、当前最优解、全局最优解作为parent2,只选择其中一个作为parent2；
    输入：bird-粒子,pLine-当前最优解,gLine-全局最优解,w-惯性因子,c1-自我认知因子,c2-社会认知因子；
    输出：交叉后的粒子-croBird；
    '''
    croBird = [None] * len(bird)  # 初始化
    parent1 = bird  # 选择parent1
    # 选择parent2（轮盘赌操作）
    randNum = random.uniform(0, sum([w, c1, c2]))
    if randNum <= w:
        parent2 = [bird[i] for i in range(len(bird) - 1, -1, -1)]  # bird的逆序
    elif randNum <= w + c1:
        parent2 = pLine
    else:
        parent2 = gLine

    # parent1-> croBird
    start_pos = random.randint(0, len(parent1) - 1)
    end_pos = random.randint(0, len(parent1) - 1)
    if start_pos > end_pos: start_pos, end_pos = end_pos, start_pos
    croBird[start_pos:end_pos + 1] = parent1[start_pos:end_pos + 1].copy()

    # parent2 -> croBird
    list2 = list(range(0, start_pos))
    list1 = list(range(end_pos + 1, len(parent2)))
    list_index = list1 + list2  # croBird从后往前填充
    j = -1
    for i in list_index:
        for j in range(j + 1, len(parent2) + 1):
            if parent2[j] not in croBird:
                croBird[i] = parent2[j]
                break

    return croBird


def draw_path(car_routes, CityCoordinates):
    '''
    #画路径图
    输入：line-路径，CityCoordinates-城市坐标；
    输出：路径图
    '''
    # for route in car_routes:
    #     x, y = [], []
    #     for i in route:
    #         Coordinate = CityCoordinates[i]
    #         x.append(Coordinate[0])
    #         y.append(Coordinate[1])
    #     x.append(x[0])
    #     y.append(y[0])
    #     plt.plot(x, y, 'o-', alpha=0.8, linewidth=0.8)
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.show()

    m = folium.Map(location=[30.29924, 120.809512], zoom_start=10,
                   tiles='http://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=7&x={x}&y={y}&z={z}',
                   attr='default'
                   )  # 中心区域的确定

    color = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'gray', 'black', 'pink']
    l = []
    location = []
    p = car_routes

    for point in p:
        for i in point:
            a = CityCoordinates[i]
            l.append([a[1], a[0]])
        location.append(l)
        l = []
    # print(location)

    for i in range(len(location)):
        loca = location[i]
        point = p[i]
        # print(loca)
        # print(len(loca))
        # print(point)
        # print(len(point))
        for j in range(len(loca)):
            # print("第{}号点的坐标为{}".format(point[j], loca[j]))
            folium.CircleMarker(location=[loca[j][0], loca[j][1]],
                                radius=4, popup=point[j],
                                color='black', fill=True,
                                fill_color='orange').add_to(m)

    for i in range(len(location)):
        route = folium.PolyLine(  # polyline方法为将坐标用线段形式连接起来
            location[i],  # 将坐标点连接起来
            weight=3,  # 线的大小为3
            color=color[i],  # 线的颜色为橙色
            popup=i,
            opacity=0.8  # 线的透明度
        ).add_to(m)

    m.save(os.path.join(r'C:\Users\22041210\PycharmProjects\pso_rode', 'cvrp.html'))  # 将结果以HTML形式保存到桌面上


if __name__ == '__main__':
    # 车辆参数
    # CAPACITY = 23  # 车辆最大容量
    DISTABCE = 1000  # 车辆最大行驶距离
    V = 60  # 速度，km/h
    # C0 = 220.8  # 车辆启动成本
    # C1 = 2.314  # 车辆单位距离行驶成本

    # PSO参数
    birdNum = 50  # 粒子数量
    w = 0.2  # 惯性因子
    c1 = 0.4  # 自我认知因子
    c2 = 0.4  # 社会认知因子
    pBest, pLine = 0, []  # 当前最优值、当前最优解，（自我认知部分）
    gBest, gLine = 0, []  # 全局最优值、全局最优解，（社会认知部分）

    # 其他参数
    iterMax = 10000  # 迭代次数
    iterI = 1  # 当前迭代次数
    bestfit = []  # 记录每代最优值

    # 读入数据,
    # DistributionCenter = #配送中心
    # Customer = [(50, 50), (96, 24), (40, 5), (49, 8), (13, 7), (29, 89), (48, 30), (84, 39), (14, 47), (2, 24),
    #             (3, 82), (65, 10), (98, 52), (84, 25), (41, 69), (1, 65),
    #             (51, 71), (75, 83), (29, 32), (83, 3), (50, 93), (80, 94), (5, 42), (62, 70), (31, 62), (19, 97),
    #             (91, 75), (27, 49), (23, 15), (20, 70), (85, 60), (98, 85)]  #客户坐标点

    Customer = [(120.809512, 30.29924), (119.788449, 30.769), (119.782318, 30.777408), (119.712324, 30.665222),
                (119.623712, 30.829714), (119.384080, 30.571701), (119.680676, 30.659247), (119.670816, 30.665036),
                (119.680353, 30.638675), (119.686617, 30.631837), (119.782318, 30.777408), (120.258767, 30.843226),
                (119.676461, 30.656651), (119.782318, 30.777408), (119.782318, 30.777408), (119.389073, 30.575829),
                (119.566473, 30.598395), (119.569681, 30.582537), (119.698264, 30.617036), (119.645151, 30.774609),
                (119.782318, 30.777408), (119.566420, 30.598355), (119.678924, 30.657969), (119.697415, 30.63885),
                (119.659394, 30.632242), (119.621393, 30.519815), (119.697276, 30.604234), (119.686536, 30.63874),
                (119.690617, 30.637546)]  # 客户坐标点


    # Demand = [0, 16, 11, 6, 10, 7, 12, 16, 6, 16, 8, 14, 7, 16, 3, 22, 18, 19, 1, 14, 8, 12, 4, 8, 24, 24, 2, 10, 15, 2,
    #           14, 9]  #客户需求量

    # 客户需求量
    Demand = [0, 2.5589375, 3.738, 1.067967742, 3.427, 6.6375, 0.063, 22.21635484, 2.3545, 7.4145, 5.3415, 3.38,
              8.2278, 3.848125, 0.021, 2.4405, 10.7832, 3.26, 0.1295, 9.175, 3.10475, 3.0318, 10.5865, 3.833,
              2.538461538, 3.228, 5.507928571, 0.5388, 0.576]

    data_matrix = pd.read_csv('data_matrix.csv', delimiter=",")
    data_matrix = np.array(data_matrix)
    print(data_matrix)
    print(data_matrix.shape)
    dis_matrix = pd.DataFrame(data=None, columns=range(len(Customer)), index=range(len(Customer)))
    for i in range(len(Customer)):
        for j in range(len(Customer)):
            dis_matrix.iloc[i, j] = round(data_matrix[i][j]/1000, 3)

    # dis_matrix = calDistance(Customer)  # 计算城市间距离
    print(dis_matrix)

    birdPop = [greedy(Customer, dis_matrix) for i in range(birdNum)]  # 贪婪算法构造初始解
    # birdPop = [random.sample(range(1,len(Customer)),len(Customer)-1) for i in range(birdNum)]#客户点编码，随机初始化生成种群

    birdPop_car, fits, birdPop_car_type, birdPop_car_dis, times, moneys = calFitness(birdPop, Demand, dis_matrix, DISTABCE, V)  # 分配车辆，计算种群适应度

    gBest = pBest = min(fits)  # 全局最优值、当前最优值
    gLine = pLine = birdPop[fits.index(min(fits))]  # 全局最优解、当前最优解
    gLine_car = pLine_car = birdPop_car[fits.index(min(fits))]
    gLine_car_type = pLine_car_type = birdPop_car_type[fits.index(min(fits))]
    gLine_car_dis = pLine_car_dis = birdPop_car_dis[fits.index(min(fits))]
    gTimes = pTimes = times[fits.index(min(fits))]
    gMoneys = pMoneys = moneys[fits.index(min(fits))]
    bestfit.append(gBest)
    print("初始化时的全局最优解为：", bestfit[0])
    print("初始化的最优总时间为：", sum(gTimes))
    flag = 0
    while iterI <= iterMax:  # 迭代开始
        for i in range(birdNum):
            birdPop[i] = crossover(birdPop[i], pLine, gLine, w, c1, c2)

        birdPop_car, fits, birdPop_car_type, birdPop_car_dis, times, moneys = calFitness(birdPop, Demand, dis_matrix, DISTABCE, V)  # 分配车辆，计算种群适应度
        pBest, pLine, pLine_car, pLine_car_type, pLine_car_dis = min(fits), birdPop[fits.index(min(fits))],\
                                                  birdPop_car[fits.index(min(fits))],\
                                                  birdPop_car_type[fits.index(min(fits))], birdPop_car_dis[fits.index(min(fits))]
        pTimes, pMoneys = times[fits.index(min(fits))], moneys[fits.index(min(fits))]

        if min(fits) == gBest:
            flag = 1

        if min(fits) < gBest:
            gBest, gLine, gLine_car, gLine_car_type, gLine_car_dis = min(fits), birdPop[fits.index(min(fits))], birdPop_car[fits.index(min(fits))], \
                                                      birdPop_car_type[fits.index(min(fits))], birdPop_car_dis[fits.index(min(fits))]
            gTimes, gMoneys = times[fits.index(min(fits))], moneys[fits.index(min(fits))]

        if min(fits) <= gBest:
            gBest, gLine, gLine_car, gLine_car_type, gLine_car_dis = min(fits), birdPop[fits.index(min(fits))], birdPop_car[fits.index(min(fits))], \
                                                      birdPop_car_type[fits.index(min(fits))], birdPop_car_dis[fits.index(min(fits))]
            gMoneys = moneys[fits.index(min(fits))]

            if sum(pTimes) <= sum(gTimes):
                gTimes = times[fits.index(min(fits))]


        bestfit.append(gBest)
        print("第{}次迭代，当前最优解为{}，全局最优解为{}，当前代的全局最优解为{}，当前代的全局最优总时间为{}".
              format(iterI, pBest, gBest, bestfit[iterI], round(sum(gTimes), 3)))  # 打印当前代数和最佳适应度值
        iterI += 1  # 迭代计数加一

    print()
    print("分车后的路径数据：", gLine_car)  # 路径顺序
    print("路径分车的车型：", gLine_car_type)  #路径分车的车型
    print("分车后单个车的行驶距离: ", gLine_car_dis)  # 分车后单个车的行驶距离
    print("分车后单个车的行驶时间: ", gTimes)  # 分车后单个车的行驶时间
    print("分车后单个车的行驶费用: ", gMoneys)  # 分车后单个车的行驶费用
    for i in range(0, len(gLine_car)):
        print("第{}辆车，型号为{}，路径距离长度为: {}km，所需时间为: {}h，所需费用为: {}元，路径为: {}".format(i+1, gLine_car_type[i], round(gLine_car_dis[i], 3), gTimes[i], gMoneys[i], gLine_car[i]))
    print("全局最优的总费用为: {}, 全局最优的总时间为: {} ".format(gBest, round(sum(gTimes), 3)))
    print()
    draw_path(gLine_car, Customer)  # 画路径图
