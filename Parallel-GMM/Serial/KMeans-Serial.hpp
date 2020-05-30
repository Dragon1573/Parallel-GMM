/**
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright @ 2020 Dragon1573
 */
#pragma once
#include <iostream>
using namespace std;

// 超参数初始化模式
enum InitMode {
    Randomly, // 随机初始化
    Manually, // 指定值初始化
    Uniformly // 均匀初始化
};

/**
 @breif K均值算法

 @date 2003/10/16 Fei Wang
 @date 2013       Jave Lu     （阿凡卢）
 @date 2020/05/25 Jeff Huang  （べ断桥烟雨ミ）
 */
class KMeans {
public:
    /**
     * @brief 构造函数
     *
     * @param dimensions    数据维度
     * @param clusters      聚类簇总数
     * @return
    */
    KMeans(const int dimensions = 1, const int clusters = 1);
    // 析构函数
    ~KMeans();

    /**
     * @brief 设置聚类中心坐标
     *
     * @param i     类别编号
     * @param point 坐标点
    */
    void setMean(const int i, const double *point);
    // 设置初始化模式
    void setInitMode(enum InitMode mode);
    // 设置最大迭代次数
    void setMaxIterations(const int iterations);
    // 设置最大误差
    void setEpsilon(const double epsilon);
    // 获取聚类中心
    const double *const getMean(const int i);
    // 获取初始化状态
    const int getInitMode();
    // 获取最大迭代次数
    const int getMaxIterations();
    // 获取最大误差
    const double getEpsilon();

    /**
     * @brief 对模型进行聚类
     *
     * @param datasets      数据集（一维数组，以行优先模拟二维矩阵）
     * @param dataSize    数据量
    */
    const int *fit_transform(const double *const datasets, const int dataSize);
    /**
     * @brief 从文件加载数据集
     *
     * @param  file          数据集文件流
     * @param  dimensions    特征维度
     * @param  size          数据量
     * @return 数据集矩阵
    */
    const double *const loadFile(ifstream &file, const int size);

    /**
     * @brief 无监督聚类预测
     *
     * @param dataset
     * @param dataSize
    */
    const int *predict(const double *const dataset, const int dataSize);

    /**
     * @brief 输出模型信息
     *
     * @param out       输出流
     * @param kmeans    K阶中心距聚类模型
    */
    friend ostream &operator<<(ostream &, const KMeans *const);

private:
    // 维度数量
    int dimensions;
    // 聚类簇总数
    int clusters;
    // 聚类中心矩阵
    double **means;
    // 超参数初始化模式
    InitMode initMode;
    // 最大迭代次数（超过即结束）
    int maxInterations;
    // 误差精度限（小于即结束）
    double epsilon;

    /**
     * @brief 获取标签
     *
     * @param sample    样本坐标点
    */
    const double getCost(const double *sample, int &label);
    /**
     @brief 计算数据点之间的欧几里得距离

     @param x           坐标点A
     @param y           坐标点B
     @param dimensions  空间维度
     */
    const double getDistance(const double *, const double *, const int);
};
