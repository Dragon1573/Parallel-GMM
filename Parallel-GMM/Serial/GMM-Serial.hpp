#pragma once
#include <fstream>
#include "KMeans-Serial.hpp"

/**
 * @brief 基于对角协方差矩阵的高斯混合模型
 *
 * @date 2003/11/01 Fei Wang
 * @date 2013       Jerry Lu
 * @date 2020/05/29 Jeff Huang
 */
class GMM {
private:
    // 样本维度
    int dimensions;
    // 高斯分布数量
    int mixtures;
    // 高斯分布的权重
    double *priorities = nullptr;
    // 高斯分布的均值（聚类中心矩阵）
    double **means = nullptr;
    // 高斯分布的方差
    double **variances = nullptr;
    // 最小方差
    double *minVariance = nullptr;
    // 最大迭代次数（用作迭代停止阀）
    int maxIterations;
    // 成本误差限（用作迭代停止阀）
    double epsilon;
    // KMeans聚类模型
    KMeans *kmeans = nullptr;

    /**
     * @brief 获取样本属于指定高斯分布的概率
     *
     * @param   sample    样本
     * @param   clusterId 类别编号
     * @return  概率值
     */
    const double getProbability(const double *const sample, const int clusterId);

    /**
     * @brief 初始化模型
     *
     * @param dataSet   数据集
     * @param dataSize  数据量
     */
    void initialize(const double *const dataSet, const int dataSize);

public:
    /**
     * @brief 构造函数
     *
     * @param dimensions    数据集维度
     * @param mixtures       单高斯分布混合数量
     */
    GMM(const int dimensions = 1, const int mixtures = 1);

    /**
     * @brief 训练数据集
     *
     * @param dataSet   数据集
     * @param dataSize  数据量
     */
    void fit(const double *const dataSet, const int dataSize);

    /**
     * @brief 对数据集进行聚类
     *
     * @param sample   数据集
     * @return 概率数组
     */
    const double getProbability(const double *const sample);

    // 输出模型
    friend ostream &operator<<(ostream &out, const GMM *const model);
};
