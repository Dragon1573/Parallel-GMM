/**
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright @ 2020 Dragon1573
 */
#include "GMM-Serial.hpp"
#include <cmath>

 // 圆周率（不要问为什么这么多位，小时候吃饱了没事干）
constexpr double PI = 3.1415926535897932;

const double GMM::getProbability(const double *const sample, const int clusterId) {
    // 初始化概率值
    double p = 1;

    /*
    计算单高斯分布概率密度函数，公式来自原作者（Jerry Lu）的博客：
    http://tinyurl.com/ycvhvp57 （新浪短链接，不放心请自行还原）
    */
    for (int i = 0; i < this->dimensions; i++) {
        p *= 1 / sqrt(2 * PI * this->variances[clusterId][i]);
        double square = pow(sample[i] - this->means[clusterId][i], 2);
        p *= exp(-0.5 * square / this->variances[clusterId][i]);
    }

    // 返回概率值
    return p;
}

GMM::GMM(const int dimensions, const int mixtures) {
    // 导入配置项
    this->dimensions = dimensions;
    this->mixtures = mixtures;
    this->maxIterations = 100;
    this->epsilon = 0.001;

    // 初始化KMeans模型（后续需要用KMeans模型提供的方法解析数据集文件）
    this->kmeans = new KMeans();
    this->kmeans = new KMeans(this->dimensions, this->mixtures);
    this->kmeans->setInitMode(InitMode::Uniformly);

    // 初始化数组参数
    this->priorities = new double[this->mixtures];
    this->means = new double *[this->mixtures];
    this->variances = new double *[this->mixtures];
    for (int i = 0; i < this->mixtures; i++) {
        this->means[i] = new double[this->dimensions];
        this->variances[i] = new double[this->dimensions];
    }
    this->minVariance = new double[this->dimensions];
}

void GMM::fit(const double *const dataSet, const int dataSize) {
    // 初始化模型
    this->initialize(dataSet, dataSize);

    // 初始化迭代计数器
    int interations = 0;

    // 初始化成本寄存器 = {上次迭代的成本，本次迭代的成本}
    double costs[2] = {0, 0};

    // 初始化权重寄存器
    double *nextPriorities = new double[this->mixtures];
    // 初始化方差寄存器
    double **nextVariances = new double *[this->mixtures];
    // 初始化均值寄存器
    double **nextMeans = new double *[this->mixtures];
    for (int i = 0; i < this->mixtures; i++) {
        nextVariances[i] = new double[this->dimensions];
        nextMeans[i] = new double[this->dimensions];
    }

    // 开始训练
    for (int i = 0; i < this->maxIterations; i++) {
        // 清空寄存器
        memset(nextPriorities, 0, this->mixtures * sizeof(double));
        for (int j = 0; j < this->mixtures; j++) {
            memset(nextVariances[j], 0, this->dimensions * sizeof(double));
            memset(nextMeans[j], 0, this->dimensions * sizeof(double));
        }

        // 更新成本寄存器
        costs[0] = costs[1];
        costs[1] = 0;

        // 预测
        for (int j = 0; j < dataSize; j++) {
            // 获取样本
            double *const sample = new double[this->dimensions];
            for (int k = 0; k < this->dimensions; k++) {
                sample[k] = dataSet[j * this->dimensions + k];
            }

            // 计算样本概率
            const double probability = this->getProbability(sample);

            // 遍历所有高斯分布
            for (int k = 0; k < this->mixtures; k++) {
                // 计算样本属于当前高斯分布的概率
                const double SampleProbability = this->getProbability(sample, k)
                    * this->priorities[k] / probability;
                // 累计当前高斯分布的权重
                nextPriorities[k] += SampleProbability;

                // 遍历每个特征维度
                for (int d = 0; d < this->dimensions; d++) {
                    // 累计当前高斯分布的均值
                    nextMeans[k][d] += SampleProbability * sample[d];
                    // 累计当前高斯分布的方差
                    nextVariances[k][d] += SampleProbability * pow(sample[d], 2);
                }
            }

            // 累计样本引入的成本
            /*
                TIPS：未知的魔法值

                其实这里的 1e-20 已经远小于双精度浮点精度限了，
                至于 Jerry Lu 为什么要设置这样一个魔法值，我也不知道...
            */
            costs[1] += (probability > 1e-20 ? log10(probability) : -20);

            // 释放样本
            delete[] sample;
        }

        // 计算当前平均成本
        costs[1] /= dataSize;

        // 优化
        for (int j = 0; j < this->mixtures; j++) {
            // 刷新当前高斯分布的权重
            this->priorities[j] = nextPriorities[j] / dataSize;
            if (this->priorities[j] > 0) {
                /* 当前高斯分布有意义 */

                for (int k = 0; k < this->dimensions; k++) {
                    // 刷新当前高斯分布的均值
                    this->means[j][k] = nextMeans[j][k] / nextPriorities[j];
                    /*
                    TIPS：未知的极小值

                        经过我个人的测试，位于 GMM-test.cpp 中的数据集产生均值会出现一个
                    1e-65 数量级的值，而正确的结果应该是0。
                        个人猜测是因为 nextMeans[j][k] 的计算过程中，只要其一不为 0 ，
                    那么最终的结果也不会是 0 ，即使这个值已经远小于 DBL_EPSILON 双精度浮
                    点的最小精度限。
                        因此，我在这里将小于精度限的值统一视作0。
                    */
                    if (fabs(this->means[j][k]) < DBL_EPSILON) {
                        this->means[j][k] = 0;
                    }

                    // 刷新当前高斯分布的方差
                    this->variances[j][k] =
                        nextVariances[j][k] / nextPriorities[j];
                    this->variances[j][k] -= pow(this->means[j][k], 2);

                    // 高斯分布总有最小方差
                    this->variances[j][k] = max(
                        this->variances[j][k],
                        this->minVariance[k]
                    );
                }
            }
        }

        // 如果成本变化小于阈值（认为成本不再变化），则停止迭代
        if (fabs(costs[1] - costs[0]) < this->epsilon * fabs(costs[1])) {
            break;
        }
    }

    // 释放寄存器
    for (int i = 0; i < this->mixtures; i++) {
        delete[] nextMeans[i];
        delete[] nextVariances[i];
    }
    delete[] nextMeans;
    delete[] nextVariances;
    delete[] nextPriorities;
}

void GMM::initialize(const double *const dataSet, const int dataSize) {
    // 设置最小方差
    const double MIN_VARIANCE = 1e-10;

    // 获取KMeans聚类的标签集
    const int *labels = this->kmeans->fit_transform(dataSet, dataSize);

    // 各KMeans聚类的样本数量
    int *counts = new int[this->mixtures];
    // 所有样本的整体中心
    double *overallMeans = new double[this->dimensions];
    memset(overallMeans, 0, this->dimensions * sizeof(double));

    for (int i = 0; i < this->mixtures; i++) {
        counts[i] = 0;
        this->priorities[i] = 0;
        memcpy(
            this->means[i], this->kmeans->getMean(i),
            this->dimensions * sizeof(double)
        );
        memset(this->variances[i], 0, this->dimensions * sizeof(double));
    }
    memset(this->minVariance, 0, this->dimensions * sizeof(double));

    // 遍历数据集
    for (int i = 0; i < dataSize; i++) {
        // 当前样本数据
        double *const sample = new double[this->dimensions];
        for (int j = 0; j < this->dimensions; j++) {
            sample[j] = dataSet[i * this->dimensions + j];
        }

        // 将当前样本分配至相应的聚类集合
        counts[labels[i]] += 1;

        // 获取当前类别的聚类中心
        const double *const center = this->kmeans->getMean(labels[i]);

        for (int j = 0; j < this->dimensions; j++) {
            // 累计当前聚类中心各特征的距离
            this->variances[labels[i]][j] += pow(sample[j] - center[j], 2);
            // 累计总体中心
            overallMeans[j] += sample[j];
            // 累计所有样本各特征的方差
            this->minVariance[j] += pow(sample[j], 2);
        }

        // 释放动态开辟的数组空间
        delete[] sample;
    }

    for (int i = 0; i < this->dimensions; i++) {
        // 计算总体中心
        overallMeans[i] /= dataSize;

        // 以全局方差的 1% 作为最小方差（不小于 1e-10）
        this->minVariance[i] = max(
            MIN_VARIANCE,
            0.01 * (this->minVariance[i] / dataSize - pow(overallMeans[i], 2))
        );
    }

    // 初始化各高斯分布
    for (int i = 0; i < this->mixtures; i++) {
        // 计算各分布的权重
        this->priorities[i] = 1.0 * counts[i] / dataSize;

        if (this->priorities[i] > 0) {
            /* 这是一个有效的高斯分布 */

            for (int j = 0; j < this->dimensions; j++) {
                // 计算高斯分布的方差
                this->variances[i][j] /= counts[i];

                // 高斯分布存在一个最小方差，任何数据集的方差不可能小于此数值
                this->variances[i][j] = max(
                    this->variances[i][j], this->minVariance[j]
                );
            }
        } else {
            /* 此高斯分布权重为0，对高斯混合模型无影响 */

            // 置相应的高斯分布为最小方差
            memcpy(
                this->variances[i],
                this->minVariance,
                this->dimensions * sizeof(double)
            );
            cerr << "[警告] 高斯混合模型中的第" << i << "个高斯分布无实际意义！" << endl;
        }
    }

    // 释放动态开辟的数组空间
    delete[] overallMeans;
    delete[] counts;
    delete[] labels;
}

const double GMM::getProbability(const double *const sample) {
    // 初始化概率
    double probability = 0;

    // 遍历所有高斯分布
    for (int j = 0; j < this->mixtures; j++) {
        // 累计当前样本的多维高斯联合分布
        probability +=
            this->priorities[j] * this->getProbability(sample, j);
    }

    // 返回概率
    return probability;
}

ostream &operator<<(ostream &out, const GMM *const model) {
    out << "模型名称：高斯混合模型" << endl
        << "数据维度：" << model->dimensions << endl
        << "高斯分布数量：" << model->mixtures << endl
        << "高斯分布权重：(" << model->priorities[0];
    for (int i = 1; i < model->mixtures; i++) {
        out << ", " << model->priorities[i];
    }
    out << ")" << endl
        << "高斯分布均值：" << endl;
    for (int i = 0; i < model->mixtures; i++) {
        out << "\t" << i << "：(" << model->means[i][0];
        for (int j = 1; j < model->dimensions; j++) {
            out << ", " << model->means[i][j];
        }
        out << ")" << endl;
    }
    out << "高斯分布方差：" << endl;
    for (int i = 0; i < model->mixtures; i++) {
        out << "\t" << i << "：(" << model->variances[i][0];
        for (int j = 1; j < model->dimensions; j++) {
            out << ", " << model->variances[i][j];
        }
        out << ")" << endl;
    }

    return out;
}
