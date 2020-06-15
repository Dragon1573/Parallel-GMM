/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright @ 2020 Dragon1573
*/
#pragma once
#include <float.h>
#include <math.h>
#include <memory.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

/* C/C++没有圆周率常量，这一点很不爽。用反三角的话估计是每次都要算一遍，效率太低了... */
// 圆周率
const double PI = 3.1415926535897932;

// 数据集
double *datasets;
// 数据量
size_t dataSize;
// 数据维度
size_t dimensions;
// 聚类簇数量
size_t clusters;

// KMeans聚类中心
double *centers;
// 标签集
int *labels;

// 高斯均值
double *means;
// 高斯方差
double *variances;
// 高斯权重
double *priorities;
// 高斯概率
double *probabilities;

/**
 * @brief 载入数据集
 *
 * @param fileName  文件名
*/
void loadFile(const char *const fileName) {
    FILE *file = fopen(fileName, "r");
    datasets = (double *)calloc(dimensions * dataSize, sizeof(double));
    for (int i = 0; i < dataSize; i++) {
        for (int j = 0; j < dimensions; j++) {
            /* 不过是为了关闭一个无意义的WARNING罢了 */
        #pragma warning(disable: 6031)
            fscanf(file, "%lf", datasets + i * dimensions + j);
        }
    }
    fclose(file);
}

/**
 * @brief 欧几里得距离
 *
 * @param sampleId  样本编号
 * @param clusterId 聚类编号
*/
const double getDistance(const size_t sampleId, const size_t clusterId) {
    // 距离（公有）
    double distance = 0;

    /* 由于嵌套调用的关系，此处无需进行初始化即可直接使用MPI函数 */
    int processors = 0, rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &processors);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    /* 并行计算（求和） */
    double dist_p = 0;
    const size_t interval = dimensions / processors + (dimensions % processors != 0);
    for (int i = rank * interval; i < min(dimensions, ((size_t)rank + 1) * interval); i++) {
        dist_p += pow(
            datasets[sampleId * dimensions + i]
            - centers[clusterId * dimensions + i],
            2
        );
    }
    MPI_Reduce(&dist_p, &distance, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    /* MPI环境是外层启动的，此处不需要清理 */

    /* 返回距离值 */
    return sqrt(distance);
}

/**
 * @brief 统计样本的成本
 *
 * @param sampleId  样本编号
*/
const double getCost(const size_t sampleId) {
    // 初始化样本标签
    labels[sampleId] = -1;
    // 最短距离
    double minimum = DBL_MAX;
    for (int i = 0; i < clusters; i++) {
        double temp = getDistance(sampleId, i);
        if (temp < minimum) {
            minimum = temp;
            labels[sampleId] = i;
        }
    }
    return minimum;
}

/**
 * @brief KMeans聚类算法
*/
void kMeans_clustering() {
#pragma region Initialization
    // 最大迭代次数
    const int maxInterations = 100;

    // 成本最小降幅
    const int epsilon = 0.001;

    // 聚类中心
    centers = (double *)calloc(clusters * dimensions, sizeof(double));
    for (int i = 0; i < clusters; i++) {
        // 数据集单元格编号
        const size_t start = i * dataSize * dimensions / clusters;
        for (int j = 0; j < dimensions; j++) {
            centers[i * dimensions + j] = datasets[start + j];
        }
    }

    // 标签集
    labels = (int *)calloc(dataSize, sizeof(int));
#pragma warning(disable: 6387)
    memset(labels, 0, dataSize * sizeof(int));

    // 模型成本（上次，本次）
    double costs[2] = {0, 0};

    // 各簇样本数量
    int *sampleCounts = (int *)calloc(clusters, sizeof(int));
    memset(sampleCounts, 0, clusters * sizeof(int));
#pragma endregion
#pragma region Estimations
    /* 迭代优化 */
    for (int i = 0; i < maxInterations; i++) {
        // 清空聚类计数器
        memset(sampleCounts, 0, clusters * sizeof(int));
        // 聚类中心矩阵
        double *nextMeans =
            (double *)calloc(clusters * dimensions, sizeof(double));
        if (nextMeans == NULL) {
            fprintf(stderr, "[ERROR] AllocateFailedException: nextMeans\n");
            exit(EXIT_FAILURE);
        }
        memset(nextMeans, 0, clusters * dimensions * sizeof(double));

        /* 刷新成本*/
        costs[0] = costs[1];
        costs[1] = 0;

        int processors = 0, rank = 0;
        MPI_Comm_size(MPI_COMM_WORLD, &processors);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        const size_t interval = dataSize / processors + (dataSize % processors != 0);
        double cost_p = 0,
            *sampleCounts_p = (double *)calloc(clusters, sizeof(double)),
            *nextMeans_p = (double *)calloc(clusters, sizeof(double));
        memset(sampleCounts_p, 0, clusters * sizeof(double));
        memset(nextMeans_p, 0, clusters * sizeof(double));
    #pragma warning(disable: 26451)
        for (int j = rank * interval; j < min(dataSize, interval * (rank + 1)); j++) {
            // 累计成本
            cost_p += getCost(j);
            // 类簇计数器自增
        #pragma warning(disable: 6011)
            sampleCounts_p[labels[j]] += 1;
            // 累计类簇中样本值（用于计算聚类中心）
            for (int k = 0; k < dimensions; k++) {
                nextMeans_p[labels[j] * dimensions + k] +=
                    datasets[j * dimensions + k];
            }
        }
        MPI_Reduce(&cost_p, costs + 1, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(
            sampleCounts_p, sampleCounts, clusters,
            MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD
        );
        MPI_Reduce(
            nextMeans_p, nextMeans, clusters, MPI_DOUBLE,
            MPI_SUM, 0, MPI_COMM_WORLD
        );
        free(nextMeans_p);
        free(sampleCounts_p);

        // 平均成本
        costs[1] /= dataSize;
        // 聚类中心
        for (int j = 0; j < clusters; j++) {
            if (sampleCounts[j] > 0) {
                for (int k = 0; k < dimensions; k++) {
                    centers[j * dimensions + k] =
                        nextMeans[j * dimensions + k] / sampleCounts[j];
                }
            }
        }

        // 释放局部数组
        free(nextMeans);

        // 调优条件
        if (fabs(costs[0] - costs[1]) < epsilon * costs[0]) {
            break;
        }
    }
#pragma endregion
#pragma region Cleanup
    free(sampleCounts);
#pragma endregion
}

/**
 * @brief GMM聚类算法（EM迭代）
*/
void gaussian_clustering() {
#pragma region Initialization
    // 最大迭代次数
    const int maxIterations = 100;
    // 学习率
    const double epsilon = 0.001;

    /* 初始化数组参数 */
    priorities = (double *)calloc(clusters, sizeof(double));
    memset(priorities, 0, clusters * sizeof(double));
    means = (double *)calloc(clusters * dimensions, sizeof(double));
    memcpy(means, centers, clusters * dimensions * sizeof(double));
    variances = (double *)calloc(clusters * dimensions, sizeof(double));
    memset(variances, 0, clusters * dimensions * sizeof(double));
    probabilities = (double *)calloc(dataSize, sizeof(double));
    memset(probabilities, 0, dataSize * sizeof(double));
    double *minVariances = (double *)calloc(dimensions, sizeof(double));
    memset(minVariances, 0, dimensions * sizeof(double));

    // 各KMeans类样本数量
    int *counts = (int *)calloc(clusters, sizeof(int));
    memset(counts, 0, clusters * sizeof(int));
    // 数据集整体均值
    double *overallMeans = (double *)calloc(dimensions, sizeof(double));
    memset(overallMeans, 0, dimensions * sizeof(double));

    /* 遍历数据集 */
    int i = 0;
    for (i = 0; i < dataSize; i++) {
        // 将当前样本分配至相应的聚类集合
        counts[labels[i]] += 1;

        for (int j = 0; j < dimensions; j++) {
            // 当前特征的偏离
            const double axes = datasets[i * dimensions + j]
                - centers[labels[i] * dimensions + j];
            // 累计当前聚类中心各特征的距离
            variances[labels[i] * dimensions + j] += pow(axes, 2);
            // 累计总体中心
            overallMeans[j] += datasets[i * dimensions + j];
            // 累计所有样本各特征的方差
            minVariances[j] += pow(datasets[i * dimensions + j], 2);
        }
    }

    for (int i = 0; i < dimensions; i++) {
        // 各特征的总体中心
        overallMeans[i] /= dataSize;
        // 最小方差（全局方差1%，最小为1e-10）
        minVariances[i] = max(
            1e-10,
            0.01 * (minVariances[i] / dataSize - pow(overallMeans[i], 2))
        );
    }

    /* 高斯分布初始化 */
    for (int i = 0; i < clusters; i++) {
        // 各分布权重
        priorities[i] = (double)counts[i] / dataSize;

        if (priorities[i] > 0) {
            /* 有效高斯分布 */
            for (int j = 0; j < dimensions; j++) {
                // 高斯方差
                variances[i * dimensions + j] /= counts[i];
                /* 方差一定有最小值 */
                variances[i * dimensions + j] = max(
                    variances[i * dimensions + j], minVariances[j]
                );
            }
        } else {
            /* 无用高斯分布 */
            // 用最小方差占位
            memcpy(
                variances + i * dimensions, minVariances,
                dimensions * sizeof(double)
            );
            // 产生警告
            fprintf(stderr, "[WARN] Gaussian Distribution %d is nonsense!\n", i);
        }
    }

    // 成本寄存器
    double costs[2] = {0, 0};
    // 权重寄存器
    double *nextPriorities = (double *)calloc(clusters, sizeof(double));
    // 方差寄存器
    double *nextVariances = (double *)calloc(clusters * dimensions, sizeof(double));
    // 均值寄存器
    double *nextMeans = (double *)calloc(clusters * dimensions, sizeof(double));
#pragma endregion
#pragma region Fitting
    for (int i = 0; i < maxIterations; i++) {
        /* 清空权重、方差、均值寄存器 */
        memset(nextPriorities, 0, clusters * sizeof(double));
        memset(nextVariances, 0, clusters * dimensions * sizeof(double));
        memset(nextMeans, 0, clusters * dimensions * sizeof(double));
        /* 更新成本寄存器 */
        costs[0] = costs[1];
        costs[1] = 0;

        /* 模型调优 */
        for (int j = 0; j < dataSize; j++) {
            // 样本概率
            probabilities[j] = 0;
            // 遍历分布
            for (int k = 0; k < clusters; k++) {
                // 样本属于此分布的概率
                double probability = 1;
                /* 计算单高斯分布概率密度 */
                for (int m = 0; m < dimensions; m++) {
                    probability *= 1 / sqrt(2 * PI * variances[k * dimensions + m]);
                    const double square = pow(
                        datasets[j * dimensions + m] - means[k * dimensions + m], 2
                    );
                    probability *= exp(-0.5 * square / variances[k * dimensions + m]);
                }
                probabilities[j] += priorities[k] * probability;

                /* 它与上面probability有什么区别，我也不知道... */
                // 样本属于当前高斯分布的概率
                const double sampleProbability =
                    probability * priorities[k] / probabilities[j];
                // 累计权重
                nextPriorities[k] += sampleProbability;

                // 遍历维度
                for (int m = 0; m < dimensions; m++) {
                    // 累计均值
                    nextMeans[k * dimensions + m] +=
                        sampleProbability * datasets[j * dimensions + m];
                    // 累计方差
                    nextVariances[k * dimensions + m] +=
                        sampleProbability * pow(datasets[j * dimensions + m], 2);
                }
            }

            /* 1e-20已经小于double参与计算的最小值了。别问，问就是魔法值 */
            // 累计样本引入的成本
            costs[1] += max(log10(probabilities[j]), -20);
        }

        // 计算当前平均成本
        costs[1] /= dataSize;

        // 优化
        for (int j = 0; j < clusters; j++) {
            // 刷新权重
            priorities[j] = nextPriorities[j] / dataSize;

            if (priorities[j] > 0) {
                /* 有意义的高斯分布 */
                for (int k = 0; k < dimensions; k++) {
                    /*
                    刷新均值

                    PS：别问，这叫玄学
                    */
                    means[j * dimensions + k] =
                        nextMeans[j * dimensions + k] / nextPriorities[j];
                    if (fabs(means[j * dimensions + k]) < DBL_EPSILON) {
                        means[j * dimensions + k] = 0;
                    }

                    /* 刷新方差 */
                    variances[j * dimensions + k] = max(
                        nextVariances[j * dimensions + k] / nextPriorities[j]
                        - pow(means[j * dimensions + k], 2),
                        minVariances[k]
                    );
                }
            }
        }

        /* 停止条件 */
        if (fabs(costs[1] - costs[0]) < epsilon * fabs(costs[1])) {
            break;
        }
    }
#pragma endregion
#pragma region Cleanup
    /* 释放空间 */
    free(counts);
    free(overallMeans);
    free(minVariances);
    free(nextPriorities);
    free(nextVariances);
    free(nextMeans);
#pragma endregion
}

/**
 * @brief 保存聚类结果
 *
 * @param fileName  文件名
*/
void saveFile(const char *const fileName) {
    FILE *file = fopen(fileName, "wb");

    /* 好了，XML硬核输出开始 */
    fprintf(file,
        "<?xml version=\"1.0\" encoding=\"GBK\"?>\n"
        "<model>\n"
        "<description>高斯混合模型聚类</description>\n"
        "<gaussians>\n"
        "<count>%lld</count>\n"
        , clusters);
    for (int i = 0; i < clusters; i++) {
        fprintf(file,
            "<gaussian>\n"
            "<mean>(%lg",
            centers[i * dimensions]);
        for (int j = 1; j < dimensions; j++) {
            fprintf(file, ", %lg", centers[i * dimensions + j]);
        }
        fprintf(file,
            ")</mean>\n"
            "<variance>(%lg",
            variances[i * dimensions]
        );
        for (int j = 1; j < dimensions; j++) {
            fprintf(file, ", %lg", variances[i * dimensions + j]);
        }
        fprintf(file,
            ")</variance>\n"
            "<priority>%lg</priority>"
            "</gaussian>",
            priorities[i]
        );
    }
    fprintf(file,
        "</gaussians>\n"
        "<dataset>\n"
        "<shape>(%lld, %lld)</shape>\n"
        "<samples>\n"
        , dataSize, dimensions);
    for (int i = 0; i < dataSize; i++) {
        fprintf(file,
            "<sample>\n"
            "<data>(%lg"
            , datasets[i * dimensions]);
        for (int j = 0; j < dimensions; j++) {
            fprintf(file, ", %lg", datasets[i * dimensions + j]);
        }
        fprintf(file,
            ")</data>\n"
            "<label>%d</label>\n"
            "<probability>%lf</probability>"
            "</sample>\n"
            , labels[i], probabilities[i]);
    }
    fprintf(file,
        "</samples>\n"
        "</dataset>\n"
        "</model>\n"
    );
    /* 上面的Servlet式暴力输出爽吧 */

    fclose(file);
    printf("[Success] Cluster details saved!\n");
}

/**
 * @brief 清理全局范围中开辟的空间
*/
void cleanUp() {
    free(datasets);
    free(centers);
    free(labels);
    free(means);
    free(variances);
    free(priorities);
    free(probabilities);
}
