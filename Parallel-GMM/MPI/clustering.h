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

// 进程编号
int rank;
// 进程总数
int processors;

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
    if (file == NULL) {
        fprintf(stderr, "[ERROR] Cannot Load the Dataset!\n\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        exit(EXIT_FAILURE);
    }
    datasets = (double *)calloc(dimensions * dataSize, sizeof(double));
    for (int i = 0; i < dataSize; i++) {
        for (int j = 0; j < dimensions; j++) {
            /* 不过是为了关闭一个无意义的WARNING罢了 */
        #pragma warning(disable: 6031)
            fscanf(file, "%lf", datasets + i * dimensions + j);
        }
    }
    fclose(file);

#ifdef _DEBUG
    fprintf(stderr, "[Success] Dataset Loaded!\n\n");
#endif
}

/**
 * @brief 计算欧几里得距离
 *
 * @param sample    样本
 * @param clusterId 类簇编号
*/
double getDistance(const double *const sample, const size_t clusterId) {
    double sum = 0;
    for (size_t i = 0; i < dimensions; i++) {
        sum += pow(sample[i] - centers[clusterId * dimensions + i], 2);
    }
    return sqrt(sum);
}

/**
 * @brief KMeans聚类算法
*/
void kMeans_clustering() {
    /* 初始化内存空间，由于进程[0]也将参与计算（支持单进程MPI），所以也要为其开辟本地空间 */
#pragma region Init_Memory
    // 数据集分块方案
    const size_t interval = dataSize / processors;

    // 标签集（进程[0]独占）
    labels = (int *)calloc(dataSize, sizeof(int));
#pragma warning (disable:6387)
    memset(labels, 0, dataSize * sizeof(int));
    // 累计距离（进程[0]独占）
    double *g_sumDistance = (double *)calloc(clusters, sizeof(double));
    // 训练成本
    double costs[2] = {0, 0};

    // 数据集（进程间隔离）
    double *localDatasets = (double *)calloc(interval * dimensions, sizeof(double));
    memset(localDatasets, 0, interval * dimensions * sizeof(double));
    // 标签集（进程间隔离）
    int *localLabels = (double *)calloc(interval, sizeof(int));
    memset(localLabels, 0, interval * sizeof(int));
    // 累计距离（进程间隔离）
    double *sumDistance = (double *)calloc(clusters, sizeof(double));
#pragma endregion

    /* TIPS: 通信死锁解决方案

     MPI_Send() 函数是一个同步通信函数，调用此函数后，进程将被挂起，直到有配对进程调用
     MPI_Recv() 进行接收。由于进程[0]也需要参与计算，存在进程[0]自我收发行为，同步通信将导
     致进程[0]向自己发送消息后被挂起而无法调用接收函数。所以我们选择从逻辑上避免进程[0]自我
     收发，将『自我收发』变成一个简单的内存复制，借助 memcpy() 实现『通信』。
    */
    /* 数据集划分 */
#pragma region Data_Delivery
    if (rank == 0) {
        /* 进程[0]『自我通信』 */
        memcpy(localDatasets, datasets, interval * dimensions * sizeof(double));

        /* 进程[0]向其他进程分发数据集 */
        for (size_t i = interval; i < dataSize; i += interval) {
            MPI_Send(datasets + i * dimensions, interval * dimensions,
                MPI_DOUBLE, i / interval, 0, MPI_COMM_WORLD);
        }
    } else {
        /* 其他进程接收来自进程[0]的数据块 */
        MPI_Recv(localDatasets, interval * dimensions, MPI_DOUBLE,
            0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
#pragma endregion

    /* 聚类中心初始化 */
#pragma region Init_Cluster
    centers = (double *)calloc(clusters * dimensions, sizeof(double));
    memset(centers, 0, clusters * dimensions * sizeof(double));

    /* 进程[0]等距初始化聚类中心 */
    if (rank == 0) {
        for (size_t i = 0; i < clusters; i++) {
            memcpy(centers + i * dimensions,
                datasets + i * dataSize / clusters * dimensions,
                dimensions * sizeof(double));
        }
    }
#pragma endregion

    /* 迭代调优 */
#pragma region Iterations
    /* 最多迭代100次 */
    for (size_t i = 0; i < 100; i++) {
        /* 进程[0]广播聚类中心 */
        MPI_Bcast(centers, clusters * dimensions, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        /* 并行聚类 */
        for (size_t j = 0; j < interval; j++) {
            /* 以最大双精度浮点数进行初始化 */
            // 最近距离
            double minDistance = DBL_MAX;
            for (size_t k = 0; k < clusters; k++) {
                const double temporary =
                    getDistance(localDatasets + j * dimensions, k);
                if (temporary < minDistance) {
                    /* 持续刷新最近距离 */
                    minDistance = temporary;
                    /* 更新样本所属的标签 */
                    localLabels[j] = k;
                }
            }
        }

        /* 计算本次训练的成本 */
        memset(sumDistance, 0, clusters * sizeof(double));
        for (size_t j = 0; j < clusters; j++) {
            for (size_t k = 0; k < interval; k++) {
                if (j == localLabels[k]) {
                    sumDistance[j] = getDistance(localDatasets + k * dimensions, j);
                }
            }
        }

        /* 进程[0]汇总标签集 */
        MPI_Gather(localLabels, interval, MPI_INT, labels,
            interval, MPI_INT, 0, MPI_COMM_WORLD);

        /* 进程[0]计算训练成本 */
        memset(g_sumDistance, 0, clusters * sizeof(double));
        MPI_Reduce(sumDistance, g_sumDistance, clusters,
            MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            costs[1] = 0;
            for (size_t j = 0; j < clusters; j++) {
                costs[1] += g_sumDistance[j];
            }
        #ifdef _DEBUG
            fprintf(stderr, "[INFO] Costs of Iteration %llu: %lf\n\n", i + 1, costs[1]);
        #endif
        }

        /* 进程[0]广播训练成本 */
        MPI_Bcast(costs + 1, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        /* 当成本不再显著下降，退出迭代 */
        if (fabs(costs[1] - costs[0]) < DBL_EPSILON) {
        #ifdef _DEBUG
            fprintf(stderr, "[INFO] Iterations Elapsed: %llu\n\n", i + 1);
        #endif
            break;
        }

        /* 刷新成本 */
        costs[0] = costs[1];

        /* 进程[0]更新聚类中心 */
        if (rank == 0) {
            // 聚类中心寄存器
            double *nextCenters = (double *)calloc(clusters * dimensions, sizeof(double));
            memset(nextCenters, 0, clusters * dimensions * sizeof(double));

            for (size_t j = 0; j < clusters; j++) {
                // 聚类中心拥有的样本数量
                size_t counter = 0;

                for (size_t k = 0; k < dataSize; k++) {
                    if (j == labels[k]) {
                        for (size_t m = 0; m < dimensions; m++) {
                            /* 累计样本坐标 */
                            nextCenters[j * dimensions + m] +=
                                datasets[k * dimensions + m];
                        }

                        /* 累计样本数量 */
                        ++counter;
                    }
                }

                /* 计算聚类中心 */
                for (size_t k = 0; k < dimensions; k++) {
                    centers[j * dimensions + k] =
                        nextCenters[j * dimensions + k] / counter;
                }
            }

            free(nextCenters);
        }
    }
#pragma endregion

    /* 释放内存空间，防止内存泄漏 */
#pragma region Cleanup
    free(sumDistance);
    free(localLabels);
    free(localDatasets);
    free(g_sumDistance);
#pragma endregion
}

/**
 * @brief 保存聚类结果
 *
 * @param fileName  文件名
*/
void saveKMeans(const char *const fileName) {
    FILE *file = fopen(fileName, "wb");

    /* 好了，XML硬核输出开始 */
    fprintf(file,
        "<?xml version=\"1.0\" encoding=\"GBK\"?>\n"
        "<model>\n"
        "<description>K阶中心距聚类</description>\n"
        "<clusters>\n"
        "<count>%lld</count>\n"
        , clusters);
    for (int i = 0; i < clusters; i++) {
        fprintf(file,
            "<cluster>\n"
            "<mean>(%lg",
            centers[i * dimensions]);
        for (int j = 1; j < dimensions; j++) {
            fprintf(file, ", %lg", centers[i * dimensions + j]);
        }
        fprintf(file,
            ")</mean>\n"
            "</cluster>\n"
        );
    }
    fprintf(file,
        "</clusters>\n"
        "<dataset>\n"
        "<shape>(%lld, %lld)</shape>\n"
        "<samples>\n"
        , dataSize, dimensions);
    for (int i = 0; i < dataSize; i++) {
        fprintf(file,
            "<sample>\n"
            "<data>(%lg"
            , datasets[i * dimensions]);
        for (int j = 1; j < dimensions; j++) {
            fprintf(file, ", %lg", datasets[i * dimensions + j]);
        }
        fprintf(file,
            ")</data>\n"
            "<label>%d</label>\n"
            "</sample>\n"
            , labels[i]);
    }
    fprintf(file,
        "</samples>\n"
        "</dataset>\n"
        "</model>\n"
    );
    /* 上面的Servlet式暴力输出爽吧 */

    fclose(file);
    printf("[Success] KMeans details saved!\n");
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
void saveGaussian(const char *const fileName) {
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
    printf("[Success] Gaussian details saved!\n");
}

/**
 * @brief 清理全局范围中开辟的空间
*/
void cleanUp() {
    free(centers);
    free(labels);
    free(means);
    if (rank == 0) {
        free(datasets);
        free(variances);
        free(priorities);
        free(probabilities);
    }
}
