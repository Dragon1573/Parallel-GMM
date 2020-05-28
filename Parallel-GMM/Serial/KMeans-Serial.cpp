#pragma once
#include "KMeans-Serial.hpp"
#include <cassert>
#include <cmath>
#include <ctime>
#include <fstream>
#include <string>
#include <sstream>
using namespace std;

constexpr int DOUBLE = sizeof(double);
constexpr int INT = sizeof(int);

KMeans::KMeans(const int dimensions, const int clusters) {
    // 指定数据集维度
    this->dimensions = dimensions;
    // 指定聚类数量
    this->clusters = clusters;

    // 初始化坐标数组，数组每一项都是一个存储了各维度值的数组
    this->means = new double *[this->clusters];
    for (int i = 0; i < this->clusters; i++) {
        // 初始化聚类中心，所有类别的聚类中心都位于超空间原点
        this->means[i] = new double[this->dimensions];
        memset(this->means[i], 0, static_cast<size_t>(this->dimensions) * DOUBLE);
    }

    // 随机初始化矩阵
    this->initMode = InitMode::Randomly;
    // 聚类中心最多更新100次
    this->maxInterations = 100;
    // 设置最大误差为0.1%
    this->epsilon = 0.001;
}

KMeans::~KMeans() {
    // 由 new 关键字开辟的堆区不会随默认析构而释放
    for (int i = 0; i < clusters; i++) {
        // 先释放内层的数组
        delete[] this->means[i];
    }
    // 再释放外层的数组
    delete[] this->means;
}

void KMeans::setMean(const int i, const double *point) {
    memcpy(this->means[i], point, DOUBLE);
}

void KMeans::setInitMode(enum InitMode mode) {
    this->initMode = mode;
}

void KMeans::setMaxIterations(const int iterations) {
    this->maxInterations = iterations;
}

void KMeans::setEpsilon(const double epsilon) {
    this->epsilon = epsilon;
}

const double *const KMeans::getMean(const int i) {
    return this->means[i];
}

const int KMeans::getInitMode() {
    return this->initMode;
}

const int KMeans::getMaxIterations() {
    return this->maxInterations;
}

const double KMeans::getEpsilon() {
    return this->epsilon;
}

const int *KMeans::fit_transform(
    const double *const datasets, const int dataSize) {
    switch (this->initMode) {
    case Randomly: // 随机选取各类别的1个样本作为聚类中心
    {
        // 获取训练集每个类别的样本数量
        int interval = dataSize / this->clusters;
        // 以当前时间初始化伪随机数生成器
        srand((unsigned int)time(NULL));

        for (int i = 0; i < clusters; i++) {
            // 随机选取一个样本点
            int selection = (int)((interval - 1) * (double)rand() / RAND_MAX);
            selection += interval * i;
            // 提取数据并
            for (int j = 0; j < this->dimensions; j++) {
                this->means[i][j] = datasets[selection * this->dimensions + j];
            }
        }

        break;
    }
    case Uniformly: // 将各类别首个样本作为聚类中心
    {
        for (int i = 0; i < this->clusters; i++) {
            int selection = i * dataSize / this->clusters;
            for (int j = 0; j < this->dimensions; j++) {
                this->means[i][j] = datasets[selection * this->dimensions + j];
            }
        }
        break;
    }
    case Manually: // 用户自行初始化
        // 什么都不干
        break;
    }

    return predict(datasets, dataSize);
}

const double *const KMeans::loadFile(
    ifstream &file, const int dataSize
) {
    // 数据量至少要和聚类簇数相当
    assert(dataSize >= this->clusters);

    // 创建数据集
    double *dataset = new double[this->dimensions * dataSize];

    // 解析CSV数据集（以制表符"\t"作为列分隔符）
    string line;
    for (int i = 0; i < dataSize; i++) {
        getline(file, line);
        istringstream spliter(line);
        string cell;
        for (int j = 0; j < this->dimensions; j++) {
            spliter >> dataset[i * this->dimensions + j];
        }
    }

    return dataset;
}

const int *KMeans::predict(const double *const dataset, const int dataSize) {
    // 标签集
    int *labels = new int[dataSize];
    memset(labels, 0, static_cast<size_t>(dataSize) * INT);
    // 模型成本（上一次，当前）
    double *costs = new double[2] {0, 0};
    // 每一类的样本数量
    int *sampleCounts = new int[this->clusters];

    // 迭代聚类
    for (int i = 0; i < this->maxInterations; i++) {
        // 清空聚类结果
        memset(sampleCounts, 0, static_cast<size_t>(this->clusters) * INT);

        // 新的聚类中心
        double **nextMeans = new double *[this->clusters];
        // 清空聚类中心信息
        for (int j = 0; j < this->clusters; j++) {
            nextMeans[j] = new double[this->dimensions];
            memset(
                nextMeans[j], 0,
                static_cast<size_t>(this->dimensions) * DOUBLE
            );
        }
        // 刷新成本
        costs[0] = costs[1];
        costs[1] = 0;

        // 分类
        for (int j = 0; j < dataSize; j++) {
            // 当前样本
            double *sample = new double[this->dimensions];
            for (int k = 0; k < this->dimensions; k++) {
                sample[k] = dataset[j * this->dimensions + k];
            }

            // 聚类并累计模型成本
            costs[1] += getCost(sample, labels[j]);
            // 将当前样本归属至相应的聚类（计数用于后续计算平均成本）
            sampleCounts[labels[j]] += 1;

            // 累计当前聚类中所有样本的坐标信息（用于后续计算聚类中心）
            for (int k = 0; k < this->dimensions; k++) {
                nextMeans[labels[j]][k] += sample[k];
            }

            // 释放数组
            delete[] sample;
        }

        // 计算平均成本
        costs[1] /= dataSize;
        // 计算聚类中心
        for (int j = 0; j < this->clusters; j++) {
            if (sampleCounts[j] > 0) {
                for (int k = 0; k < this->dimensions; k++) {
                    this->means[j][k] = nextMeans[j][k] / sampleCounts[j];
                }
            }

            // 结果同步后，顺便释放空间
            delete[] nextMeans[j];
        }

        // 释放二维数组
        delete nextMeans;

        // 若成本变化小于精度限，则停止迭代
        if (fabs(costs[0] - costs[1]) < this->epsilon * costs[0]) {
            break;
        }
    }

    // 清理局部动态数组
    delete[] sampleCounts;
    delete[] costs;

    // 输出标签集
    return labels;
}

const double KMeans::getCost(const double *sample, int &label) {
    label = -1;
    double minDistance = INT_MAX;
    for (int i = 0; i < this->clusters; i++) {
        double temp = getDistance(sample, this->means[i], this->dimensions);
        if (temp < minDistance) {
            minDistance = temp;
            label = i;
        }
    }
    return minDistance;
}

const double KMeans::getDistance(
    const double *x, const double *y, const int dimensions) {
    double sumSquares = 0;
    for (int i = 0; i < dimensions; i++) {
        sumSquares += pow(x[i] - y[i], 2);
    }
    return sqrt(sumSquares);
}

ostream &operator<<(ostream &out, const KMeans *const kmeans) {
    out << "模型: K阶中心距聚类" << endl
        << "数据维度: " << kmeans->dimensions << endl
        << "聚类簇数量：" << kmeans->clusters << endl
        << "聚类中心：" << endl;
    for (int i = 0; i < kmeans->clusters; i++) {
        out << "(" << kmeans->means[i][0];
        for (int j = 1; j < kmeans->dimensions; j++) {
            out << ", " << kmeans->means[i][j];
        }
        out << ")" << endl;
    }
    return out;
}
