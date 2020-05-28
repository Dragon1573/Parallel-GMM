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
    // ָ�����ݼ�ά��
    this->dimensions = dimensions;
    // ָ����������
    this->clusters = clusters;

    // ��ʼ���������飬����ÿһ���һ���洢�˸�ά��ֵ������
    this->means = new double *[this->clusters];
    for (int i = 0; i < this->clusters; i++) {
        // ��ʼ���������ģ��������ľ������Ķ�λ�ڳ��ռ�ԭ��
        this->means[i] = new double[this->dimensions];
        memset(this->means[i], 0, static_cast<size_t>(this->dimensions) * DOUBLE);
    }

    // �����ʼ������
    this->initMode = InitMode::Randomly;
    // ��������������100��
    this->maxInterations = 100;
    // ����������Ϊ0.1%
    this->epsilon = 0.001;
}

KMeans::~KMeans() {
    // �� new �ؼ��ֿ��ٵĶ���������Ĭ���������ͷ�
    for (int i = 0; i < clusters; i++) {
        // ���ͷ��ڲ������
        delete[] this->means[i];
    }
    // ���ͷ���������
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
    case Randomly: // ���ѡȡ������1��������Ϊ��������
    {
        // ��ȡѵ����ÿ��������������
        int interval = dataSize / this->clusters;
        // �Ե�ǰʱ���ʼ��α�����������
        srand((unsigned int)time(NULL));

        for (int i = 0; i < clusters; i++) {
            // ���ѡȡһ��������
            int selection = (int)((interval - 1) * (double)rand() / RAND_MAX);
            selection += interval * i;
            // ��ȡ���ݲ�
            for (int j = 0; j < this->dimensions; j++) {
                this->means[i][j] = datasets[selection * this->dimensions + j];
            }
        }

        break;
    }
    case Uniformly: // ��������׸�������Ϊ��������
    {
        for (int i = 0; i < this->clusters; i++) {
            int selection = i * dataSize / this->clusters;
            for (int j = 0; j < this->dimensions; j++) {
                this->means[i][j] = datasets[selection * this->dimensions + j];
            }
        }
        break;
    }
    case Manually: // �û����г�ʼ��
        // ʲô������
        break;
    }

    return predict(datasets, dataSize);
}

const double *const KMeans::loadFile(
    ifstream &file, const int dataSize
) {
    // ����������Ҫ�;�������൱
    assert(dataSize >= this->clusters);

    // �������ݼ�
    double *dataset = new double[this->dimensions * dataSize];

    // ����CSV���ݼ������Ʊ��"\t"��Ϊ�зָ�����
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
    // ��ǩ��
    int *labels = new int[dataSize];
    memset(labels, 0, static_cast<size_t>(dataSize) * INT);
    // ģ�ͳɱ�����һ�Σ���ǰ��
    double *costs = new double[2] {0, 0};
    // ÿһ�����������
    int *sampleCounts = new int[this->clusters];

    // ��������
    for (int i = 0; i < this->maxInterations; i++) {
        // ��վ�����
        memset(sampleCounts, 0, static_cast<size_t>(this->clusters) * INT);

        // �µľ�������
        double **nextMeans = new double *[this->clusters];
        // ��վ���������Ϣ
        for (int j = 0; j < this->clusters; j++) {
            nextMeans[j] = new double[this->dimensions];
            memset(
                nextMeans[j], 0,
                static_cast<size_t>(this->dimensions) * DOUBLE
            );
        }
        // ˢ�³ɱ�
        costs[0] = costs[1];
        costs[1] = 0;

        // ����
        for (int j = 0; j < dataSize; j++) {
            // ��ǰ����
            double *sample = new double[this->dimensions];
            for (int k = 0; k < this->dimensions; k++) {
                sample[k] = dataset[j * this->dimensions + k];
            }

            // ���ಢ�ۼ�ģ�ͳɱ�
            costs[1] += getCost(sample, labels[j]);
            // ����ǰ������������Ӧ�ľ��ࣨ�������ں�������ƽ���ɱ���
            sampleCounts[labels[j]] += 1;

            // �ۼƵ�ǰ����������������������Ϣ�����ں�������������ģ�
            for (int k = 0; k < this->dimensions; k++) {
                nextMeans[labels[j]][k] += sample[k];
            }

            // �ͷ�����
            delete[] sample;
        }

        // ����ƽ���ɱ�
        costs[1] /= dataSize;
        // �����������
        for (int j = 0; j < this->clusters; j++) {
            if (sampleCounts[j] > 0) {
                for (int k = 0; k < this->dimensions; k++) {
                    this->means[j][k] = nextMeans[j][k] / sampleCounts[j];
                }
            }

            // ���ͬ����˳���ͷſռ�
            delete[] nextMeans[j];
        }

        // �ͷŶ�ά����
        delete nextMeans;

        // ���ɱ��仯С�ھ����ޣ���ֹͣ����
        if (fabs(costs[0] - costs[1]) < this->epsilon * costs[0]) {
            break;
        }
    }

    // ����ֲ���̬����
    delete[] sampleCounts;
    delete[] costs;

    // �����ǩ��
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
    out << "ģ��: K�����ľ����" << endl
        << "����ά��: " << kmeans->dimensions << endl
        << "�����������" << kmeans->clusters << endl
        << "�������ģ�" << endl;
    for (int i = 0; i < kmeans->clusters; i++) {
        out << "(" << kmeans->means[i][0];
        for (int j = 1; j < kmeans->dimensions; j++) {
            out << ", " << kmeans->means[i][j];
        }
        out << ")" << endl;
    }
    return out;
}
