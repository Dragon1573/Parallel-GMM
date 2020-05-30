/**
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright @ 2020 Dragon1573
 */
#include "GMM-Serial.hpp"
#include "KMeans-Serial.hpp"
#include <fstream>
#include <iostream>

int main(int argc, const char *argv[]) {
    // 描述数据集
    const int dataSize = 9;
    const int dimensions = 2;
    const int clusters = 3;

    // 初始化数据集
    double *dataset = new double[dataSize * dimensions] {
        -0.1, 0.1,
        0.0, 0.0,
        0.1, -0.1,
        4.9, 5.1,
        5.0, 5.0,
        5.1, 4.9,
        9.9, 10.1,
        10.0, 10.0,
        10.1, 9.9
    };

    // 初始化模型
    GMM *gmm = new GMM(dimensions, 3);
    // 训练模型
    gmm->fit(dataset, dataSize);
    // 输出模型信息
    cout << gmm << endl;
    // 输出聚类结果
    for (int i = 0; i < dataSize; i++) {
        // 提取样本点
        double *const sample = new double[dimensions];
        for (int j = 0; j < dimensions; j++) {
            sample[j] = dataset[i * dimensions + j];
        }

        cout << "(" << sample[0];
        for (int j = 1; j < dimensions; j++) {
            cout << ", " << sample[j];
        }
        cout << ") = " << gmm->getProbability(sample) << endl;
    }

    // 释放模型
    delete gmm;

    return EXIT_SUCCESS;
}
