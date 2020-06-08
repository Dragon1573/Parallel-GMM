/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright @ 2020 Dragon1573
*/
#include "clustering.h"

int mian(int argc, const char *argv[]) {
    /* 语法检查 */
    if (argc < 6) {
        fprintf(
            stderr,
            "Usage: mpiexec.exe -n <threads> OpenMP.exe <input> <output> "
            "<size> <dimensions> <clusters>\n"
        );
        return EXIT_FAILURE;
    }

    /* 解析参数 */
    dataSize = strtoull(argv[3], NULL, 10);
    dimensions = strtoull(argv[4], NULL, 10);
    clusters = strtoull(argv[5], NULL, 10);

    /* 进入MPI并行环境 */
    MPI_Init(&argc, &argv);
    int processors = 0, rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &processors);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* 载入文件 */
    if (rank == 0) {
        loadFile(argv[1]);
    }
    /* KMeans聚类（用于初始化GMM） */
    kMeans_clustering();
    /* GMM聚类 */
    gaussian_clustering();
    if (rank == 0) {
        /* 输出结果 */
        saveFile(argv[2]);
        /* 清理开辟的数组空间 */
        cleanUp();
    }

    return EXIT_SUCCESS;
}
