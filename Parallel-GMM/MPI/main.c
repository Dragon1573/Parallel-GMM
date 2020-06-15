/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright @ 2020 Dragon1573
*/
#include "clustering.h"

int main(int argc, const char *argv[]) {
    /* 进入MPI并行环境 */
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &processors);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* 语法检查 */
    if (argc < 6) {
        if (rank == 0) {
            fprintf(stderr,
                "Usage: mpiexec.exe -n <threads> OpenMP.exe <input> <output> "
                "<size> <dimensions> <clusters>\n");
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return EXIT_FAILURE;
    }

    /* 解析参数 */
    dataSize = strtoull(argv[3], NULL, 10);
    dimensions = strtoull(argv[4], NULL, 10);
    clusters = strtoull(argv[5], NULL, 10);

    /* 检查数据分配，此处没有处理任意分配的问题，所以无法整除就直接报错 */
    if (dataSize % processors != 0) {
        if (rank == 0) {
            fprintf(stderr,
                "[ERROR] Unknown Partition: Datasets could not be delivered"
                " properly into all processors!\n");
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return EXIT_FAILURE;
    }

    /* 载入文件 */
    if (rank == 0) {
        loadFile(argv[1]);
    }

    /* KMeans聚类（用于初始化GMM） */
    kMeans_clustering();

    if (rank == 0) {
        /* GMM聚类 */
        gaussian_clustering();

        /* 输出结果 */
        saveFile(argv[2]);
    }

    /* 清理开辟的数组空间 */
    cleanUp();

    /* 退出并行环境 */
    MPI_Finalize();
    return EXIT_SUCCESS;
}
