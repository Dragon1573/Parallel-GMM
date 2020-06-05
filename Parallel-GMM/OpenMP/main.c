/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright @ 2020 Dragon1573
*/
#include <stdlib.h>
#include <stdio.h>
#include "clustering.h"

int main(int argc, const char *argv[]) {
    /* 语法检查 */
    if (argc < 7) {
        fprintf(
            stderr,
            "Usage: OpenMP.exe <threads> <input> <output> "
            "<size> <dimensions> <clusters>\n"
        );
        return EXIT_FAILURE;
    }

    /* 解析参数 */
    threads = strtoull(argv[1], NULL, 10);
    dataSize = strtoull(argv[4], NULL, 10);
    dimensions = strtoull(argv[5], NULL, 10);
    clusters = strtoull(argv[6], NULL, 10);

    /* 载入文件 */
    loadFile(argv[2]);
    /* KMeans聚类（用于初始化GMM） */
    kMeans_clustering();
    /* GMM聚类 */
    gaussian_clustering();
    /* 输出结果 */
    saveFile(argv[3]);
    /* 清理开辟的数组空间 */
    cleanUp();

    return EXIT_SUCCESS;
}
