// SPDX-License-Identifier: Apache-2.0
//
// The OpenSearch Contributors require contributions made to
// this file be licensed under the Apache-2.0 license or a
// compatible open source license.
//
// Modifications Copyright OpenSearch Contributors. See
// GitHub history for details.

#include "faiss_cagra_builder.h"

#include <omp.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <queue>
#include <utility>
#include <vector>

#include "faiss/IndexScalarQuantizer.h"
#include "faiss/impl/FaissAssert.h"
#include "faiss/impl/ScalarQuantizer.h"

#ifdef KNN_WITH_MKL
#include <mkl.h>
#endif

namespace knn_jni {
namespace cagra_builder {

namespace {

// GEMM is O(N^2): beyond the HNSW-incremental crossover (~300-400k measured)
// batch build loses, so fall back to incremental above this many vectors.
constexpr int kDefaultMaxN = 300000;

int maxBatchN() {
    if (const char* env = std::getenv("KNN_CAGRA_MAX_N")) {
        int v = std::atoi(env);
        if (v > 0) {
            return v;
        }
    }
    return kDefaultMaxN;
}

// Accepts exactly "HNSW<M>,SQbf16" ("HNSW,SQbf16" -> faiss default M=32).
bool parseHnswSqBf16(const std::string& desc, int* mOut) {
    if (desc.rfind("HNSW", 0) != 0) {
        return false;
    }
    size_t comma = desc.find(',');
    if (comma == std::string::npos || desc.substr(comma + 1) != "SQbf16") {
        return false;
    }
    std::string mStr = desc.substr(4, comma - 4);
    int m = mStr.empty() ? 32 : std::atoi(mStr.c_str());
    if (m <= 0) {
        return false;
    }
    *mOut = m;
    return true;
}

}  // namespace

faiss::IndexHNSW* tryCreateCagraBatchIndex(
        const std::string& indexDescription,
        int dim,
        faiss::MetricType metric,
        int numVectors) {
#ifndef KNN_WITH_MKL
    (void)indexDescription;
    (void)dim;
    (void)metric;
    (void)numVectors;
    std::fprintf(stderr, "[KNN-CAGRA] built without MKL, falling back to incremental build\n");
    return nullptr;
#else
    int m = 0;
    if (metric != faiss::METRIC_INNER_PRODUCT || !parseHnswSqBf16(indexDescription, &m)) {
        std::fprintf(
                stderr,
                "[KNN-CAGRA] description '%s' / metric %d not eligible (need HNSW<M>,SQbf16 + inner product), "
                "falling back to incremental build\n",
                indexDescription.c_str(),
                (int)metric);
        return nullptr;
    }
    if (numVectors > maxBatchN()) {
        std::fprintf(
                stderr,
                "[KNN-CAGRA] numVectors=%d exceeds max batch N=%d (O(N^2) GEMM), "
                "falling back to incremental build\n",
                numVectors,
                maxBatchN());
        return nullptr;
    }

    // The IndexHNSWCagra constructor only offers fp32/fp16 storage, so create
    // with fp32 and swap in a bf16 scalar quantizer (codes are then directly
    // usable as MKL_BF16 GEMM input). own_fields is set by the constructor.
    auto* cagra = new faiss::IndexHNSWCagra(dim, m, metric);
    delete cagra->storage;
    cagra->storage = new faiss::IndexScalarQuantizer(dim, faiss::ScalarQuantizer::QT_bf16, metric);
    // Graph is assembled flat (level 0 only) at writeIndex time; this also
    // makes accidental add() on the HNSW index throw instead of building.
    cagra->base_level_only = true;
    cagra->num_base_level_search_entrypoints = 32;
    cagra->is_trained = true;
    return cagra;
#endif
}

faiss::IndexHNSWCagra* asCagraBatchIndex(faiss::IndexIDMap* idMap) {
    if (idMap == nullptr) {
        return nullptr;
    }
    return dynamic_cast<faiss::IndexHNSWCagra*>(idMap->index);
}

void insertStorageOnly(
        faiss::IndexIDMap* idMap,
        faiss::IndexHNSWCagra* cagra,
        int numVectors,
        const float* vectors,
        const int64_t* ids) {
    // Encode fp32 chunk into bf16 SQ codes; the graph is built in finalizeGraph.
    cagra->storage->add(numVectors, vectors);
    cagra->ntotal = cagra->storage->ntotal;
    // Mirror IndexIDMap::add_with_ids bookkeeping.
    for (int i = 0; i < numVectors; i++) {
        idMap->id_map.push_back(ids[i]);
    }
    idMap->ntotal = cagra->ntotal;
}

void finalizeGraph(faiss::IndexHNSWCagra* cagra) {
#ifndef KNN_WITH_MKL
    (void)cagra;
    FAISS_THROW_MSG("cagra batch build requires an MKL-enabled build (KNN_LINK_MKL)");
#else
    // MKL must share faiss' OpenMP runtime (libgomp); the default Intel layer
    // would pull a second OpenMP runtime into the process.
    static bool mklInit = [] {
        mkl_set_threading_layer(MKL_THREADING_GNU);
        return true;
    }();
    (void)mklInit;

    auto* sq = dynamic_cast<faiss::IndexScalarQuantizer*>(cagra->storage);
    FAISS_THROW_IF_NOT_MSG(sq != nullptr, "cagra batch build expects scalar-quantizer storage");
    FAISS_THROW_IF_NOT_MSG(sq->sq.qtype == faiss::ScalarQuantizer::QT_bf16, "cagra batch build expects bf16 codes");

    const int64_t n = sq->ntotal;
    const int64_t d = sq->d;
    if (n == 0) {
        return;
    }

    // All points live on level 0 (CAGRA-style flat graph, searched via random
    // entry points because base_level_only is set).
    cagra->hnsw.levels.assign(n, 1);
    cagra->hnsw.prepare_level_tab(n, /*preset_levels=*/true);

    const int nb0 = cagra->hnsw.nb_neighbors(0);  // = 2M edges on level 0
    const int kg = (int)std::min<int64_t>(std::max(64, 2 * nb0), n - 1);
    if (kg <= 0) {
        return;  // single vector: nothing to link
    }

    double t0 = omp_get_wtime();

    // bf16 codes are row-major [n x d] uint16 == the MKL_BF16 GEMM input.
    const MKL_BF16* codes = reinterpret_cast<const MKL_BF16*>(sq->codes.data());

    std::vector<int64_t> knnI((size_t)n * kg);
    std::vector<float> knnD((size_t)n * kg);

    // Blocked scores = X_block * X^T, scratch capped at ~1 GiB.
    int64_t block = 2048;
    while (block > 256 && block * n * (int64_t)sizeof(float) > ((int64_t)1 << 30)) {
        block /= 2;
    }
    block = std::min(block, n);
    std::vector<float> scores((size_t)block * n);

    for (int64_t i0 = 0; i0 < n; i0 += block) {
        const int64_t nb = std::min(block, n - i0);
        cblas_gemm_bf16bf16f32(
                CblasRowMajor,
                CblasNoTrans,
                CblasTrans,
                (MKL_INT)nb,
                (MKL_INT)n,
                (MKL_INT)d,
                1.0f,
                codes + (size_t)i0 * d,
                (MKL_INT)d,
                codes,
                (MKL_INT)d,
                0.0f,
                scores.data(),
                (MKL_INT)n);

#pragma omp parallel for schedule(dynamic, 8)
        for (int64_t r = 0; r < nb; r++) {
            const float* row = scores.data() + (size_t)r * n;
            const int64_t self = i0 + r;
            // top-kg by inner product via min-heap, self excluded
            using P = std::pair<float, int64_t>;
            std::priority_queue<P, std::vector<P>, std::greater<P>> heap;
            for (int64_t j = 0; j < n; j++) {
                if (j == self) {
                    continue;
                }
                const float s = row[j];
                if ((int)heap.size() < kg) {
                    heap.emplace(s, j);
                } else if (s > heap.top().first) {
                    heap.pop();
                    heap.emplace(s, j);
                }
            }
            int cnt = (int)heap.size();
            for (int t = cnt - 1; t >= 0; t--) {
                // distance convention for init_level_0_from_knngraph: -IP
                knnI[(size_t)self * kg + t] = heap.top().second;
                knnD[(size_t)self * kg + t] = -heap.top().first;
                heap.pop();
            }
            for (int t = cnt; t < kg; t++) {
                knnI[(size_t)self * kg + t] = -1;
                knnD[(size_t)self * kg + t] = 0.0f;
            }
        }
    }
    double t1 = omp_get_wtime();

    // faiss-side pruning (shrink_neighbor_list) + level-0 assembly
    cagra->init_level_0_from_knngraph(kg, knnD.data(), knnI.data());
    double t2 = omp_get_wtime();

    std::fprintf(
            stderr,
            "[KNN-CAGRA] batch build n=%lld d=%lld kg=%d gemm+topk=%.2fs assemble=%.2fs threads=%d\n",
            (long long)n,
            (long long)d,
            kg,
            t1 - t0,
            t2 - t1,
            omp_get_max_threads());
#endif
}

}  // namespace cagra_builder
}  // namespace knn_jni
