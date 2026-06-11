// SPDX-License-Identifier: Apache-2.0
//
// The OpenSearch Contributors require contributions made to
// this file be licensed under the Apache-2.0 license or a
// compatible open source license.
//
// Modifications Copyright OpenSearch Contributors. See
// GitHub history for details.

/**
 * CAGRA-like batch build for HNSW (build_method=cagra_cpu).
 *
 * Instead of incremental greedy insertion, the kNN graph is computed in one
 * batched bf16 GEMM over the SQbf16 storage codes (MKL dispatches AMX tiles on
 * GNR/SPR), then pruned/assembled with faiss' own shrink_neighbor_list via
 * init_level_0_from_knngraph. The produced index is an IndexHNSWCagra with
 * base_level_only=true and SQbf16 storage.
 *
 * This header is JNI-free on purpose so the builder can be exercised by
 * standalone native tests.
 */

#ifndef OPENSEARCH_KNN_FAISS_CAGRA_BUILDER_H
#define OPENSEARCH_KNN_FAISS_CAGRA_BUILDER_H

#include <cstdint>
#include <string>

#include "faiss/IndexHNSW.h"
#include "faiss/IndexIDMap.h"
#include "faiss/MetricType.h"

namespace knn_jni {
namespace cagra_builder {

/**
 * Try to create an index eligible for CAGRA-like batch build.
 *
 * Eligibility: indexDescription == "HNSW<M>,SQbf16", metric == inner product,
 * numVectors <= max-N guard (KNN_CAGRA_MAX_N env, default 300000, GEMM is
 * O(N^2)) and the library was built with MKL.
 *
 * @return a new IndexHNSWCagra (bf16 SQ storage, base_level_only) or nullptr
 *         when not eligible; the caller then falls back to the regular
 *         incremental factory path.
 */
faiss::IndexHNSW* tryCreateCagraBatchIndex(
        const std::string& indexDescription,
        int dim,
        faiss::MetricType metric,
        int numVectors);

/**
 * @return the wrapped IndexHNSWCagra if idMap holds a CAGRA batch-build index,
 *         nullptr otherwise. Used by insert/write hooks to divert.
 */
faiss::IndexHNSWCagra* asCagraBatchIndex(faiss::IndexIDMap* idMap);

/**
 * insertToIndex hook: encode the chunk into the SQ storage only (fp32 -> bf16
 * codes); no graph is built. Mirrors IndexIDMap::add_with_ids bookkeeping.
 */
void insertStorageOnly(
        faiss::IndexIDMap* idMap,
        faiss::IndexHNSWCagra* cagra,
        int numVectors,
        const float* vectors,
        const int64_t* ids);

/**
 * writeIndex hook: compute the exact kNN graph with one blocked bf16 GEMM over
 * the storage codes (top-KG per row, self excluded), then assemble HNSW
 * level-0 via prepare_level_tab + init_level_0_from_knngraph.
 */
void finalizeGraph(faiss::IndexHNSWCagra* cagra);

}  // namespace cagra_builder
}  // namespace knn_jni

#endif  // OPENSEARCH_KNN_FAISS_CAGRA_BUILDER_H
