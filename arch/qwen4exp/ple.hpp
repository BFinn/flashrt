// SPDX-License-Identifier: Apache-2.0
// qwen4exp n-gram (PLE) table indexing. Each token reads (ngram - 1) * heads_per_ngram rows
// of per_layer_token_embd: for n = 2..ngram, the n-gram ending at the token is hashed as
// tok[0]*m[0] ^ tok[-1]*m[1] ^ ... and head h maps it to row  hash % vocab[h] + offset[h].
// Missing predecessors (before the sequence start) and anything at or before an earlier EOS
// read as EOS; the token's own EOS does not cut its context. Semantics follow llama.cpp's
// qwen4exp implementation (MIT), which follows the reference model.
#pragma once

#include <cstdint>
#include <vector>

namespace flashrt {
struct Gguf;
}

namespace flashrt::qwen4exp {

struct Ple {
    int ngram = 0, heads_per_ngram = 0, n_heads = 0;
    int32_t eos = 0;
    std::vector<uint64_t> mult;                  // [ngram]
    std::vector<uint64_t> offset, vocab;         // [n_heads]
    uint64_t row_bytes = 0;                      // bytes per table row
    uint64_t table_offset = 0;                   // absolute file offset of the table
    int table_shard = 0;
};

Ple parse_ple(const Gguf& g);

// Rows for the token at seq[pos], writing n_heads row ids to out.
void ple_rows(const Ple& p, const int32_t* seq, int64_t pos, uint32_t* out);

}  // namespace flashrt::qwen4exp
