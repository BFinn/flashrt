// SPDX-License-Identifier: Apache-2.0
// fr_ple: n-gram (PLE) table reads for a real prompt, from the SSD at high queue depth.
//
//   fr_ple MODEL.gguf --ids PROMPT.txt [--n N] [--depth D] [--decode-tokens T]
//
// Computes the 16 rows per token for the first N prompt tokens, reports how many distinct
// rows and pages they touch, then times cold O_DIRECT fetches with RowReader: the whole
// prompt at once, in 2,048-token chunks, and one token at a time (decode). Verifies sampled
// rows against buffered reads of the same bytes.
#include "arch/qwen4exp/ple.hpp"
#include "core/gguf.hpp"
#include "core/row_reader.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

using namespace flashrt;
using Clock = std::chrono::steady_clock;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: fr_ple MODEL.gguf --ids PROMPT.txt [--n N] [--depth D] [--decode-tokens T]\n");
        return 2;
    }
    std::string ids_path;
    long n = 32000;
    int depth = 64, decode_tokens = 512;
    for (int i = 2; i + 1 < argc; i += 2) {
        if (!std::strcmp(argv[i], "--ids")) ids_path = argv[i + 1];
        else if (!std::strcmp(argv[i], "--n")) n = std::atol(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--depth")) depth = std::atoi(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--decode-tokens")) decode_tokens = std::atoi(argv[i + 1]);
        else { std::fprintf(stderr, "unknown argument %s\n", argv[i]); return 2; }
    }
    std::vector<int32_t> seq;
    {
        std::ifstream f(ids_path);
        long v;
        while (f >> v) seq.push_back(int32_t(v));
    }
    if (seq.empty()) { std::fprintf(stderr, "no token ids in %s\n", ids_path.c_str()); return 1; }
    n = std::min<long>(n, long(seq.size()));

    const Gguf g = Gguf::open(argv[1]);
    const qwen4exp::Ple p = qwen4exp::parse_ple(g);
    std::printf("PLE: %d-gram, %d heads, %llu B rows, table at offset %llu of shard %d\n", p.ngram, p.n_heads,
                (unsigned long long) p.row_bytes, (unsigned long long) p.table_offset, p.table_shard);

    std::vector<uint32_t> rows(size_t(n) * p.n_heads);
    for (long i = 0; i < n; ++i) qwen4exp::ple_rows(p, seq.data(), i, rows.data() + size_t(i) * p.n_heads);
    std::unordered_set<uint32_t> distinct(rows.begin(), rows.end());
    std::unordered_set<uint64_t> pages;
    for (uint32_t r : rows) pages.insert((p.table_offset + uint64_t(r) * p.row_bytes) >> 12);
    std::printf("prompt: %ld tokens, %zu row reads, %zu distinct rows (%.1f%%), %zu distinct pages\n", n, rows.size(),
                distinct.size(), 100.0 * distinct.size() / rows.size(), pages.size());

    RowReader rr(g.shards[p.table_shard], p.table_offset, p.row_bytes, depth);
    std::vector<uint8_t> out(rows.size() * p.row_bytes);

    auto t0 = Clock::now();
    size_t reads = rr.fetch(rows.data(), rows.size(), out.data());
    double dt = std::chrono::duration<double>(Clock::now() - t0).count();
    std::printf("whole prompt at depth %d: %zu device reads in %.3f s (%.0f reads/s, %.0f tokens/s)\n", depth, reads, dt,
                reads / dt, n / dt);

    t0 = Clock::now();
    reads = 0;
    const long chunk = 2048;
    for (long c = 0; c < n; c += chunk) {
        const long m = std::min(chunk, n - c);
        reads += rr.fetch(rows.data() + size_t(c) * p.n_heads, size_t(m) * p.n_heads, out.data() + size_t(c) * p.n_heads * p.row_bytes);
    }
    dt = std::chrono::duration<double>(Clock::now() - t0).count();
    std::printf("2048-token chunks: %zu device reads in %.3f s (%.0f tokens/s)\n", reads, dt, n / dt);

    std::vector<double> lat;
    std::vector<uint8_t> one(size_t(p.n_heads) * p.row_bytes);
    for (int t = 0; t < std::min<long>(decode_tokens, n); ++t) {
        const auto a = Clock::now();
        rr.fetch(rows.data() + size_t(t) * p.n_heads, p.n_heads, one.data());
        lat.push_back(std::chrono::duration<double, std::micro>(Clock::now() - a).count());
    }
    std::sort(lat.begin(), lat.end());
    std::printf("decode (one token, %d rows): p50 %.0f us, p90 %.0f us, p99 %.0f us over %zu tokens\n", p.n_heads,
                lat[lat.size() / 2], lat[lat.size() * 9 / 10], lat[lat.size() * 99 / 100], lat.size());

    // verify against buffered reads
    const int fd = open(g.shards[p.table_shard].c_str(), O_RDONLY);
    std::mt19937 rng(3);
    int bad = 0;
    std::vector<uint8_t> want(p.row_bytes);
    for (int k = 0; k < 2000; ++k) {
        const size_t i = rng() % rows.size();
        if (pread(fd, want.data(), p.row_bytes, off_t(p.table_offset + uint64_t(rows[i]) * p.row_bytes)) != ssize_t(p.row_bytes)) {
            std::perror("pread");
            return 1;
        }
        bad += std::memcmp(want.data(), out.data() + i * p.row_bytes, p.row_bytes) != 0;
    }
    close(fd);
    std::printf("verify: 2000 sampled rows, %d mismatches\n%s\n", bad, bad ? "fr_ple FAILED" : "fr_ple OK");
    return bad ? 1 : 0;
}
