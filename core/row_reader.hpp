// SPDX-License-Identifier: Apache-2.0
// RowReader: fetch fixed-size rows from a large file at high queue depth, bypassing the page
// cache. Used for the n-gram (PLE) table, which is too big for RAM beside the expert arena.
//
// A persistent pool of `depth` threads issues blocking O_DIRECT preads, so `depth` reads are
// in flight at once (P0: a 990 PRO gives ~16K IOPS at depth 1 and ~714K at depth 64). Rows
// that share a 4 KiB page are served by one read.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace flashrt {

class RowReader {
public:
    // Rows of `row_bytes` start at byte `base` of `path`.
    RowReader(const std::string& path, uint64_t base, size_t row_bytes, int depth);
    ~RowReader();
    RowReader(const RowReader&) = delete;
    RowReader& operator=(const RowReader&) = delete;

    // Copies row rows[i] to out + i * row_bytes, for i < n. Blocks until all are read.
    // Returns the number of device reads issued (after merging rows that share a page).
    size_t fetch(const uint32_t* rows, size_t n, uint8_t* out);

    size_t row_bytes() const { return row_bytes_; }

private:
    struct Read {
        uint64_t page_off;   // aligned file offset
        uint32_t span;       // bytes to read (4 or 8 KiB when a row crosses a page)
        uint32_t first, count;   // range in order_
    };
    void worker();

    int fd_ = -1;
    uint64_t base_;
    size_t row_bytes_;
    std::vector<std::thread> threads_;
    std::mutex mu_;
    std::condition_variable cv_, done_cv_;
    uint64_t generation_ = 0;
    bool quit_ = false;
    int busy_ = 0;

    // the current job
    const uint32_t* rows_ = nullptr;
    uint8_t* out_ = nullptr;
    std::vector<uint32_t> order_;   // indices into rows_, sorted by file offset
    std::vector<Read> reads_;
    std::atomic<size_t> next_{0};
    std::atomic<int> error_{0};
};

}  // namespace flashrt
