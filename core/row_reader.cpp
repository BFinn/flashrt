// SPDX-License-Identifier: Apache-2.0
#include "core/row_reader.hpp"

#include "core/platform.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <stdexcept>

#ifndef O_DIRECT
#define O_DIRECT 0   // macOS builds (tests only)
#endif

namespace flashrt {

namespace {
constexpr uint64_t kPage = 4096;
}

RowReader::RowReader(const std::string& path, uint64_t base, size_t row_bytes, int depth)
    : base_(base), row_bytes_(row_bytes) {
    if (row_bytes == 0 || row_bytes > kPage) throw std::runtime_error("RowReader: row size must be 1..4096 bytes");
    fd_ = open(path.c_str(), O_RDONLY | O_DIRECT);
    if (fd_ < 0) throw std::runtime_error("RowReader: open " + path + ": " + std::strerror(errno));
    for (int i = 0; i < std::max(1, depth); ++i) threads_.emplace_back([this] { worker(); });
}

RowReader::~RowReader() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        quit_ = true;
    }
    cv_.notify_all();
    for (auto& t : threads_) t.join();
    if (fd_ >= 0) close(fd_);
}

size_t RowReader::fetch(const uint32_t* rows, size_t n, uint8_t* out) {
    if (n == 0) return 0;
    // sort by offset and merge rows whose bytes lie in the same page (or page pair)
    order_.resize(n);
    std::iota(order_.begin(), order_.end(), 0u);
    std::sort(order_.begin(), order_.end(), [&](uint32_t a, uint32_t b) { return rows[a] < rows[b]; });
    reads_.clear();
    for (uint32_t i = 0; i < n;) {
        const uint64_t off = base_ + uint64_t(rows[order_[i]]) * row_bytes_;
        const uint64_t page = off & ~(kPage - 1);
        uint64_t end = off + row_bytes_;
        uint32_t j = i + 1;
        while (j < n) {
            const uint64_t o = base_ + uint64_t(rows[order_[j]]) * row_bytes_;
            if ((o & ~(kPage - 1)) != page) break;
            end = std::max(end, o + row_bytes_);
            ++j;
        }
        const uint32_t span = uint32_t(((end - page) + kPage - 1) & ~(kPage - 1));
        reads_.push_back(Read{page, span, i, j - i});
        i = j;
    }
    {
        std::unique_lock<std::mutex> lk(mu_);
        rows_ = rows;
        out_ = out;
        next_.store(0);
        error_.store(0);
        ++generation_;
        busy_ = int(threads_.size());
    }
    cv_.notify_all();
    {
        std::unique_lock<std::mutex> lk(mu_);
        done_cv_.wait(lk, [&] { return busy_ == 0; });
    }
    if (error_.load()) throw std::runtime_error(std::string("RowReader: read failed: ") + std::strerror(error_.load()));
    return reads_.size();
}

void RowReader::worker() {
    unpin_current_thread();   // not on the creator's (possibly pinned) CPU
    uint8_t* buf = static_cast<uint8_t*>(std::aligned_alloc(kPage, 2 * kPage));
    uint64_t seen = 0;
    for (;;) {
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [&] { return quit_ || generation_ != seen; });
            if (quit_) break;
            seen = generation_;
        }
        for (size_t k; (k = next_.fetch_add(1)) < reads_.size();) {
            const Read& r = reads_[k];
            const ssize_t got = pread(fd_, buf, r.span, off_t(r.page_off));
            if (got < 0) {
                error_.store(errno);
                continue;
            }
            for (uint32_t m = r.first; m < r.first + r.count; ++m) {
                const uint32_t idx = order_[m];
                const uint64_t off = base_ + uint64_t(rows_[idx]) * row_bytes_;
                if (off + row_bytes_ > r.page_off + uint64_t(got)) {
                    error_.store(EIO);
                    continue;
                }
                std::memcpy(out_ + size_t(idx) * row_bytes_, buf + (off - r.page_off), row_bytes_);
            }
        }
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (--busy_ == 0) done_cv_.notify_one();
        }
    }
    std::free(buf);
}

}  // namespace flashrt
