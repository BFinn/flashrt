// SPDX-License-Identifier: Apache-2.0
// Platform services: large host allocations (optionally on 2 MiB pages), physical-core
// topology and thread pinning. The expert arena and the CPU pool are built on these.
#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace flashrt {

enum class PageMode {
    Default,   // 4 KiB pages
    THP,       // transparent huge pages via madvise(MADV_HUGEPAGE); falls back silently
    HugeTLB,   // explicit MAP_HUGETLB 2 MiB pages; fails if the pool is too small
};

struct HostBuffer {
    void*       ptr   = nullptr;
    std::size_t bytes = 0;
    PageMode    mode  = PageMode::Default;
};

// touch_threads <= 0 skips the touch, for callers that write every page themselves.
// Anonymous mapping of `bytes` (rounded up to 2 MiB), touched by `touch_threads` threads so
// that pages are faulted in before any timing. Returns ptr == nullptr on failure.
HostBuffer host_alloc(std::size_t bytes, PageMode mode, int touch_threads = 1);
void       host_free(HostBuffer& buf);

// Fraction of the buffer backed by huge pages, from /proc/self/smaps (0 if unknown).
double huge_page_fraction(const HostBuffer& buf);

// One logical CPU per physical core, ordered by core id (SMT siblings dropped).
std::vector<int> physical_cpus();

// Pin the calling thread to one logical CPU. Returns false on failure.
bool pin_current_thread(int cpu);

PageMode    parse_page_mode(const std::string& s);   // "4k" | "thp" | "hugetlb"
const char* page_mode_name(PageMode m);

}  // namespace flashrt
