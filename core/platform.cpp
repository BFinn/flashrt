// SPDX-License-Identifier: Apache-2.0
#include "core/platform.hpp"

#include <sched.h>
#include <sys/mman.h>
#include <pthread.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <thread>
#include <utility>

namespace flashrt {

namespace {
constexpr std::size_t kHuge = std::size_t(2) << 20;

std::size_t round_up(std::size_t x, std::size_t a) { return (x + a - 1) / a * a; }

int read_int(const std::string& path, int fallback) {
    std::ifstream f(path);
    int v = fallback;
    if (f) f >> v;
    return v;
}
}  // namespace

HostBuffer host_alloc(std::size_t bytes, PageMode mode, int touch_threads) {
    HostBuffer buf;
    buf.bytes = round_up(bytes, kHuge);
    buf.mode  = mode;

    int flags = MAP_PRIVATE | MAP_ANONYMOUS;
    if (mode == PageMode::HugeTLB) flags |= MAP_HUGETLB;
    void* p = mmap(nullptr, buf.bytes, PROT_READ | PROT_WRITE, flags, -1, 0);
    if (p == MAP_FAILED) {
        buf.ptr = nullptr;
        return buf;
    }
    if (mode == PageMode::THP) madvise(p, buf.bytes, MADV_HUGEPAGE);
    buf.ptr = p;

    // first touch in parallel: faults the pages in now, not inside a timed loop
    if (touch_threads <= 0) return buf;   // caller first-touches (e.g. a parallel loader)
    const std::size_t chunk = round_up(buf.bytes / touch_threads, kHuge);
    std::vector<std::thread> ts;
    for (int t = 0; t < touch_threads; ++t) {
        const std::size_t off = std::size_t(t) * chunk;
        if (off >= buf.bytes) break;
        const std::size_t len = std::min(chunk, buf.bytes - off);
        ts.emplace_back([=] { std::memset(static_cast<char*>(p) + off, 1, len); });
    }
    for (auto& t : ts) t.join();
    return buf;
}

void host_free(HostBuffer& buf) {
    if (buf.ptr) munmap(buf.ptr, buf.bytes);
    buf = HostBuffer{};
}

double huge_page_fraction(const HostBuffer& buf) {
    if (!buf.ptr) return 0.0;
    std::ifstream f("/proc/self/smaps");
    std::string line;
    const auto base = reinterpret_cast<std::uintptr_t>(buf.ptr);
    bool in = false;
    std::size_t anon_huge_kb = 0, hugetlb_kb = 0;
    while (std::getline(f, line)) {
        std::uintptr_t lo = 0, hi = 0;
        char dash = 0;
        std::istringstream ss(line);
        if (ss >> std::hex >> lo >> dash >> hi && dash == '-') {   // a mapping header line
            in = lo >= base && hi <= base + buf.bytes;
            continue;
        }
        if (!in) continue;
        std::size_t kb = 0;
        if (std::sscanf(line.c_str(), "AnonHugePages: %zu kB", &kb) == 1) anon_huge_kb += kb;
        if (std::sscanf(line.c_str(), "Private_Hugetlb: %zu kB", &kb) == 1) hugetlb_kb += kb;
    }
    return double(anon_huge_kb + hugetlb_kb) * 1024.0 / double(buf.bytes);
}

std::vector<int> physical_cpus() {
    // (package, core) -> lowest logical cpu with that pair
    std::map<std::pair<int, int>, int> first;
    const int n = int(std::thread::hardware_concurrency());
    for (int cpu = 0; cpu < n; ++cpu) {
        const std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/";
        const int core = read_int(base + "core_id", cpu);
        const int pkg  = read_int(base + "physical_package_id", 0);
        auto key = std::make_pair(pkg, core);
        if (!first.count(key)) first[key] = cpu;
    }
    std::vector<int> out;
    for (auto& [key, cpu] : first) out.push_back(cpu);
    std::sort(out.begin(), out.end());
    return out;
}

bool pin_current_thread(int cpu) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    return pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0;
}

bool unpin_current_thread() {
    cpu_set_t set;
    CPU_ZERO(&set);
    const int n = int(std::thread::hardware_concurrency());
    for (int cpu = 0; cpu < n && cpu < CPU_SETSIZE; ++cpu) CPU_SET(cpu, &set);
    return pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0;
}

PageMode parse_page_mode(const std::string& s) {
    if (s == "thp") return PageMode::THP;
    if (s == "hugetlb") return PageMode::HugeTLB;
    return PageMode::Default;
}

const char* page_mode_name(PageMode m) {
    switch (m) {
        case PageMode::THP: return "thp";
        case PageMode::HugeTLB: return "hugetlb";
        default: return "4k";
    }
}

}  // namespace flashrt
