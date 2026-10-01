// SPDX-License-Identifier: Apache-2.0
// core/row_reader on a synthetic file: random rows (duplicates, rows crossing a 4 KiB page, a
// base offset, row sizes that do not divide the page) must come back byte-exact, through many
// fetches in a row (the worker generations), and a row past the end of the file must throw.
#include "core/row_reader.hpp"

#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace flashrt;

namespace {
int failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::printf("FAIL %s\n", what.c_str());
        ++failures;
    }
}

uint8_t byte_at(uint64_t off) { return uint8_t((off * 2654435761u) >> 13); }
}  // namespace

int main() {
    const std::string path = (std::filesystem::temp_directory_path() / ("flashrt-test-rows-" + std::to_string(::getpid()))).string();
    const uint64_t size = 3 << 20;
    {
        std::vector<char> data(size);
        for (uint64_t i = 0; i < size; ++i) data[i] = char(byte_at(i));
        std::ofstream(path, std::ios::binary).write(data.data(), std::streamsize(size));
    }
    std::mt19937 rng(7);
    for (const auto& [base, row] : {std::pair<uint64_t, size_t>{0, 4096}, {100, 1000}, {4000, 3000}, {12345, 7}, {0, 1}, {2048, 4096}}) {
        RowReader rr(path, base, row, 8);
        const uint32_t n_rows = uint32_t((size - base) / row);
        const std::string tag = "base " + std::to_string(base) + ", row " + std::to_string(row);
        bool all_ok = true;
        size_t reads = 0, asked = 0;
        for (int f = 0; f < 200; ++f) {   // many fetches: every worker goes through every generation
            const size_t n = 1 + rng() % 96;
            std::vector<uint32_t> rows(n);
            for (auto& r : rows) r = rng() % 4 == 0 ? n_rows - 1 - rng() % 3 : rng() % n_rows;   // the last rows often
            if (n > 2) rows[1] = rows[0];                                                        // a duplicate
            std::vector<uint8_t> out(n * row, 0);
            reads += rr.fetch(rows.data(), n, out.data());
            asked += n;
            for (size_t i = 0; i < n && all_ok; ++i)
                for (size_t b = 0; b < row; ++b)
                    if (out[i * row + b] != byte_at(base + uint64_t(rows[i]) * row + b)) {
                        expect(false, tag + ": row " + std::to_string(rows[i]) + " byte " + std::to_string(b));
                        all_ok = false;
                        break;
                    }
        }
        expect(reads <= asked, tag + ": no more reads than rows");
        expect(rr.fetch(nullptr, 0, nullptr) == 0, tag + ": an empty fetch");
        uint32_t past[2] = {0, n_rows + 1};   // entirely past the end of the file
        bool threw = false;
        try {
            std::vector<uint8_t> out(2 * row);
            rr.fetch(past, 2, out.data());
        } catch (const std::runtime_error&) {
            threw = true;
        }
        expect(threw, tag + ": a row past the end throws");
        std::vector<uint8_t> out(row);   // and the reader still works after the error
        uint32_t one = n_rows / 2;
        rr.fetch(&one, 1, out.data());
        expect(out[0] == byte_at(base + uint64_t(one) * row), tag + ": works after an error");
    }
    bool threw = false;
    try {
        RowReader rr(path, 0, 8192, 1);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    expect(threw, "rejects rows above 4 KiB");
    std::filesystem::remove(path);
    std::printf("%s: %d failure(s)\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
