// SPDX-License-Identifier: Apache-2.0
// flashrt-engine: the engine process. Speaks the JSON-lines protocol in docs/design.md on
// stdin/stdout. Phase 0: the handshake and control ops only; "generate" answers with an
// error until the P1 decode path lands.
#include <cstdio>
#include <iostream>
#include <string>

namespace {

// Minimal field extraction for the control ops. P1 replaces this with a real JSON parser
// (vendored, permissively licensed) once requests carry prompts and sampling settings.
std::string field(const std::string& line, const std::string& key) {
    const std::string pat = "\"" + key + "\"";
    size_t at = line.find(pat);
    if (at == std::string::npos) return {};
    at = line.find(':', at + pat.size());
    if (at == std::string::npos) return {};
    at = line.find_first_not_of(" \t", at + 1);
    if (at == std::string::npos) return {};
    if (line[at] == '"') {
        const size_t end = line.find('"', at + 1);
        return end == std::string::npos ? std::string{} : line.substr(at + 1, end - at - 1);
    }
    const size_t end = line.find_first_of(",}", at);
    return line.substr(at, end == std::string::npos ? std::string::npos : end - at);
}

void emit(const std::string& json) {
    std::fwrite(json.data(), 1, json.size(), stdout);
    std::fputc('\n', stdout);
    std::fflush(stdout);
}

}  // namespace

int main() {
    std::ios::sync_with_stdio(false);
    emit(R"({"ev":"ready","version":"0.0.1","arch":"none","max_context":0,"features":["stop"]})");

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        const std::string op = field(line, "op");
        const std::string id = field(line, "id");
        if (op == "quit") break;
        if (op == "stop") continue;   // nothing is running yet
        if (op == "generate") {
            emit(R"({"ev":"error","id":")" + id + R"(","msg":"generation is not implemented yet (phase 1)"})");
            continue;
        }
        emit(R"({"ev":"error","id":")" + id + R"(","msg":"unknown op"})");
    }
    return 0;
}
