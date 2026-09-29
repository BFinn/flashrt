// SPDX-License-Identifier: Apache-2.0
// flashrt-engine: the engine process. Speaks the JSON-lines protocol of docs/design.md on stdin
// and stdout (one object per line); logs go to stderr.
//
//   flashrt-engine MODEL.gguf [--mtp DRAFT.gguf [--spec K] [--draft-vocab RANKS] [--mtp-bits B]]
//                  [--ctx N] [--kv-hot BLOCKS] [--kv f16|q8] [--workers W] [--reserve-mib R] [--cache-prior FILE]
//                  [--prefill-chunk C (0: the longest that fits VRAM)] [--prefill-chunk-max C] [--chunk-min N]
//
// One sequence at a time: generate requests queue in arrival order. A reader thread takes the
// input lines, so a "stop" for the running request cancels it between decode steps.
//
// A request that fails gets an error event and leaves the engine serving (Session::generate
// resets itself). After a failure the process cannot recover from, the error event says so and
// the engine exits with status 3; the server sees its output end.
//
// FLASHRT_FAULT_INJECTION=1 honours a request's "debug_fail" (1: after the first prefill step,
// 2: in the first decode step), for bench/engine_smoke.py --faults.
#include "core/json.hpp"
#include "engine/session.hpp"

#include <cuda_runtime.h>

#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>

using namespace flashrt;

namespace {

std::mutex out_mu;
void emit(const Json& j) {
    const std::string line = j.dump();
    std::lock_guard<std::mutex> lk(out_mu);
    std::fwrite(line.data(), 1, line.size(), stdout);
    std::fputc('\n', stdout);
    std::fflush(stdout);
}
Json event(const char* ev, const std::string& id) {
    Json j = Json::object();
    j.set("ev", ev);
    if (!id.empty()) j.set("id", id);
    return j;
}
void error(const std::string& id, const std::string& msg) { emit(event("error", id).set("msg", msg)); }

struct Queue {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<Json> ops;
    bool quit = false;
    std::string running;              // id of the request being generated
    std::atomic<bool> cancel{false};
};

// A whole number in lo..hi; absent: the fallback.
int64_t whole(const Json& v, const char* what, int64_t lo, int64_t hi, int64_t fallback) {
    if (v.is_null()) return fallback;
    const double d = v.num();
    if (!v.is_number() || d != std::floor(d) || d < double(lo) || d > double(hi))
        throw std::runtime_error(std::string(what) + " must be a whole number in " + std::to_string(lo) + ".." + std::to_string(hi));
    return int64_t(d);
}
// A finite number; absent: the fallback. Session::generate checks the ranges.
float real(const Json& v, const char* what, double fallback) {
    if (v.is_null()) return float(fallback);
    if (!v.is_number() || !std::isfinite(v.num())) throw std::runtime_error(std::string(what) + " must be a number");
    return float(v.num());
}
const std::vector<Json>& id_list(const Json& v, const char* what) {
    if (!v.is_null() && !v.is_array()) throw std::runtime_error(std::string(what) + " must be an array of token ids");
    return v.items();
}

GenerateRequest to_request(const Json& op) {
    constexpr int64_t kIdMax = 0x7fffffff, kSeedMax = int64_t(1) << 53;   // seeds: exact in a double
    GenerateRequest r;
    if (!op["prompt"].is_array()) throw std::runtime_error("prompt must be an array of token ids");
    for (const Json& t : op["prompt"].items()) {
        if (!t.is_number()) throw std::runtime_error("prompt must be an array of token ids");
        r.prompt.push_back(int32_t(whole(t, "a prompt token id", 0, kIdMax, 0)));
    }
    r.max_new = int(whole(op["max_new"], "max_new", 1, kIdMax, 256));
    const Json& sp = op["sampling"];
    if (!sp.is_null() && !sp.is_object()) throw std::runtime_error("sampling must be an object");
    r.sampling.temperature = real(sp["temperature"], "temperature", 0.0);
    r.sampling.top_k = int(whole(sp["top_k"], "top_k", 1, sample::kMaxTopK, 20));
    r.sampling.top_p = real(sp["top_p"], "top_p", 1.0);
    r.sampling.min_p = real(sp["min_p"], "min_p", 0.0);
    r.seed = uint64_t(whole(sp["seed"], "seed", 0, kSeedMax, 0));
    for (const Json& t : id_list(op["stop_ids"], "stop_ids")) {
        if (!t.is_number()) throw std::runtime_error("stop_ids must be an array of token ids");
        r.stop_ids.push_back(int32_t(whole(t, "a stop id", 0, kIdMax, 0)));
    }
    static const bool faults = [] {
        const char* e = std::getenv("FLASHRT_FAULT_INJECTION");
        return e && e[0] == '1';
    }();
    if (faults) r.fail_at = int(whole(op["debug_fail"], "debug_fail", 0, 2, 0));
    return r;
}

}  // namespace

int main(int argc, char** argv) {
    std::ios::sync_with_stdio(false);
    if (argc < 2) {
        std::fprintf(stderr, "usage: flashrt-engine MODEL.gguf [--mtp DRAFT.gguf [--spec K] [--draft-vocab RANKS]] [--ctx N] ...\n");
        return 2;
    }
    SessionOptions o;
    o.model = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--mtp") o.mtp = next();
        else if (a == "--spec") o.spec_k = std::atoi(next());
        else if (a == "--draft-vocab") o.draft_vocab = next();
        else if (a == "--draft-vocab-n") o.draft_vocab_n = std::atoi(next());
        else if (a == "--mtp-bits") o.mtp_bits = std::atoi(next());
        else if (a == "--ctx") o.max_ctx = std::atoi(next());
        else if (a == "--kv-hot") o.kv_hot = std::atoi(next());
        else if (a == "--kv") o.kv_q8 = std::string(next()) == "q8";
        else if (a == "--workers") o.workers = std::atoi(next());
        else if (a == "--reserve-mib") o.reserve_mib = std::atoi(next());
        else if (a == "--swap-budget") o.swap_budget = std::atoi(next());
        else if (a == "--cache-prior") o.cache_prior = next();
        else if (a == "--prefill-chunk") o.prefill_chunk = std::atoi(next());
        else if (a == "--prefill-chunk-max") o.prefill_chunk_max = std::atoi(next());
        else if (a == "--chunk-min") o.chunk_min = std::atoi(next());
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    std::unique_ptr<Session> session;
    try {
        session = std::make_unique<Session>(o);
    } catch (const std::exception& e) {
        error("", std::string("load failed: ") + e.what());
        return 1;
    }
    Json features = Json::array();
    features.push("stop").push("sampling").push("prefix_reuse");
    if (session->speculative()) features.push("mtp");
    emit(event("ready", "").set("version", "0.1.0").set("arch", session->arch()).set("max_context", session->max_context())
             .set("features", features));

    Queue q;
    std::thread reader([&] {
        std::string line;
        while (std::getline(std::cin, line)) {
            if (line.empty()) continue;
            Json op;
            try {
                op = Json::parse(line);
            } catch (const std::exception& e) {
                error("", e.what());
                continue;
            }
            const std::string kind = op["op"].str(), id = op["id"].str();
            std::lock_guard<std::mutex> lk(q.mu);
            if (kind == "quit") break;
            if (kind == "stop") {
                if (id == q.running) q.cancel.store(true);
                for (auto it = q.ops.begin(); it != q.ops.end(); ++it)   // a queued request just goes away
                    if ((*it)["id"].str() == id) {
                        emit(event("done", id).set("generated", 0).set("finish", "cancelled"));
                        q.ops.erase(it);
                        break;
                    }
                continue;
            }
            if (kind == "generate") {
                q.ops.push_back(std::move(op));
                q.cv.notify_one();
                continue;
            }
            error(id, "unknown op");
        }
        std::lock_guard<std::mutex> lk(q.mu);
        q.quit = true;
        q.cancel.store(true);
        q.cv.notify_one();
    });

    for (;;) {
        Json op;
        {
            std::unique_lock<std::mutex> lk(q.mu);
            q.cv.wait(lk, [&] { return q.quit || !q.ops.empty(); });
            if (q.ops.empty()) break;   // quit
            op = std::move(q.ops.front());
            q.ops.pop_front();
            q.running = op["id"].str();
            q.cancel.store(false);
        }
        const std::string id = op["id"].str();
        try {
            const GenerateRequest r = to_request(op);
            const GenerateResult res = session->generate(
                r, [&](int32_t t) { emit(event("token", id).set("tok", int(t))); },
                [&](int done, int total) {
                    emit(event("progress", id).set("prompt_done", done).set("prompt_total", total));
                },
                q.cancel);
            Json drafts = Json::object();
            drafts.set("proposed", int64_t(res.drafts_proposed)).set("accepted", int64_t(res.drafts_accepted));
            emit(event("done", id)
                     .set("generated", res.generated)
                     .set("prompt_tokens", res.prompt_tokens)
                     .set("reused", res.reused)
                     .set("prompt_ms", res.prompt_ms)
                     .set("decode_ms", res.decode_ms)
                     .set("finish", res.finish)
                     .set("drafts", drafts));
        } catch (const std::exception& e) {
            if (!session->healthy()) {
                error(id, std::string(e.what()) + " (fatal: the engine exits)");
                std::_Exit(3);   // destructors could wait on a stuck miss server or a broken device
            }
            error(id, e.what());
            (void)cudaGetLastError();   // a failed request must not leave its error to the next one's checks (sw86)
        }
        std::lock_guard<std::mutex> lk(q.mu);
        q.running.clear();
    }
    reader.detach();   // it may be blocked in getline; the process is exiting
    return 0;
}
