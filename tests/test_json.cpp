// SPDX-License-Identifier: Apache-2.0
// core/json against the engine protocol's needs (docs/design.md): round trips, escapes and
// UTF-8, numbers exact up to 2^53, JSON's number grammar (strtod alone would take nan, inf and
// hex), surrogates that are not a pair, and malformed input rejected with an exception.
#include "core/json.hpp"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

using namespace flashrt;

namespace {
int failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::printf("FAIL %s\n", what.c_str());
        ++failures;
    }
}

bool rejects(const std::string& text) {
    try {
        Json::parse(text);
    } catch (const std::runtime_error&) {
        return true;
    }
    return false;
}
}  // namespace

int main() {
    // a request line as the server sends it
    const Json r = Json::parse(R"({"op":"generate","id":"r1","prompt":[1,2,248045],"max_new":256,
        "sampling":{"temperature":0.6,"top_k":20,"top_p":0.95},"seed":9007199254740991,"stop_ids":[]})");
    expect(r["op"].str() == "generate" && r["id"].str() == "r1", "strings");
    expect(r["prompt"].items().size() == 3 && r["prompt"].items()[2].num() == 248045, "array of ids");
    expect(r["sampling"]["top_p"].num() == 0.95 && r["sampling"]["temperature"].num() == 0.6, "nested numbers");
    expect(r["seed"].num() == 9007199254740991.0, "2^53 - 1 exact");
    expect(r["missing"].is_null() && r["missing"]["deeper"].is_null(), "a missing key reads as null");
    expect(r["stop_ids"].is_array() && r["stop_ids"].items().empty(), "empty array");
    expect(r["op"].num(7) == 7 && r["seed"].boolean(true), "typed reads fall back on another type");

    // dump: insertion order, integers without a fraction, escapes
    Json d = Json::object();
    d.set("ev", "done").set("n", 384).set("ms", 12.5).set("big", int64_t(1) << 53).set("s", "a\"b\\c\nd\x01");
    d.set("ev", "token");   // an existing key keeps its place
    expect(d.dump() == R"({"ev":"token","n":384,"ms":12.5,"big":9007199254740992,"s":"a\"b\\c\nd\u0001"})", "dump: " + d.dump());
    const Json back = Json::parse(d.dump());
    expect(back["s"].str() == "a\"b\\c\nd\x01" && back["big"].num() == 9007199254740992.0, "round trip");

    // UTF-8 passes through; \u escapes decode, surrogate pairs included
    expect(Json::parse("\"h\xc3\xa9 \xe4\xb8\xad \xf0\x9f\x98\x80\"").str() == "h\xc3\xa9 \xe4\xb8\xad \xf0\x9f\x98\x80", "raw UTF-8");
    expect(Json::parse(R"("\u00e9\u4e2d\ud83d\ude00\/")").str() == "\xc3\xa9\xe4\xb8\xad\xf0\x9f\x98\x80/", "escapes and a pair");
    // a surrogate that is not half of a pair is U+FFFD, and what follows it is kept
    const std::string fffd = "\xef\xbf\xbd";
    expect(Json::parse(R"("\ud800\u0041")").str() == fffd + "A", "high surrogate before a non-surrogate");
    expect(Json::parse(R"("\ud800x")").str() == fffd + "x", "lone high surrogate");
    expect(Json::parse(R"("\ude00")").str() == fffd, "lone low surrogate");
    expect(Json::parse(R"("\ud800\ud83d\ude00")").str() == fffd + "\xf0\x9f\x98\x80", "high surrogate before a pair");

    // numbers: JSON's grammar only
    expect(Json::parse("-0").num() == 0 && Json::parse("1e3").num() == 1000 && Json::parse("-2.5E-1").num() == -0.25, "valid numbers");
    for (const char* bad : {"nan", "NaN", "inf", "-inf", "Infinity", "+1", "0x10", ".5", "1.", "-", "1e", "1e+", "01", "--1"})
        expect(rejects(bad), std::string("rejects the number ") + bad);
    expect(rejects(R"({"prompt":[1,nan]})"), "rejects nan inside a request");

    // malformed input
    for (const char* bad : {"", " ", "{", "[1,2", "{\"a\" 1}", "{\"a\":1,}", "[1,]", "\"abc", "\"\\x\"", "\"\\u12\"", "tru", "{} x",
                            "{1:2}"})
        expect(rejects(bad), std::string("rejects ") + bad);
    std::string deep(66, '[');   // the outermost value is depth 0
    deep += std::string(66, ']');
    expect(rejects(deep), "rejects nesting deeper than 64");
    std::string ok(65, '[');
    ok += std::string(65, ']');
    expect(!rejects(ok), "accepts nesting to depth 64");

    // non-integers dump in the shortest form that reads back exactly; no NaN in JSON
    for (double v : {0.1, 12.5, 61234.567, 1.0 / 3.0, -2.5e-300, 1e300})
        expect(Json::parse(Json(v).dump()).num() == v, "exact round trip of " + Json(v).dump());
    expect(Json(0.1).dump() == "0.1" && Json(std::nan("")).dump() == "null", "short forms: " + Json(0.1).dump());

    std::printf("%s: %d failure(s)\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
