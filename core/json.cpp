// SPDX-License-Identifier: Apache-2.0
#include "core/json.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace flashrt {

class JsonParser {
public:
    explicit JsonParser(const std::string& t) : t_(t) {}

    Json parse_all() {
        Json v = value(0);
        ws();
        if (i_ != t_.size()) fail("trailing characters");
        return v;
    }

private:
    [[noreturn]] void fail(const char* what) const {
        throw std::runtime_error(std::string("json: ") + what + " at offset " + std::to_string(i_));
    }
    void ws() {
        while (i_ < t_.size() && (t_[i_] == ' ' || t_[i_] == '\t' || t_[i_] == '\n' || t_[i_] == '\r')) ++i_;
    }
    bool lit(const char* s) {
        size_t n = 0;
        while (s[n]) ++n;
        if (t_.compare(i_, n, s) != 0) return false;
        i_ += n;
        return true;
    }
    Json value(int depth) {
        if (depth > 64) fail("nesting too deep");
        ws();
        if (i_ >= t_.size()) fail("unexpected end");
        const char c = t_[i_];
        if (c == '{') return object(depth);
        if (c == '[') return array(depth);
        if (c == '"') return Json(string());
        if (lit("true")) return Json(true);
        if (lit("false")) return Json(false);
        if (lit("null")) return Json();
        return number();
    }
    Json object(int depth) {
        Json o = Json::object();
        ++i_;
        ws();
        if (i_ < t_.size() && t_[i_] == '}') { ++i_; return o; }
        for (;;) {
            ws();
            if (i_ >= t_.size() || t_[i_] != '"') fail("expected a key");
            std::string k = string();
            ws();
            if (i_ >= t_.size() || t_[i_] != ':') fail("expected ':'");
            ++i_;
            o.set(k, value(depth + 1));
            ws();
            if (i_ < t_.size() && t_[i_] == ',') { ++i_; continue; }
            if (i_ < t_.size() && t_[i_] == '}') { ++i_; return o; }
            fail("expected ',' or '}'");
        }
    }
    Json array(int depth) {
        Json a = Json::array();
        ++i_;
        ws();
        if (i_ < t_.size() && t_[i_] == ']') { ++i_; return a; }
        for (;;) {
            a.push(value(depth + 1));
            ws();
            if (i_ < t_.size() && t_[i_] == ',') { ++i_; continue; }
            if (i_ < t_.size() && t_[i_] == ']') { ++i_; return a; }
            fail("expected ',' or ']'");
        }
    }
    Json number() {
        const char* b = t_.c_str() + i_;
        char* e = nullptr;
        const double d = std::strtod(b, &e);
        if (e == b) fail("bad value");
        i_ += size_t(e - b);
        return Json(d);
    }
    unsigned hex4() {
        if (i_ + 4 > t_.size()) fail("bad \\u escape");
        unsigned v = 0;
        for (int k = 0; k < 4; ++k) {
            const char h = t_[i_++];
            v <<= 4;
            if (h >= '0' && h <= '9') v |= unsigned(h - '0');
            else if (h >= 'a' && h <= 'f') v |= unsigned(h - 'a' + 10);
            else if (h >= 'A' && h <= 'F') v |= unsigned(h - 'A' + 10);
            else fail("bad \\u escape");
        }
        return v;
    }
    static void utf8(std::string& out, unsigned cp) {
        if (cp < 0x80) out += char(cp);
        else if (cp < 0x800) { out += char(0xC0 | (cp >> 6)); out += char(0x80 | (cp & 0x3F)); }
        else if (cp < 0x10000) { out += char(0xE0 | (cp >> 12)); out += char(0x80 | ((cp >> 6) & 0x3F)); out += char(0x80 | (cp & 0x3F)); }
        else {
            out += char(0xF0 | (cp >> 18));
            out += char(0x80 | ((cp >> 12) & 0x3F));
            out += char(0x80 | ((cp >> 6) & 0x3F));
            out += char(0x80 | (cp & 0x3F));
        }
    }
    std::string string() {
        ++i_;   // opening quote
        std::string out;
        for (;;) {
            if (i_ >= t_.size()) fail("unterminated string");
            const char c = t_[i_++];
            if (c == '"') return out;
            if (c != '\\') { out += c; continue; }
            if (i_ >= t_.size()) fail("unterminated escape");
            const char e = t_[i_++];
            switch (e) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    unsigned cp = hex4();
                    if (cp >= 0xD800 && cp < 0xDC00 && i_ + 6 <= t_.size() && t_[i_] == '\\' && t_[i_ + 1] == 'u') {
                        i_ += 2;
                        const unsigned lo = hex4();
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    }
                    utf8(out, cp);
                    break;
                }
                default: fail("bad escape");
            }
        }
    }

    const std::string& t_;
    size_t i_ = 0;
};

Json Json::parse(const std::string& text) { return JsonParser(text).parse_all(); }

const Json& Json::operator[](const std::string& key) const {
    static const Json null;
    auto it = o_.find(key);
    return it == o_.end() ? null : it->second;
}

namespace {
void dump_string(const std::string& s, std::string& out) {
    out += '"';
    for (const unsigned char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += char(c);
                }
        }
    }
    out += '"';
}
}  // namespace

std::string Json::dump() const {
    std::string out;
    switch (type_) {
        case Type::Null: return "null";
        case Type::Bool: return b_ ? "true" : "false";
        case Type::Number: {
            if (std::isfinite(d_) && d_ == std::floor(d_) && std::fabs(d_) < 9.007199254740992e15) return std::to_string(int64_t(d_));
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%.6g", std::isfinite(d_) ? d_ : 0.0);
            return buf;
        }
        case Type::String: dump_string(s_, out); return out;
        case Type::Array:
            out += '[';
            for (size_t k = 0; k < a_.size(); ++k) {
                if (k) out += ',';
                out += a_[k].dump();
            }
            out += ']';
            return out;
        case Type::Object:
            out += '{';
            for (size_t k = 0; k < keys_.size(); ++k) {
                if (k) out += ',';
                dump_string(keys_[k], out);
                out += ':';
                out += o_.at(keys_[k]).dump();
            }
            out += '}';
            return out;
    }
    return out;
}

}  // namespace flashrt
