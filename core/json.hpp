// SPDX-License-Identifier: Apache-2.0
// A small JSON value for the engine protocol (docs/design.md): parse one line, read fields,
// build replies. Numbers are doubles (token ids and counts fit exactly); strings are UTF-8 with
// the standard escapes (\uXXXX, surrogate pairs included).
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace flashrt {

class Json {
public:
    enum class Type { Null, Bool, Number, String, Array, Object };

    Json() = default;
    Json(std::nullptr_t) {}
    Json(bool b) : type_(Type::Bool), b_(b) {}
    Json(double d) : type_(Type::Number), d_(d) {}
    Json(int v) : type_(Type::Number), d_(v) {}
    Json(int64_t v) : type_(Type::Number), d_(double(v)) {}
    Json(const char* s) : type_(Type::String), s_(s) {}
    Json(std::string s) : type_(Type::String), s_(std::move(s)) {}
    static Json array() { Json j; j.type_ = Type::Array; return j; }
    static Json object() { Json j; j.type_ = Type::Object; return j; }

    // Throws std::runtime_error on malformed input.
    static Json parse(const std::string& text);
    std::string dump() const;

    Type type() const { return type_; }
    bool is_null() const { return type_ == Type::Null; }
    bool is_number() const { return type_ == Type::Number; }
    bool is_string() const { return type_ == Type::String; }
    bool is_array() const { return type_ == Type::Array; }
    bool is_object() const { return type_ == Type::Object; }

    // Typed reads with a fallback when the value is absent or of another type.
    double num(double fallback = 0) const { return type_ == Type::Number ? d_ : fallback; }
    bool boolean(bool fallback = false) const { return type_ == Type::Bool ? b_ : fallback; }
    const std::string& str() const { return s_; }
    const std::vector<Json>& items() const { return a_; }

    // Object access: a missing key reads as null.
    const Json& operator[](const std::string& key) const;
    bool has(const std::string& key) const { return o_.count(key) != 0; }
    Json& set(const std::string& key, Json v) {   // objects keep insertion order in dump()
        if (!o_.count(key)) keys_.push_back(key);
        o_[key] = std::move(v);
        return *this;
    }
    Json& push(Json v) {
        a_.push_back(std::move(v));
        return *this;
    }

private:
    Type type_ = Type::Null;
    bool b_ = false;
    double d_ = 0;
    std::string s_;
    std::vector<Json> a_;
    std::map<std::string, Json> o_;
    std::vector<std::string> keys_;
    friend class JsonParser;
};

}  // namespace flashrt
