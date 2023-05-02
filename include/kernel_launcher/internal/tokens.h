#ifndef KERNEL_LAUNCHER_TOKENIZER_H
#define KERNEL_LAUNCHER_TOKENIZER_H

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace kernel_launcher {
namespace internal {

enum class TokenKind {
    String,
    Number,
    ParenL,
    ParenR,
    BracketL,
    BracketR,
    BraceL,
    BraceR,
    AngleL,
    AngleR,
    DirectiveBegin,
    DirectiveEnd,
    Ident,
    Punct,
    Comma,
    Unknown,
    EndOfFile,
};

struct Token {
    uint16_t begin = 0;
    uint16_t end = 0;
    TokenKind kind = TokenKind::Unknown;

    Token() = default;
    Token(uint16_t begin, uint16_t end, TokenKind kind) :
        begin(begin),
        end(end),
        kind(kind) {}

    bool operator==(const Token& that) const {
        return begin == that.begin && end == that.end && kind == that.kind;
    }

    bool operator!=(const Token& that) const {
        return !(*this == that);
    }
};

struct TokenStream {
    explicit TokenStream(const TokenStream&) = default;
    TokenStream(TokenStream&&) = default;

    TokenStream(std::string file, std::string input);
    void seek(Token t);
    bool has_next() const;
    Token next();
    Token peek();
    void prev();

    bool matches(Token t, char c) const;
    bool matches(Token t, const char* needle) const;

    bool matches(Token t, const std::string& s) const {
        return matches(t, s.c_str());
    }

    bool matches(Token t, TokenKind kind) const {
        return t.kind == kind;
    }

    template<typename T, size_t N>
    bool matches(Token t, const std::array<T, N>& options) const {
        for (const auto& option : options) {
            if (matches(t, option)) {
                return true;
            }
        }

        return false;
    }

    template<typename T>
    bool next_if(T&& pattern) {
        if (!matches(peek(), std::forward<T>(pattern))) {
            return false;
        }

        next();
        return true;
    }

    [[noreturn]] void throw_expecting_token(Token t, TokenKind k) const;
    [[noreturn]] void throw_expecting_token(Token t, const char* c) const;

    [[noreturn]] void
    throw_expecting_token(Token t, const std::string& s) const {
        throw_expecting_token(t, s.c_str());
    }

    [[noreturn]] void throw_expecting_token(Token t, char c) const {
        char str[2] = {c, '\0'};
        throw_expecting_token(t, str);
    }

    template<typename T>
    Token consume(const T& pattern) {
        Token t = next();
        if (!matches(t, pattern)) {
            throw_expecting_token(t, pattern);
        }

        return t;
    }

    std::string span(size_t begin, size_t end) const;

    std::string span(Token t) const {
        return span(t.begin, t.end);
    }

    std::string span(Token begin, Token end) const {
        return span(begin.begin, end.end);
    }

    [[noreturn]] void throw_unexpected_token(
        size_t begin,
        size_t end,
        const std::string& reason = "") const;

    [[noreturn]] void
    throw_unexpected_token(Token t, const std::string& reason = "") const {
        throw_unexpected_token(t.begin, t.end, reason);
    }

  private:
    std::string file_;
    std::string text_;
    size_t index_ = 0;
    std::vector<Token> tokens_;
};

}  // namespace internal
}  // namespace kernel_launcher

#endif
