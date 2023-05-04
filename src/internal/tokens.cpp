#include "kernel_launcher/internal/tokens.h"

#include <algorithm>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace kernel_launcher {
namespace internal {

using index_t = uint16_t;

static bool iswhitespace(char c) {
    return c == ' ' || c == '\n' || c == '\t' || c == '\r' || c == '\v'
        || c == '\f';
}

static bool isdigit(char c) {
    return c >= '0' && c <= '9';
}

static bool isident(char c) {
    return isdigit(c) || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')
        || c == '_' || c == '$';
}

static index_t
advance_single_line_comment(index_t i, const std::string& input) {
    while (i < input.size() && input[i] != '\n') {
        i++;
    }

    return i;
}

static index_t advance_multi_line_comment(index_t i, const std::string& input) {
    char prev = '\0';

    while (i < input.size()) {
        char curr = input[i++];

        if (prev == '*' && curr == '/') {
            break;
        }

        prev = curr;
    }

    return i;
}

static index_t advance_number(index_t i, const std::string& input) {
    while (isdigit(input[i])) {
        i++;
    }

    return i;
}

static index_t advance_ident(index_t i, const std::string& input) {
    while (isident(input[i])) {
        i++;
    }

    return i;
}

static index_t advance_string(index_t i, const std::string& input) {
    char quote = input[i];
    bool prev_backslash = false;
    i++;

    while (i < input.size()) {
        if (input[i] == quote && !prev_backslash) {
            i++;
            return i;
        }

        prev_backslash = input[i] == '\\' && !prev_backslash;
        i++;
    }

    return i;
}

TokenKind char2_to_kind(char a, char b) {
    if ((a == '=' && b == '=') || (a == '!' && b == '=')
        || (a == '<' && b == '=') || (a == '>' && b == '=')
        || (a == '&' && b == '&') || (a == '|' && b == '|')
        || (a == '<' && b == '<') || (a == '>' && b == '>')
        || (a == ':' && b == ':')) {
        return TokenKind::Punct;
    }

    return TokenKind::Unknown;
}

TokenKind char_to_kind(char c) {
    switch (c) {
        case '{':
            return TokenKind::BraceL;
        case '}':
            return TokenKind::BraceR;
        case '[':
            return TokenKind::BracketL;
        case ']':
            return TokenKind::BracketR;
        case '(':
            return TokenKind::ParenL;
        case ')':
            return TokenKind::ParenR;
        case ',':
            return TokenKind::Comma;
        case '<':
            return TokenKind::AngleL;
        case '>':
            return TokenKind::AngleR;
        case '+':
        case '=':
        case '-':
        case '*':
        case '/':
        case '!':
        case '~':
        case '&':
        case '|':
        case '^':
        case '%':
            return TokenKind::Punct;
        default:
            // Remaining: .:;?@
            return TokenKind::Unknown;
    }
}

std::vector<Token> tokenize(const std::string& input) {
    std::vector<Token> tokens;

    if (input.size() >= std::numeric_limits<index_t>::max()) {
        throw std::runtime_error("TODO");
    }

    index_t index = 0;
    bool inside_directive = false;

    while (index < input.size()) {
        index_t begin = index;
        char c = input[index];
        char next = input[index + 1];
        TokenKind kind = TokenKind::Unknown;

        if (!inside_directive && c == '#') {
            kind = TokenKind::DirectiveBegin;
            index++;
            inside_directive = true;
        } else if (inside_directive && c == '\n') {
            kind = TokenKind::DirectiveEnd;
            index++;
            inside_directive = false;
        } else if (inside_directive && c == '\\' && next == '\n') {
            index++;  // skip backslash
            index++;  // skip newline
            continue;
        } else if (iswhitespace(c)) {
            index++;
            continue;
        } else if (c == '/' && next == '/') {
            index = advance_single_line_comment(index, input);
            continue;
        } else if (c == '/' && next == '*') {
            index = advance_multi_line_comment(index, input);
            continue;
        } else if (isdigit(c)) {
            index = advance_number(index, input);
            kind = TokenKind::Number;
        } else if (isident(c)) {
            index = advance_ident(index, input);
            kind = TokenKind::Ident;
        } else if (c == '"' || c == '\'') {
            index = advance_string(index, input);
            kind = TokenKind::String;
        } else if ((kind = char2_to_kind(c, next)) != TokenKind::Unknown) {
            index++;
            index++;
        } else if ((kind = char_to_kind(c)) != TokenKind::Unknown) {
            index++;
        } else {
            // Unknown character :(
            kind = TokenKind::Unknown;
            index++;
        }

        tokens.emplace_back(begin, index, kind);
    }

    if (inside_directive) {
        tokens.emplace_back(index, index, TokenKind::DirectiveEnd);
    }

    tokens.emplace_back(index, index, TokenKind::EndOfFile);
    return tokens;
}

TokenStream::TokenStream(std::string file, std::string input) :
    file_(std::move(file)),
    text_(std::move(input)),
    index_(0) {
    tokens_ = tokenize(text_);
}

void TokenStream::seek(Token t) {
    auto it = std::lower_bound(
        tokens_.begin(),
        tokens_.end(),
        t,
        [&](const auto& lhs, const auto& rhs) {
            return lhs.begin < rhs.begin;
        });

    if (it == tokens_.end() || *it != t) {
        throw std::runtime_error("cannot reset to unknown token");
    }

    index_ = static_cast<size_t>(it - tokens_.begin());
}

bool TokenStream::has_next() const {
    return index_ < tokens_.size();
}

Token TokenStream::next() {
    Token t = peek();
    index_++;
    return t;
}

Token TokenStream::peek() {
    if (index_ >= tokens_.size()) {
        throw std::runtime_error("unexpected EOF while parsing");
    }

    return tokens_[index_];
}

void TokenStream::prev() {
    if (index_ > 0) {
        index_--;
    }
}

bool TokenStream::matches(Token t, char c) const {
    if (t.begin + 1 != t.end || t.begin >= text_.size()) {
        return false;
    }

    return text_[t.begin] == c;
}

bool TokenStream::matches(Token t, const char* needle) const {
    if (t.end > text_.size()) {
        return false;
    }

    for (index_t i = t.begin; i < t.end; i++) {
        if (*needle != text_[i] || *needle == '\0') {
            return false;
        }

        needle++;
    }

    return *needle == '\0';
}

static std::string clean_string(const std::string& input) {
    std::stringstream output;

    for (char c : input) {
        if (isprint(c) != 0) {
            output << c;
        } else if (c == '\n') {
            output << "\\n";
        } else if (c == '\0') {
            output << "\\0";
        } else {
            output << "?";
        }
    }

    return output.str();
}

static std::string token_description(TokenKind k) {
    switch (k) {
        case TokenKind::String:
            return "string";
        case TokenKind::Number:
            return "integer";
        case TokenKind::ParenL:
            return "'('";
        case TokenKind::ParenR:
            return "')'";
        case TokenKind::BracketL:
            return "'['";
        case TokenKind::BracketR:
            return "']'";
        case TokenKind::BraceL:
            return "'{'";
        case TokenKind::BraceR:
            return "'}'";
        case TokenKind::AngleL:
            return "'<'";
        case TokenKind::AngleR:
            return "'>'";
        case TokenKind::DirectiveBegin:
            return "'#'";
        case TokenKind::DirectiveEnd:
            return "<directive end>";
        case TokenKind::Ident:
            return "identifier";
        case TokenKind::Comma:
            return "','";
        case TokenKind::EndOfFile:
            return "<end of file>";
        default:
            return "unknown token";
    }
}

[[noreturn]] void
TokenStream::throw_expecting_token(Token t, TokenKind k) const {
    throw_unexpected_token(t, "expecting token " + token_description(k));
}

[[noreturn]] void
TokenStream::throw_expecting_token(Token t, const char* c) const {
    std::string reason = "expecting token \"" + clean_string(c) + "\"";
    throw_unexpected_token(t, reason);
}

static std::string
underlined_span(size_t begin, size_t end, const std::string& text) {
    size_t begin_line = begin;
    while (begin_line > 0 && text[begin_line - 1] != '\n') {
        begin_line--;
    }

    size_t end_line = begin_line;
    while (end_line < text.size() && text[end_line] != '\n') {
        end_line++;
    }

    // In these cases, there is nothing to underline (empty token? empty line?)
    if (begin >= end || begin_line >= end_line || begin >= end_line
        || begin_line >= end) {
        return "";
    }

    std::stringstream msg;
    for (size_t i = begin_line; i < end_line; i++) {
        msg << text[i];
    }

    msg << "\n";
    for (size_t i = begin_line; i < end_line; i++) {
        if (i >= begin && i < end) {
            msg << '^';
        } else {
            msg << ' ';
        }
    }

    return msg.str();
}

static std::pair<int, int>
extract_line_column(size_t offset, const std::string& text) {
    int lineno = 1;
    int colno = 1;

    for (size_t i = 0; i < text.size() && i < offset; i++) {
        if (text[i] == '\n') {
            lineno++;
            colno = 1;
        } else {
            colno++;
        }
    }

    return {lineno, colno};
}

void TokenStream::throw_unexpected_token(
    size_t begin,
    size_t end,
    const std::string& reason) const {
    auto line_col = extract_line_column(begin, text_);

    std::stringstream msg;
    msg << "error:" << file_ << ":" << line_col.first << ":" << line_col.second
        << ": found invalid token \"" << clean_string(span(begin, end)) << "\"";

    if (!reason.empty()) {
        msg << ", " << reason;
    }

    std::string snippet = underlined_span(begin, end, text_);
    if (!snippet.empty()) {
        msg << "\n" << snippet;
    }

    throw std::runtime_error(msg.str());
}

std::string TokenStream::span(size_t begin, size_t end) const {
    if (begin > end || end > text_.size()) {
        throw std::runtime_error("index out of bounds");
    }

    return text_.substr(begin, end - begin);
}

}  // namespace internal
}  // namespace kernel_launcher