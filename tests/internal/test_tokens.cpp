#include <utility>

#include "catch.hpp"
#include "kernel_launcher/internal/tokens.hpp"

using namespace kernel_launcher::internal;

// Returns (kind, text) for every token, including the final `EndOfFile`.
static std::vector<std::pair<TokenKind, std::string>> tokenize_all(const std::string& input) {
    TokenStream stream("test.cu", input);
    std::vector<std::pair<TokenKind, std::string>> result;

    while (stream.has_next()) {
        Token t = stream.next();
        result.emplace_back(t.kind, stream.span(t));
    }

    return result;
}

TEST_CASE("TokenStream tokenize") {
    using K = TokenKind;

    SECTION("basic tokens") {
        auto tokens = tokenize_all("int x = 42; foo(\"a\\\"b\", 'c') [ ] { } < >");

        std::vector<std::pair<TokenKind, std::string>> expected = {
            {K::Ident, "int"},
            {K::Ident, "x"},
            {K::Punct, "="},
            {K::Number, "42"},
            {K::Unknown, ";"},
            {K::Ident, "foo"},
            {K::ParenL, "("},
            {K::String, "\"a\\\"b\""},
            {K::Comma, ","},
            {K::String, "'c'"},
            {K::ParenR, ")"},
            {K::BracketL, "["},
            {K::BracketR, "]"},
            {K::BraceL, "{"},
            {K::BraceR, "}"},
            {K::AngleL, "<"},
            {K::AngleR, ">"},
            {K::EndOfFile, ""},
        };

        CHECK(tokens == expected);
    }

    SECTION("operators") {
        auto tokens = tokenize_all("<= >= == != && || :: + - * / % ! ~ ^ >>");

        std::vector<std::pair<TokenKind, std::string>> expected = {
            {K::Punct, "<="},
            {K::Punct, ">="},
            {K::Punct, "=="},
            {K::Punct, "!="},
            {K::Punct, "&&"},
            {K::Punct, "||"},
            {K::Punct, "::"},
            {K::Punct, "+"},
            {K::Punct, "-"},
            {K::Punct, "*"},
            {K::Punct, "/"},
            {K::Punct, "%"},
            {K::Punct, "!"},
            {K::Punct, "~"},
            {K::Punct, "^"},
            // `>>` is two tokens, so it can close nested templates.
            {K::AngleR, ">"},
            {K::AngleR, ">"},
            {K::EndOfFile, ""},
        };

        CHECK(tokens == expected);
    }

    SECTION("comments are skipped") {
        auto tokens = tokenize_all("a // comment\nb /* multi\nline */ c");

        std::vector<std::pair<TokenKind, std::string>> expected = {
            {K::Ident, "a"},
            {K::Ident, "b"},
            {K::Ident, "c"},
            {K::EndOfFile, ""},
        };

        CHECK(tokens == expected);
    }

    SECTION("directives") {
        auto tokens = tokenize_all("#define A \\\n 1\nx\n#pragma y");

        std::vector<std::pair<TokenKind, std::string>> expected = {
            {K::DirectiveBegin, "#"},
            {K::Ident, "define"},
            {K::Ident, "A"},
            {K::Number, "1"},
            {K::DirectiveEnd, "\n"},
            {K::Ident, "x"},
            {K::DirectiveBegin, "#"},
            {K::Ident, "pragma"},
            {K::Ident, "y"},
            // A directive at the end of the input is closed by an empty token.
            {K::DirectiveEnd, ""},
            {K::EndOfFile, ""},
        };

        CHECK(tokens == expected);
    }

    SECTION("empty input") {
        auto tokens = tokenize_all("");

        std::vector<std::pair<TokenKind, std::string>> expected = {{K::EndOfFile, ""}};
        CHECK(tokens == expected);
    }
}

TEST_CASE("TokenStream navigation") {
    TokenStream stream("test.cu", "a b c");

    Token a = stream.peek();
    CHECK(stream.next() == a);
    Token b = stream.next();
    CHECK(stream.span(b) == "b");

    stream.prev();
    CHECK(stream.next() == b);

    Token c = stream.consume(TokenKind::Ident);
    CHECK(stream.span(c) == "c");
    CHECK(stream.span(a, c) == "a b c");

    Token eof = stream.next();
    CHECK(eof.kind == TokenKind::EndOfFile);
    CHECK_FALSE(stream.has_next());
    CHECK_THROWS_WITH(stream.peek(), Catch::Contains("unexpected EOF"));

    stream.seek(b);
    CHECK(stream.next() == b);

    stream.seek(eof);
    CHECK(stream.next() == eof);

    CHECK_THROWS(stream.seek(Token(1, 2, TokenKind::Ident)));

    // Copies have their own position.
    stream.seek(a);
    TokenStream copy(stream);
    copy.next();
    CHECK(stream.peek() == a);
    CHECK(copy.peek() == b);

    stream.prev();
    stream.prev();
    CHECK(stream.peek() == a);
}

TEST_CASE("TokenStream matching") {
    TokenStream stream("test.cu", "foo ( <= bar");
    Token foo = stream.next();
    Token paren = stream.next();
    Token lte = stream.next();

    CHECK(stream.matches(foo, "foo"));
    CHECK_FALSE(stream.matches(foo, "fo"));
    CHECK_FALSE(stream.matches(foo, "food"));
    CHECK(stream.matches(foo, TokenKind::Ident));
    CHECK_FALSE(stream.matches(foo, TokenKind::Number));
    CHECK(stream.matches(paren, '('));
    CHECK_FALSE(stream.matches(foo, 'f'));
    CHECK(stream.matches(lte, "<="));
    CHECK_FALSE(stream.matches(lte, '<'));

    std::array<const char*, 2> options = {"bar", "baz"};
    CHECK(stream.matches(stream.peek(), options));
    CHECK_FALSE(stream.matches(foo, options));

    CHECK_FALSE(stream.next_if("baz"));
    CHECK(stream.next_if("bar"));
    CHECK(stream.next_if(TokenKind::EndOfFile));
}

TEST_CASE("TokenStream errors") {
    TokenStream stream("test.cu", "first line\nint x = 42;");

    SECTION("consume") {
        stream.next();
        stream.next();
        stream.next();

        // `x` is at line 2, column 5.
        CHECK_THROWS_WITH(
            stream.consume(TokenKind::Number),
            "test.cu:2:5: error occurred at token \"x\", expecting token integer\n"
            "int x = 42;\n"
            "    ^      ");
    }

    SECTION("consume string") {
        CHECK_THROWS_WITH(
            stream.consume("second"),
            Catch::StartsWith(
                "test.cu:1:1: error occurred at token \"first\", expecting token \"second\""));
    }

    SECTION("consume char") {
        CHECK_THROWS_WITH(stream.consume('('), Catch::Contains("expecting token \"(\""));
    }

    SECTION("custom reason") {
        Token first = stream.next();
        Token line = stream.next();
        CHECK_THROWS_WITH(
            stream.throw_unexpected_token(line, "bad"),
            Catch::StartsWith("test.cu:1:7: error occurred at token \"line\", bad"));

        CHECK_THROWS_WITH(
            stream.throw_unexpected_token(first.begin, line.end),
            "test.cu:1:1: error occurred at token \"first line\"\n"
            "first line\n"
            "^^^^^^^^^^");
    }

    SECTION("span out of bounds") {
        CHECK_THROWS(stream.span(5, 100));
        CHECK_THROWS(stream.span(5, 4));
    }
}
