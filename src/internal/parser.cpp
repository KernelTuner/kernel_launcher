#include "kernel_launcher/internal/parser.hpp"

#include <array>
#include <cerrno>
#include <cstdlib>

namespace kernel_launcher {
namespace internal {

static bool parse_long(const std::string& s, long& output) {
    char* end_ptr = nullptr;
    errno = 0;
    output = strtol(s.c_str(), &end_ptr, 10);
    return *end_ptr == '\0' && errno == 0;
}

// Skips tokens until a ',' or '>' is found that is not nested inside '<...>'.
static void skip_template_default(TokenStream& stream) {
    int depth = 0;

    while (true) {
        Token t = stream.peek();

        if (depth == 0 && (t.kind == TokenKind::Comma || t.kind == TokenKind::AngleR)) {
            return;
        }

        if (t.kind == TokenKind::AngleL) {
            depth++;
        } else if (t.kind == TokenKind::AngleR) {
            depth--;
        } else if (t.kind == TokenKind::EndOfFile) {
            stream.throw_unexpected_token(t, "unexpected end of file");
        }

        stream.next();
    }
}

// Parses the default value of an integral template parameter. Only literals
// (`true`, `false`, and possibly negative integers) are supported; any other
// default value is skipped and an empty `Primitive` is returned.
static Primitive parse_template_default(TokenStream& stream) {
    Token begin = stream.peek();
    Primitive result;

    if (stream.next_if("true")) {
        result = true;
    } else if (stream.next_if("false")) {
        result = false;
    } else {
        bool negative = stream.next_if('-');
        Token v = stream.peek();
        long l;

        if (stream.next_if(TokenKind::Number) && parse_long(stream.span(v), l)) {
            result = negative ? -l : l;
        }
    }

    Token end = stream.peek();
    if (result.is_empty() || !(end.kind == TokenKind::Comma || end.kind == TokenKind::AngleR)) {
        stream.seek(begin);
        skip_template_default(stream);
        return {};
    }

    return result;
}

static std::vector<TemplateParam> parse_template_params(TokenStream& stream) {
    std::vector<TemplateParam> params;

    do {
        // The type can consist of multiple identifiers (e.g.,
        // `unsigned int N`), the last identifier is the name.
        Token ty_begin = stream.consume(TokenKind::Ident);
        Token ty_end = ty_begin;
        Token name = stream.consume(TokenKind::Ident);

        while (stream.matches(stream.peek(), TokenKind::Ident)) {
            ty_end = name;
            name = stream.next();
        }

        std::string ty = stream.span(ty_begin, ty_end);
        Primitive default_value;

        // integral parameters do not start with "typename" or "class"
        bool is_integral = !(ty == "typename" || ty == "class");

        if (stream.next_if('=')) {
            if (is_integral) {
                default_value = parse_template_default(stream);
            } else {
                skip_template_default(stream);
            }
        }

        params.push_back(TemplateParam {name, is_integral, ty, default_value});
    } while (stream.next_if(TokenKind::Comma));

    return params;
}

static std::vector<FunctionParam> parse_kernel_params(TokenStream& stream) {
    std::vector<FunctionParam> params;

    // Empty parameter list: `()` or `(void)`
    if (stream.matches(stream.peek(), TokenKind::ParenR)) {
        return params;
    }

    Token t = stream.peek();
    if (stream.next_if("void")) {
        if (stream.matches(stream.peek(), TokenKind::ParenR)) {
            return params;
        }

        stream.seek(t);
    }

    do {
        Token begin = stream.next();
        Token before_name = begin;
        Token name = stream.next();
        Token end = stream.peek();
        int template_depth = 0;

        while (template_depth > 0
               || !(end.kind == TokenKind::Comma || end.kind == TokenKind::ParenR)) {
            if (name.kind == TokenKind::AngleL) {
                template_depth++;
            } else if (name.kind == TokenKind::AngleR && template_depth > 0) {
                template_depth--;
            }

            before_name = name;
            name = stream.next();
            end = stream.peek();
        }

        if (name.kind != TokenKind::Ident) {
            stream.throw_expecting_token(name, TokenKind::Ident);
        }

        params.push_back({
            stream.span(begin, before_name),
            name,
        });
    } while (stream.next_if(TokenKind::Comma));

    return params;
}

// Checks if the next directive is `#pragma <pragma_name>`. If so, the stream
// is positioned after the `#` and `true` is returned. Otherwise, the stream
// is not moved and `false` is returned.
static bool next_directive_matches(TokenStream& stream, std::string_view pragma_name) {
    Token begin = stream.peek();

    if (!stream.next_if(TokenKind::DirectiveBegin)) {
        return false;
    }

    Token after_hash = stream.peek();
    bool matches = stream.next_if("pragma") && stream.next_if(pragma_name);
    stream.seek(matches ? after_hash : begin);
    return matches;
}

static bool
extract_kernel_tuner_directives(TokenStream& stream, std::vector<Token>& directives_out) {
    static constexpr const std::array<const char*, 3> PRAGMA_NAMES = {
        "kernel",
        "kernel_launcher",
        "kernel_tuner"};

    // Check if directive starts with correct pragma. If not, this is
    // not a relevant pragma, and we do not need to parse it.
    Token t = stream.peek();
    if (!stream.next_if("pragma") || !stream.matches(stream.peek(), TokenKind::Ident)) {
        stream.seek(t);
        return false;
    }

    std::string pragma_name = stream.span(stream.next());
    stream.seek(t);

    // Check if `pragma_name` is one of `PRAGMA_NAMES`
    bool is_valid_name = false;
    for (const auto& valid_name : PRAGMA_NAMES) {
        if (pragma_name == valid_name) {
            is_valid_name = true;
        }
    }

    if (!is_valid_name) {
        return false;
    }

    // Parse all consecutive pragmas with the same name
    do {
        stream.consume("pragma");
        stream.consume(pragma_name);
        t = stream.next();
        directives_out.push_back(t);

        while (t.kind != TokenKind::DirectiveEnd) {
            t = stream.next();
        }
    } while (next_directive_matches(stream, pragma_name));

    return true;
}

static AnnotatedKernelSpec parse_kernel(
    TokenStream& stream,
    const std::vector<std::string>& namespaces,
    std::vector<Token> directives) {
    std::vector<TemplateParam> template_params;

    // check for 'template' '<' ... '>'
    if (stream.next_if("template")) {
        stream.consume(TokenKind::AngleL);
        template_params = parse_template_params(stream);
        stream.consume(TokenKind::AngleR);
    }

    // check for '__global__' 'void' IDENT
    stream.consume("__global__");
    stream.consume("void");
    Token name = stream.consume(TokenKind::Ident);

    // check for '(' ... ')'
    stream.consume(TokenKind::ParenL);
    auto fun_params = parse_kernel_params(stream);
    stream.consume(TokenKind::ParenR);

    std::string qualified_name;
    for (const auto& n : namespaces) {
        qualified_name += n;
        qualified_name += "::";
    }
    qualified_name += stream.span(name);

    return {
        std::move(qualified_name),
        name,
        std::move(directives),
        std::move(template_params),
        std::move(fun_params),
    };
}

// Returns the given directive text with every line turned into a `//`
// comment. This keeps the line numbers intact and, unlike `/* */`, cannot be
// terminated early by the content of the directive. Trailing whitespace (the
// indentation of the line that follows the directives) is not commented out.
static std::string comment_out(std::string_view text) {
    size_t content_end = text.find_last_not_of(" \t\r\n\v\f");
    std::string result = "//";

    for (size_t i = 0; i < text.size(); i++) {
        result += text[i];

        if (text[i] == '\n' && i < content_end) {
            result += "//";
        }
    }

    return result;
}

enum struct Scope {
    Paren,
    Bracket,
    Brace,
    Namespace,
};

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
AnnotatedDocument extract_annotated_kernels(TokenStream& stream) {
    std::vector<std::string> namespace_stack;
    std::vector<Scope> scope_stack;
    std::vector<AnnotatedKernelSpec> kernels;
    std::vector<Token> directives;
    std::string source;

    auto assert_pop_scope = [&](Token t, Scope scope, const char* msg) {
        if (scope_stack.empty() || scope_stack.back() != scope) {
            stream.throw_unexpected_token(t, msg);
        }

        scope_stack.pop_back();
    };

    Token last {};
    Token cur;

    while (stream.has_next()) {
        cur = stream.next();

        if (cur.kind == TokenKind::Ident && stream.matches(cur, "namespace")) {
            // Handles `namespace a {`, `namespace a::b {`, and `namespace {`.
            // Other uses, such as `using namespace a;` or `namespace x = a;`,
            // do not open a scope and are skipped.
            std::string name;

            while (stream.matches(stream.peek(), TokenKind::Ident)) {
                name += stream.span(stream.next());

                if (!stream.next_if("::")) {
                    break;
                }

                name += "::";
            }

            if (stream.next_if(TokenKind::BraceL)) {
                if (name.empty()) {
                    scope_stack.push_back(Scope::Brace);
                } else {
                    namespace_stack.push_back(name);
                    scope_stack.push_back(Scope::Namespace);
                }
            }
        } else if (cur.kind == TokenKind::BraceL) {
            scope_stack.push_back(Scope::Brace);
        } else if (cur.kind == TokenKind::BraceR) {
            if (!scope_stack.empty() && scope_stack.back() == Scope::Namespace) {
                namespace_stack.pop_back();
                scope_stack.back() = Scope::Brace;
            }

            assert_pop_scope(cur, Scope::Brace, "no matching '{' found");
        } else if (cur.kind == TokenKind::ParenL) {
            scope_stack.push_back(Scope::Paren);
        } else if (cur.kind == TokenKind::ParenR) {
            assert_pop_scope(cur, Scope::Paren, "no matching '(' found");
        } else if (cur.kind == TokenKind::BracketL) {
            scope_stack.push_back(Scope::Bracket);
        } else if (cur.kind == TokenKind::BracketR) {
            assert_pop_scope(cur, Scope::Bracket, "no matching '[' found");
        } else if (cur.kind == TokenKind::DirectiveBegin) {
            if (extract_kernel_tuner_directives(stream, directives)) {
                Token before_dir = cur;
                Token after_dir = stream.peek();

                source.append(stream.span(last.begin, before_dir.begin));
                source.append(comment_out(stream.span(before_dir.begin, after_dir.begin)));

                kernels.push_back(parse_kernel(stream, namespace_stack, std::move(directives)));
                directives.clear();

                last = after_dir;
            } else {
                while (cur.kind != TokenKind::DirectiveEnd) {
                    cur = stream.next();
                }
            }
        }
    }

    source.append(stream.span(last, cur));
    return AnnotatedDocument {std::move(kernels), std::move(source)};
}

}  // namespace internal
}  // namespace kernel_launcher
