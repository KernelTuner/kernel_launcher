#include "kernel_launcher/internal/parser.h"

namespace kernel_launcher {
namespace internal {

static bool parse_long(const std::string& s, long& output) {
    char* end_ptr = nullptr;
    output = strtol(s.c_str(), &end_ptr, 10);
    return *end_ptr == '\0';
}

static std::vector<TemplateParam> parse_template_params(TokenStream& stream) {
    std::vector<TemplateParam> params;

    do {
        std::string ty = stream.span(stream.consume(TokenKind::Ident));
        Token name = stream.consume(TokenKind::Ident);
        Value default_value;

        // integral parameters do not start with "typename" or "class"
        bool is_integral = !(ty == "typename" || ty == "class");

        if (is_integral && stream.next_if('=')) {
            // We only support numbers for now...
            Token v = stream.consume(TokenKind::Number);
            long l;

            if (!parse_long(stream.span(v), l)) {
                stream.throw_unexpected_token(v, "invalid integer");
            }

            default_value = l;
        }

        params.push_back(TemplateParam {name, is_integral, ty, default_value});
    } while (stream.next_if(TokenKind::Comma));

    return params;
}

static std::vector<FunctionParam> parse_kernel_params(TokenStream& stream) {
    std::vector<FunctionParam> params;

    do {
        Token begin = stream.next();
        Token before_name = begin;
        Token name = stream.next();
        Token end = stream.peek();

        while (end.kind != TokenKind::Comma && end.kind != TokenKind::ParenR) {
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

static bool extract_kernel_tuner_directives(
    TokenStream& stream,
    std::vector<Token>& directives_out) {
    static constexpr const std::array<const char*, 3> PRAGMA_NAMES = {
        "kernel",
        "kernel_launcher",
        "kernel_tuner"};

    // Check if directive starts with correct pragma. If not, this is
    // not a relevant pragma, and we do not need to parse it.
    Token t = stream.peek();
    if (!stream.next_if("pragma")) {
        stream.seek(t);
        return false;
    }

    std::string pragma_name = stream.span(stream.consume(TokenKind::Ident));
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

    // Parse all pragmas
    do {
        stream.consume("pragma");
        stream.consume(pragma_name);
        t = stream.next();
        directives_out.push_back(t);

        while (t.kind != TokenKind::DirectiveEnd) {
            t = stream.next();
        }
    } while (stream.next_if(TokenKind::DirectiveBegin));

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
        qualified_name,
        name,
        directives,
        template_params,
        fun_params,
    };
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
            cur = stream.consume(TokenKind::Ident);
            namespace_stack.push_back(stream.span(cur));

            stream.consume(TokenKind::BraceL);
            scope_stack.push_back(Scope::Namespace);
        } else if (cur.kind == TokenKind::BraceL) {
            scope_stack.push_back(Scope::Brace);
        } else if (cur.kind == TokenKind::BraceR) {
            if (!scope_stack.empty()
                && scope_stack.back() == Scope::Namespace) {
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
                source.append("/*");
                source.append(stream.span(before_dir.begin, after_dir.begin));
                source.append("*/");

                kernels.push_back(parse_kernel(
                    stream,
                    namespace_stack,
                    std::move(directives)));

                last = after_dir;
            } else {
                while (cur.kind != TokenKind::DirectiveEnd) {
                    cur = stream.next();
                }
            }
        }
    }

    source.append(stream.span(last, cur));
    return AnnotatedDocument {kernels, source};
}

}  // namespace internal
}  // namespace kernel_launcher
