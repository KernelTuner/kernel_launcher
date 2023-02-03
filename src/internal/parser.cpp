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

        bool is_integral = ty != "typename" && ty != "class";

        if (is_integral && stream.next_if('=')) {
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

static KernelDef
parse_kernel(TokenStream& stream, const std::vector<std::string>& namespaces) {
    std::vector<Token> directives;
    std::vector<TemplateParam> template_params;

    // Advance the stream past all directives
    while (stream.next_if(TokenKind::DirectiveBegin)) {
        Token t = stream.next();
        directives.push_back(t);

        // Find the directive end
        while (t.kind != TokenKind::DirectiveEnd) {
            t = stream.next();
        }
    }

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
std::vector<KernelDef> parse_kernels(TokenStream& stream) {
    std::vector<std::string> namespace_stack;
    std::vector<Scope> scope_stack;
    std::vector<KernelDef> kernels;

    auto assert_pop_scope = [&](Token t, Scope scope, const char* msg) {
        if (scope_stack.empty() || scope_stack.back() != scope) {
            stream.throw_unexpected_token(t, msg);
        }

        scope_stack.pop_back();
    };

    while (stream.has_next()) {
        Token t = stream.next();

        if (t.kind == TokenKind::Ident && stream.matches(t, "namespace")) {
            t = stream.consume(TokenKind::Ident);
            namespace_stack.push_back(stream.span(t));

            stream.consume(TokenKind::BraceL);
            scope_stack.push_back(Scope::Namespace);
        } else if (t.kind == TokenKind::BraceL) {
            scope_stack.push_back(Scope::Brace);
        } else if (t.kind == TokenKind::BraceR) {
            if (!scope_stack.empty()
                && scope_stack.back() == Scope::Namespace) {
                namespace_stack.pop_back();
                scope_stack.back() = Scope::Brace;
            }

            assert_pop_scope(t, Scope::Brace, "no matching '{' found");
        } else if (t.kind == TokenKind::ParenL) {
            scope_stack.push_back(Scope::Paren);
        } else if (t.kind == TokenKind::ParenR) {
            assert_pop_scope(t, Scope::Paren, "no matching '(' found");
        } else if (t.kind == TokenKind::BracketL) {
            scope_stack.push_back(Scope::Bracket);
        } else if (t.kind == TokenKind::BracketR) {
            assert_pop_scope(t, Scope::Bracket, "no matching '[' found");
        } else if (t.kind == TokenKind::DirectiveBegin) {
            bool is_pragma =
                stream.next_if("pragma") && stream.next_if("kernel_tuner");
            stream.seek(t);

            if (is_pragma) {
                kernels.push_back(parse_kernel(stream, namespace_stack));
            } else {
                while (t.kind != TokenKind::DirectiveEnd) {
                    t = stream.next();
                }
            }
        }
    }

    return kernels;
}

}  // namespace internal
}  // namespace kernel_launcher