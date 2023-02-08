#include "kernel_launcher/internal/directives.h"

#include <unordered_map>
#include <utility>
#include <vector>

#include "kernel_launcher/builder.h"
#include "kernel_launcher/expr.h"
#include "kernel_launcher/internal/parser.h"

namespace kernel_launcher {
namespace internal {

struct Context {
    std::unordered_map<std::string, ArgExpr> runtime_args;
    std::unordered_map<std::string, Expr> compile_args;
    std::unordered_map<std::string, ParamExpr> config_args;
};

static Expr parse_expr(TokenStream& stream, const Context& ctx, int prec = 0);

static Expr process_function_call(
    Token t,
    const TokenStream& stream,
    std::vector<Expr> args) {
    std::string name = stream.span(t);
    auto assert_nargs = [&](size_t n) {
        if (n != args.size()) {
            stream.throw_unexpected_token(
                t,
                "function expects " + std::to_string(n) + "arguments but "
                    + std::to_string(args.size()) + " arguments were given");
        }
    };

    if (name == "round") {
        assert_nargs(1);
        return round(args[0]);
    } else if (name == "ceil") {
        assert_nargs(1);
        return ceil(args[0]);
    } else if (name == "floor") {
        assert_nargs(1);
        return floor(args[0]);
    } else if (name == "div_ceil") {
        assert_nargs(2);
        return div_ceil(args[0], args[1]);
    } else if (name == "float") {
        assert_nargs(1);
        return cast<double>(args[0]);
    } else {
        stream.throw_unexpected_token(t, "unknown function name");
    }
}

static Expr parse_ident(Token t, TokenStream& stream, const Context& ctx) {
    if (stream.next_if(TokenKind::ParenL)) {
        std::vector<Expr> args;

        do {
            args.push_back(parse_expr(stream, ctx));
        } while (stream.next_if(TokenKind::Comma));

        stream.consume(TokenKind::ParenR);

        return process_function_call(t, stream, args);
    }

    std::string name = stream.span(t);

    if (name == "null") {
        return ScalarExpr(Value {});
    }

    if (name == "true") {
        return ScalarExpr(true);
    }

    if (name == "false") {
        return ScalarExpr(false);
    }

    // Is it a config parameter?
    {
        auto it = ctx.config_args.find(name);
        if (it != ctx.config_args.end()) {
            return it->second;
        }
    }

    // Is it a compile-time parameter?
    {
        auto it = ctx.compile_args.find(name);
        if (it != ctx.compile_args.end()) {
            return it->second;
        }
    }

    // Is it a runtime parameter?
    {
        auto it = ctx.runtime_args.find(name);
        if (it != ctx.runtime_args.end()) {
            return it->second;
        }
    }

    stream.throw_unexpected_token(t, "unknown variable name");
}

static bool parse_string(const std::string& input, std::string& output) {
    size_t n = input.size();
    if (n < 2 || input[0] != input[n - 1]) {
        return false;
    }

    bool prev_backslash = false;

    for (size_t i = 1; i < n - 1; i++) {
        char c = input[i];
        if (prev_backslash) {
            char x;

            switch (c) {
                case 'n':
                    x = '\n';
                    break;
                case 't':
                    x = '\t';
                case 'r':
                    x = '\r';
                    break;
                case '"':
                case '\'':
                case '\\':
                    x = c;
                    break;
                default:
                    return false;
            }

            prev_backslash = false;
            output += x;
        } else if (c == '\\') {
            prev_backslash = true;
        } else {
            output += c;
        }
    }

    return !prev_backslash;
}

static bool parse_long(const std::string& input, long& output) {
    char* endptr = nullptr;
    output = strtol(input.c_str(), &endptr, 10);
    return *endptr == '\0';
}

static Expr parse_prim(TokenStream& stream, const Context& ctx) {
    Token t = stream.next();

    if (t.kind == TokenKind::Ident) {
        return parse_ident(t, stream, ctx);
    } else if (t.kind == TokenKind::ParenL) {
        Expr e = parse_expr(stream, ctx);
        stream.consume(TokenKind::ParenR);
        return e;
    } else if (t.kind == TokenKind::String) {
        std::string out;
        if (!parse_string(stream.span(t), out)) {
            stream.throw_unexpected_token(t, "failed to parse string");
        }
        return ScalarExpr(out);
    } else if (t.kind == TokenKind::Number) {
        long l;
        if (!parse_long(stream.span(t), l)) {
            stream.throw_unexpected_token(t, "failed to parse as integer");
        }
        return ScalarExpr(l);
    } else if (stream.matches(t, '-')) {
        return -parse_expr(stream, ctx);
    } else if (stream.matches(t, '+')) {
        return +parse_expr(stream, ctx);
    } else if (stream.matches(t, '!')) {
        return !parse_expr(stream, ctx);
    } else {
        stream.throw_unexpected_token(t, "expecting expression");
    }
}

static Expr parse_expr(TokenStream& stream, const Context& ctx, int prec) {
    // TODO: == != <= >= && || %
    Expr lhs = parse_prim(stream, ctx);

    while (true) {
        if (prec < 6 && stream.next_if('*')) {
            lhs = lhs * parse_expr(stream, ctx, 6);
        } else if (prec < 6 && stream.next_if('/')) {
            lhs = lhs / parse_expr(stream, ctx, 6);
        } else if (prec < 5 && stream.next_if('+')) {
            lhs = lhs + parse_expr(stream, ctx, 5);
        } else if (prec < 5 && stream.next_if('-')) {
            lhs = lhs - parse_expr(stream, ctx, 5);
        } else if (prec < 3 && stream.next_if('<')) {
            lhs = lhs < parse_expr(stream, ctx, 3);
        } else if (prec < 3 && stream.next_if('>')) {
            lhs = lhs > parse_expr(stream, ctx, 3);
        } else if (prec < 3 && stream.next_if("<=")) {
            lhs = lhs <= parse_expr(stream, ctx, 3);
        } else if (prec < 3 && stream.next_if(">=")) {
            lhs = lhs >= parse_expr(stream, ctx, 3);
        } else if (prec < 3 && stream.next_if("!=")) {
            lhs = lhs != parse_expr(stream, ctx, 3);
        } else if (prec < 3 && stream.next_if("==")) {
            lhs = lhs == parse_expr(stream, ctx, 3);
        } else {
            return lhs;
        }
    }
}

static std::vector<Expr> parse_expr_list(
    TokenStream& stream,
    const Context& ctx,
    size_t max_params = 1024) {
    std::vector<Expr> output;

    stream.consume(TokenKind::ParenL);

    // Empty list
    if (stream.next_if(TokenKind::ParenR)) {
        return output;
    }

    while (output.size() < max_params) {
        output.push_back(parse_expr(stream, ctx));

        if (!stream.next_if(TokenKind::Comma)) {
            break;
        }
    }

    stream.consume(TokenKind::ParenR);
    return output;
}

static std::array<Expr, 3>
parse_expr_list3(TokenStream& stream, const Context& ctx) {
    auto list = parse_expr_list(stream, ctx, 3);
    return {
        list.size() > 0 ? list[0] : 1,  // NOLINT
        list.size() > 1 ? list[1] : 1,
        list.size() > 2 ? list[2] : 1,
    };
}

struct DummyEval: Eval {
    bool lookup(const Variable& v, Value& out) const override {
        throw std::runtime_error("internal error");
    }
};

static void parse_buffer_directive(
    TokenStream& stream,
    KernelBuilder& builder,
    Context& ctx) {
    stream.consume(TokenKind::ParenL);
    do {
        Token var_token = stream.consume(TokenKind::Ident);
        stream.consume(TokenKind::BracketL);
        Expr length = parse_expr(stream, ctx);
        stream.consume(TokenKind::BracketR);

        auto it = ctx.runtime_args.find(stream.span(var_token));
        if (it == ctx.runtime_args.end()) {
            stream.throw_unexpected_token(
                var_token,
                "this is not the name of a kernel argument");
        }

        builder.buffer_size(it->second, length);
    } while (stream.next_if(TokenKind::Comma));
    stream.consume(TokenKind::ParenR);
}

static void parse_tune_directive(
    TokenStream& stream,
    KernelBuilder& builder,
    Context& ctx) {
    std::vector<Value> values;
    std::vector<double> priors;

    stream.consume(TokenKind::ParenL);
    Token var_token = stream.consume(TokenKind::Ident);
    stream.consume('=');

    do {
        Value v = parse_expr(stream, {}).eval(DummyEval {});
        values.push_back(v);
        priors.push_back(1.0);
    } while (stream.next_if(TokenKind::Comma));

    stream.consume(TokenKind::ParenR);

    std::string var = stream.span(var_token);

    if (ctx.config_args.count(var) > 0) {
        stream.throw_unexpected_token(var_token, "variable redefined");
    }

    if (ctx.compile_args.count(var) > 0) {
        stream.throw_unexpected_token(
            var_token,
            "variable already passed as compile-time value");
    }

    auto param = builder.add(var, values, priors, values.front());
    ctx.config_args.insert({var, param});
}

static void parse_set_directive(TokenStream& stream, Context& ctx) {
    stream.consume(TokenKind::ParenL);
    Token var_token = stream.consume(TokenKind::Ident);
    stream.consume('=');
    Value value = parse_expr(stream, ctx).eval(DummyEval {});
    stream.consume(TokenKind::ParenR);

    std::string var = stream.span(var_token);
    ctx.compile_args.insert({var, value});
}

static void
process_directive(TokenStream& stream, KernelBuilder& builder, Context& ctx) {
    stream.consume("pragma");
    stream.consume("kernel_tuner");

    while (!stream.next_if(TokenKind::DirectiveEnd)) {
        Token t = stream.consume(TokenKind::Ident);
        std::string name = stream.span(t);

        if (name == "tune") {
            parse_tune_directive(stream, builder, ctx);
        } else if (name == "set") {
            parse_set_directive(stream, ctx);
        } else if (name == "buffers" || name == "buffer") {
            parse_buffer_directive(stream, builder, ctx);
        } else if (name == "tuning_key") {
            std::string key = "";

            for (const auto& expr : parse_expr_list(stream, ctx)) {
                key += expr.eval(DummyEval {}).to_string();
            }

            builder.tuning_key(std::move(key));

        } else if (name == "grid_size") {
            auto l = parse_expr_list3(stream, ctx);
            builder.grid_size(l[0], l[1], l[2]);
        } else if (name == "block_size") {
            auto l = parse_expr_list3(stream, ctx);
            builder.block_size(l[0], l[1], l[2]);
        } else if (name == "grid_divisor") {
            auto l = parse_expr_list3(stream, ctx);
            builder.grid_divisors(l[0], l[1], l[2]);
        } else if (name == "problem_size") {
            auto l = parse_expr_list3(stream, ctx);
            builder.problem_size(l[0], l[1], l[2]);
        } else if (name == "restriction") {
            for (const auto& expr : parse_expr_list(stream, ctx)) {
                builder.restriction(expr);
            }
        } else {
            stream.throw_unexpected_token(
                t,
                "this is not a supported action in kernel_launcher");
        }
    }
}

KernelBuilder builder_from_annotated_kernel(
    TokenStream& stream,
    KernelSource source,
    const AnnotatedKernelSpec& def,
    const std::vector<Value>& template_args) {
    auto builder = KernelBuilder(def.qualified_name, std::move(source));

    Context ctx;

    for (size_t i = 0; i < def.fun_params.size(); i++) {
        std::string name = stream.span(def.fun_params[i].name);
        ctx.runtime_args.insert({name, ArgExpr(uint8_t(i))});
    }

    if (template_args.size() > def.template_params.size()) {
        throw std::runtime_error(
            "cannot provide " + std::to_string(template_args.size())
            + " arguments to kernel " + def.qualified_name
            + " since it takes at most "
            + std::to_string(def.template_params.size()) + " arguments");
    }

    for (size_t i = 0; i < template_args.size(); i++) {
        std::string name = stream.span(def.template_params[i].name);
        ctx.compile_args.insert({name, template_args[i]});
    }

    for (const auto& directive : def.directives) {
        stream.seek(directive);
        process_directive(stream, builder, ctx);
    }

    for (const auto& param : def.template_params) {
        std::string name = stream.span(param.name);
        Expr e = nullptr;

        if (ctx.compile_args.count(name) > 0) {
            e = ctx.compile_args.at(name);
        } else if (ctx.config_args.count(name) > 0) {
            e = ctx.config_args.at(name);
        } else {
            stream.throw_unexpected_token(
                param.name,
                "this template parameter is not defined, please add "
                "`#pragma kernel_tuner tune("
                    + name + "=...)`");
        }

        builder.template_arg(e);
    }

    return builder;
}

}  // namespace internal
}  // namespace kernel_launcher