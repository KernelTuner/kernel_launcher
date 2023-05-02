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
    // Runtime arguments of the kernel
    std::unordered_map<std::string, ArgExpr> kernel_args;

    // Parameters from configuration space
    std::unordered_map<std::string, TunableParam> config_args;

    // Compile-time arguments passed by the user
    std::unordered_map<std::string, Value> comptime_args;

    // User-defined parameters from `#pragma kernel set(foo=1+2)`
    std::unordered_map<std::string, Expr> user_args;
};

static Expr parse_expr(TokenStream& stream, const Context& ctx);

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

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
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
        auto it = ctx.comptime_args.find(name);
        if (it != ctx.comptime_args.end()) {
            return it->second;
        }
    }

    // Is it a user-defined parameter?
    {
        auto it = ctx.user_args.find(name);
        if (it != ctx.user_args.end()) {
            return it->second;
        }
    }

    // Is it a runtime parameter?
    {
        auto it = ctx.kernel_args.find(name);
        if (it != ctx.kernel_args.end()) {
            return it->second;
        }
    }

#define CHECK_DEVICE_ATTRIBUTE(attr) \
    if (name == "DEVICE_" #attr) {   \
        return DEVICE_##attr;        \
    }

    KERNEL_LAUNCHER_DEVICE_ATTRIBUTES_FORALL(CHECK_DEVICE_ATTRIBUTE)

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
                    break;
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
        return -parse_prim(stream, ctx);
    } else if (stream.matches(t, '+')) {
        return +parse_prim(stream, ctx);
    } else if (stream.matches(t, '!')) {
        return !parse_prim(stream, ctx);
    } else {
        stream.throw_unexpected_token(t, "expecting expression");
    }
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
static Expr parse_binop(TokenStream& stream, const Context& ctx, int prec) {
    // TODO: == != <= >= && || %
    Expr lhs = parse_prim(stream, ctx);

    while (true) {
        if (prec < 6 && stream.next_if('*')) {
            lhs = lhs * parse_binop(stream, ctx, 6);
        } else if (prec < 6 && stream.next_if('/')) {
            lhs = lhs / parse_binop(stream, ctx, 6);
        } else if (prec < 5 && stream.next_if('+')) {
            lhs = lhs + parse_binop(stream, ctx, 5);
        } else if (prec < 5 && stream.next_if('-')) {
            lhs = lhs - parse_binop(stream, ctx, 5);
        } else if (prec < 3 && stream.next_if('<')) {
            lhs = lhs < parse_binop(stream, ctx, 3);
        } else if (prec < 3 && stream.next_if('>')) {
            lhs = lhs > parse_binop(stream, ctx, 3);
        } else if (prec < 3 && stream.next_if("<=")) {
            lhs = lhs <= parse_binop(stream, ctx, 3);
        } else if (prec < 3 && stream.next_if(">=")) {
            lhs = lhs >= parse_binop(stream, ctx, 3);
        } else if (prec < 3 && stream.next_if("!=")) {
            lhs = lhs != parse_binop(stream, ctx, 3);
        } else if (prec < 3 && stream.next_if("==")) {
            lhs = lhs == parse_binop(stream, ctx, 3);
        } else if (prec < 2 && stream.next_if("&&")) {
            lhs = lhs && parse_binop(stream, ctx, 2);
        } else if (prec < 1 && stream.next_if("||")) {
            lhs = lhs || parse_binop(stream, ctx, 1);
        } else {
            return lhs;
        }
    }
}

static Expr parse_ternary(TokenStream& stream, const Context& ctx) {
    Expr cond = parse_binop(stream, ctx, 0);

    if (stream.next_if('?')) {
        Expr if_true = parse_expr(stream, ctx);
        stream.consume(':');
        Expr if_false = parse_expr(stream, ctx);
        return ifelse(std::move(cond), std::move(if_true), std::move(if_false));
    } else {
        return cond;
    }
}

static Expr parse_expr(TokenStream& stream, const Context& ctx) {
    return parse_ternary(stream, ctx);
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
        return false;
    }
};

static Value parse_comptime_expr(TokenStream& stream, const Context& ctx) {
    Token before = stream.peek();
    Expr e = parse_expr(stream, ctx);
    Token after = stream.peek();

    try {
        // We can use `DummyEval` to evaluate the expression since compile-time
        // expression should not contain any variables.
        return e.eval(DummyEval {});
    } catch (const std::exception& err) {
        auto msg =
            std::string("error while evaluating expression: ") + err.what();
        stream.throw_unexpected_token(before.begin, after.begin, msg);
    }
}

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

        auto it = ctx.kernel_args.find(stream.span(var_token));
        if (it == ctx.kernel_args.end()) {
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
        Value v = parse_comptime_expr(stream, ctx);
        values.push_back(v);
        priors.push_back(1.0);
    } while (stream.next_if(TokenKind::Comma));

    stream.consume(TokenKind::ParenR);

    Value default_value = values[0];
    if (stream.next_if("default")) {
        stream.consume(TokenKind::ParenL);
        default_value = parse_comptime_expr(stream, ctx);
        stream.consume(TokenKind::ParenR);
    }

    std::string var = stream.span(var_token);

    if (ctx.config_args.count(var) > 0) {
        stream.throw_unexpected_token(var_token, "variable redefined");
    }

    auto param = builder.add(var, values, priors, default_value);
    ctx.config_args.insert({var, param});
}

static void parse_set_directive(TokenStream& stream, Context& ctx) {
    // '(' IDENT '=' EXPR ')'
    stream.consume(TokenKind::ParenL);
    do {
        Token var_token = stream.consume(TokenKind::Ident);
        std::string var = stream.span(var_token);
        if (ctx.comptime_args.count(var) > 0) {
            stream.throw_unexpected_token(var_token, "variable redefined");
        }

        stream.consume('=');

        Expr expr = parse_expr(stream, ctx);
        ctx.user_args.insert({var, expr});
    } while (stream.next_if(TokenKind::Comma));

    stream.consume(TokenKind::ParenR);
}

static void parse_tuning_key_directive(
    TokenStream& stream,
    KernelBuilder& builder,
    Context& ctx) {
    std::string key;

    stream.consume(TokenKind::ParenL);
    do {
        key += parse_comptime_expr(stream, ctx).to_string();
    } while (stream.next_if(TokenKind::Comma));
    stream.consume(TokenKind::ParenR);

    builder.tuning_key(std::move(key));
}

static void
process_directive(TokenStream& stream, KernelBuilder& builder, Context& ctx) {
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
            parse_tuning_key_directive(stream, builder, ctx);
        } else if (name == "grid_size" || name == "grid_dim") {
            auto l = parse_expr_list3(stream, ctx);
            builder.grid_size(l[0], l[1], l[2]);
        } else if (name == "block_size" || name == "block_dim") {
            auto l = parse_expr_list3(stream, ctx);
            builder.block_size(l[0], l[1], l[2]);
        } else if (name == "grid_divisor" || name == "grid_divisors") {
            auto l = parse_expr_list3(stream, ctx);
            builder.grid_divisors(l[0], l[1], l[2]);
        } else if (name == "problem_size" || name == "problem_dim") {
            auto l = parse_expr_list3(stream, ctx);
            builder.problem_size(l[0], l[1], l[2]);
        } else if (name == "restriction" || name == "restrictions") {
            for (const auto& expr : parse_expr_list(stream, ctx)) {
                builder.restriction(expr);
            }
        } else {
            stream.throw_unexpected_token(
                t,
                "this directive is not supported by kernel_launcher");
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
        ctx.kernel_args.insert({name, ArgExpr(uint8_t(i), name)});
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
        ctx.comptime_args.insert({name, template_args[i]});
    }

    for (const auto& directive : def.directives) {
        stream.seek(directive);
        process_directive(stream, builder, ctx);
    }

    for (const auto& param : def.template_params) {
        std::string name = stream.span(param.name);
        Expr e = nullptr;

        if (ctx.config_args.count(name) > 0) {
            e = ctx.config_args.at(name);
        } else if (ctx.comptime_args.count(name) > 0) {
            e = ctx.comptime_args.at(name);
        } else if (ctx.user_args.count(name) > 0) {
            e = ctx.user_args.at(name);
        } else {
            stream.throw_unexpected_token(
                param.name,
                "this template parameter is not defined, please add "
                "`#pragma kernel tune("
                    + name + "=...)`");
        }

        builder.template_arg(e);
    }

    return builder;
}

}  // namespace internal
}  // namespace kernel_launcher
