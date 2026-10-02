#include "kernel_launcher/internal/directives.hpp"

#include <array>
#include <cerrno>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "kernel_launcher/expr.hpp"

namespace kernel_launcher {
namespace internal {

struct Context {
    // Runtime arguments of the kernel
    std::unordered_map<std::string, ArgumentExpr> kernel_args;

    // Parameters from configuration space
    std::unordered_map<std::string, TunableParam> config_args;

    // Compile-time arguments passed by the user
    std::unordered_map<std::string, Primitive> comptime_args;

    // User-defined parameters from `#pragma kernel set(foo=1+2)`
    std::unordered_map<std::string, ExprPtr> user_args;

    bool is_defined(const std::string& name) const {
        return config_args.count(name) > 0 || comptime_args.count(name) > 0
            || user_args.count(name) > 0;
    }
};

static ExprPtr parse_expr(TokenStream& stream, const Context& ctx);

static ExprPtr
process_function_call(Token t, const TokenStream& stream, const std::vector<ExprPtr>& args) {
    std::string name = stream.span(t);
    auto assert_nargs = [&](size_t n) {
        if (n != args.size()) {
            stream.throw_unexpected_token(
                t,
                "function expects " + std::to_string(n) + " arguments but "
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
    } else if (name == "min" || name == "max") {
        if (args.empty()) {
            stream.throw_unexpected_token(t, "function expects at least 1 argument");
        }

        ExprPtr result = args[0];
        for (size_t i = 1; i < args.size(); i++) {
            result = into_expr(name == "min" ? min(result, args[i]) : max(result, args[i]));
        }

        return result;
    } else if (name == "float") {
        assert_nargs(1);
        return to_float(args[0]);
    } else {
        stream.throw_unexpected_token(t, "unknown function name");
    }
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
static ExprPtr parse_ident(Token t, TokenStream& stream, const Context& ctx) {
    if (stream.next_if(TokenKind::ParenL)) {
        std::vector<ExprPtr> args;

        if (!stream.next_if(TokenKind::ParenR)) {
            do {
                args.push_back(parse_expr(stream, ctx));
            } while (stream.next_if(TokenKind::Comma));

            stream.consume(TokenKind::ParenR);
        }

        return process_function_call(t, stream, args);
    }

    std::string name = stream.span(t);

    if (name == "null") {
        return ExprPtr::make<ScalarExpr>();
    }

    if (name == "true") {
        return into_expr(true);
    }

    if (name == "false") {
        return into_expr(false);
    }

    // Is it a config parameter?
    {
        auto it = ctx.config_args.find(name);
        if (it != ctx.config_args.end()) {
            return into_expr(it->second);
        }
    }

    // Is it a compile-time parameter?
    {
        auto it = ctx.comptime_args.find(name);
        if (it != ctx.comptime_args.end()) {
            return into_expr(it->second);
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
            return into_expr(it->second);
        }
    }

#define KERNEL_LAUNCHER_IMPL(attr)                                      \
    if (name == "DEVICE_" #attr) {                                      \
        return into_expr(device_attribute(CU_DEVICE_ATTRIBUTE_##attr)); \
    }

    KERNEL_LAUNCHER_FOR_EACH_DEVICE_ATTRIBUTE(KERNEL_LAUNCHER_IMPL)
#undef KERNEL_LAUNCHER_IMPL

    stream.throw_unexpected_token(t, "unknown variable name");
}

static bool parse_string(std::string_view input, std::string& output) {
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
    errno = 0;
    output = strtol(input.c_str(), &endptr, 10);
    return *endptr == '\0' && errno == 0;
}

static ExprPtr parse_prim(TokenStream& stream, const Context& ctx) {
    Token t = stream.next();

    if (t.kind == TokenKind::Ident) {
        return parse_ident(t, stream, ctx);
    } else if (t.kind == TokenKind::ParenL) {
        ExprPtr e = parse_expr(stream, ctx);
        stream.consume(TokenKind::ParenR);
        return e;
    } else if (t.kind == TokenKind::String) {
        std::string out;
        if (!parse_string(stream.span(t), out)) {
            stream.throw_unexpected_token(t, "failed to parse string");
        }
        return into_expr(out);
    } else if (t.kind == TokenKind::Number) {
        long l;
        if (!parse_long(stream.span(t), l)) {
            stream.throw_unexpected_token(t, "failed to parse as integer");
        }
        return into_expr(l);
    } else if (stream.matches(t, '-')) {
        return -parse_prim(stream, ctx);
    } else if (stream.matches(t, '+')) {
        return parse_prim(stream, ctx);
    } else if (stream.matches(t, '!')) {
        return !parse_prim(stream, ctx);
    } else {
        stream.throw_unexpected_token(t, "expecting expression");
    }
}

// Precedence levels follow C: multiplicative (6), additive (5),
// relational (4), equality (3), logical and (2), logical or (1).
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
static ExprPtr parse_binop(TokenStream& stream, const Context& ctx, int prec) {
    ExprPtr lhs = parse_prim(stream, ctx);

    while (true) {
        if (prec < 6 && stream.next_if('*')) {
            lhs = lhs * parse_binop(stream, ctx, 6);
        } else if (prec < 6 && stream.next_if('/')) {
            lhs = lhs / parse_binop(stream, ctx, 6);
        } else if (prec < 6 && stream.next_if('%')) {
            lhs = lhs % parse_binop(stream, ctx, 6);
        } else if (prec < 5 && stream.next_if('+')) {
            lhs = lhs + parse_binop(stream, ctx, 5);
        } else if (prec < 5 && stream.next_if('-')) {
            lhs = lhs - parse_binop(stream, ctx, 5);
        } else if (prec < 4 && stream.next_if('<')) {
            lhs = lhs < parse_binop(stream, ctx, 4);
        } else if (prec < 4 && stream.next_if('>')) {
            lhs = lhs > parse_binop(stream, ctx, 4);
        } else if (prec < 4 && stream.next_if("<=")) {
            lhs = lhs <= parse_binop(stream, ctx, 4);
        } else if (prec < 4 && stream.next_if(">=")) {
            lhs = lhs >= parse_binop(stream, ctx, 4);
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

static ExprPtr parse_ternary(TokenStream& stream, const Context& ctx) {
    ExprPtr cond = parse_binop(stream, ctx, 0);

    if (stream.next_if('?')) {
        ExprPtr if_true = parse_expr(stream, ctx);
        stream.consume(':');
        ExprPtr if_false = parse_expr(stream, ctx);
        return into_expr(ifelse(std::move(cond), std::move(if_true), std::move(if_false)));
    } else {
        return cond;
    }
}

static ExprPtr parse_expr(TokenStream& stream, const Context& ctx) {
    return parse_ternary(stream, ctx);
}

static std::vector<ExprPtr>
parse_expr_list(TokenStream& stream, const Context& ctx, size_t max_params = 1024) {
    std::vector<ExprPtr> output;

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

static std::array<ExprPtr, 3> parse_expr_list3(TokenStream& stream, const Context& ctx) {
    auto list = parse_expr_list(stream, ctx, 3);
    return {
        list.size() > 0 ? list[0] : into_expr(1),  // NOLINT
        list.size() > 1 ? list[1] : into_expr(1),
        list.size() > 2 ? list[2] : into_expr(1),
    };
}

struct DummyEval: EvalContext {
    bool try_lookup(const Variable& /*v*/, Primitive& /*out*/) const override {
        return false;
    }
};

static Primitive parse_comptime_expr(TokenStream& stream, const Context& ctx) {
    Token before = stream.peek();
    ExprPtr e = parse_expr(stream, ctx);
    Token after = stream.peek();

    try {
        // We can use `DummyEval` to evaluate the expression since compile-time
        // expression should not contain any variables.
        return e.eval(DummyEval {});
    } catch (const std::exception& err) {
        auto msg = std::string("error while evaluating expression: ") + err.what();
        stream.throw_unexpected_token(before.begin, after.begin, msg);
    }
}

static void parse_buffer_directive(TokenStream& stream, KernelBuilder& builder, Context& ctx) {
    stream.consume(TokenKind::ParenL);
    do {
        Token var_token = stream.consume(TokenKind::Ident);
        stream.consume(TokenKind::BracketL);
        ExprPtr length = parse_expr(stream, ctx);
        stream.consume(TokenKind::BracketR);

        auto it = ctx.kernel_args.find(stream.span(var_token));
        if (it == ctx.kernel_args.end()) {
            stream.throw_unexpected_token(var_token, "this is not the name of a kernel argument");
        }

        builder.buffer_size(it->second, length);
    } while (stream.next_if(TokenKind::Comma));
    stream.consume(TokenKind::ParenR);
}

static void parse_tune_directive(TokenStream& stream, KernelBuilder& builder, Context& ctx) {
    std::vector<Primitive> values;

    stream.consume(TokenKind::ParenL);
    Token var_token = stream.consume(TokenKind::Ident);
    stream.consume('=');

    do {
        values.push_back(parse_comptime_expr(stream, ctx));
    } while (stream.next_if(TokenKind::Comma));

    stream.consume(TokenKind::ParenR);

    Primitive default_value = values[0];
    if (stream.next_if("default")) {
        stream.consume(TokenKind::ParenL);
        default_value = parse_comptime_expr(stream, ctx);
        stream.consume(TokenKind::ParenR);
    }

    std::string var = stream.span(var_token);

    if (ctx.config_args.count(var) > 0 || ctx.user_args.count(var) > 0) {
        stream.throw_unexpected_token(var_token, "variable redefined");
    }

    auto param = builder.tune(var, std::move(values), std::move(default_value));
    ctx.config_args.insert({var, param});
}

static void parse_set_directive(TokenStream& stream, Context& ctx) {
    // '(' IDENT '=' EXPR ')'
    stream.consume(TokenKind::ParenL);
    do {
        Token var_token = stream.consume(TokenKind::Ident);
        std::string var = stream.span(var_token);
        if (ctx.is_defined(var)) {
            stream.throw_unexpected_token(var_token, "variable redefined");
        }

        stream.consume('=');

        ExprPtr expr = parse_expr(stream, ctx);
        ctx.user_args.insert({var, expr});
    } while (stream.next_if(TokenKind::Comma));

    stream.consume(TokenKind::ParenR);
}

static void parse_tuning_key_directive(TokenStream& stream, KernelBuilder& builder, Context& ctx) {
    std::string key;

    stream.consume(TokenKind::ParenL);
    do {
        key += parse_comptime_expr(stream, ctx).to_string();
    } while (stream.next_if(TokenKind::Comma));
    stream.consume(TokenKind::ParenR);

    builder.tuning_key(key);
}

template<typename... Ts>
static bool one_of(std::string_view value, Ts&... items) {
    static constexpr size_t N = sizeof...(Ts);
    const char* array[N] = {items...};

    for (size_t i = 0; i < N; i++) {
        if (value == array[i]) {
            return true;
        }
    }

    return false;
}

static void process_directive(TokenStream& stream, KernelBuilder& builder, Context& ctx) {
    while (!stream.next_if(TokenKind::DirectiveEnd)) {
        Token t = stream.consume(TokenKind::Ident);
        std::string name = stream.span(t);

        if (name == "tune") {
            parse_tune_directive(stream, builder, ctx);
        } else if (one_of(name, "set", "let")) {
            parse_set_directive(stream, ctx);
        } else if (one_of(name, "buffer", "buffers")) {
            parse_buffer_directive(stream, builder, ctx);
        } else if (name == "tuning_key") {
            parse_tuning_key_directive(stream, builder, ctx);
        } else if (one_of(name, "grid_size", "grid_dim")) {
            auto l = parse_expr_list3(stream, ctx);
            builder.grid_size(l[0], l[1], l[2]);
        } else if (one_of(name, "block_size", "block_dim")) {
            auto l = parse_expr_list3(stream, ctx);
            builder.block_size(l[0], l[1], l[2]);
        } else if (one_of(name, "grid_divisor", "grid_divisors")) {
            auto l = parse_expr_list3(stream, ctx);
            builder.grid_divisor(l[0], l[1], l[2]);
        } else if (one_of(name, "problem_size", "problem_dim")) {
            auto l = parse_expr_list3(stream, ctx);
            builder.problem_size(l[0], l[1], l[2]);
        } else if (name == "if" || one_of(name, "restriction", "restrictions", "restrict")) {
            for (const auto& expr : parse_expr_list(stream, ctx)) {
                builder.restriction(expr);
            }
        } else if (one_of(name, "assertion", "assertions", "assert")) {
            for (const auto& expr : parse_expr_list(stream, ctx)) {
                builder.assertion(expr);
            }
        } else {
            stream.throw_unexpected_token(t, "this directive is not supported by kernel launcher");
        }
    }
}

KernelBuilder builder_from_annotated_kernel(
    TokenStream& stream,
    FileSource source,
    const AnnotatedKernelSpec& def,
    const std::vector<Primitive>& template_args) {
    KernelBuilder builder(def.qualified_name, std::move(source));

    Context ctx;

    for (size_t i = 0; i < def.fun_params.size(); i++) {
        std::string name = stream.span(def.fun_params[i].name);
        ctx.kernel_args.insert({name, ArgumentExpr(i)});
    }

    if (template_args.size() > def.template_params.size()) {
        throw std::runtime_error(concat(
            "cannot provide ",
            template_args.size(),
            " arguments to kernel ",
            def.qualified_name,
            " since it takes at most ",
            def.template_params.size(),
            " arguments"));
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

        // `TemplateArg` can only hold integral and boolean values.
        if (!param.is_integral) {
            stream.throw_unexpected_token(param.name, "type template parameters are not supported");
        }

        ExprPtr e = ExprPtr::make<ScalarExpr>(param.default_value);

        if (ctx.config_args.count(name) > 0) {
            e = into_expr(ctx.config_args.at(name));
        } else if (ctx.comptime_args.count(name) > 0) {
            e = into_expr(ctx.comptime_args.at(name));
        } else if (ctx.user_args.count(name) > 0) {
            e = ctx.user_args.at(name);
        } else if (param.default_value.is_empty()) {
            stream.throw_unexpected_token(
                param.name,
                "this template parameter is not defined, please add "
                "`#pragma kernel tune("
                    + name + "=...)`");
        }

        builder.template_arg(Expr<Primitive>(e));
    }

    return builder;
}

}  // namespace internal
}  // namespace kernel_launcher
