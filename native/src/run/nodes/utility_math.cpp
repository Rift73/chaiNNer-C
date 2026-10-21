/* Native entry points of utility math nodes (pattern: image_utility.cpp).
 *
 * The Python bodies are utility_scalar.cpp's math_node and round_node over CPython floats: NumberInput with
 * precision "unlimited" enforces every number to a float (clamp_number's float(value)) and rejects NaN, so the
 * operands here are non-NaN doubles. Float arithmetic is IEEE as CPython's float ops; pow and log call the
 * process's ucrtbase.dll functions that CPython calls (cn_crt). Every error whose text needs CPython's float
 * formatting, and every result a Value cannot hold (an int beyond int64), declines to the Python body (RunPython). */
#include "run_engine.hpp"

#include "cn_crt_math.h"

#include <cfloat>
#include <cmath>
#include <optional>
#include <string>

namespace {

using namespace cn::run;

/* DOUBLE_IS_ODD_INTEGER (floatobject.c). */
bool odd_integer(double x) { return std::fmod(std::fabs(x), 2.0) == 1.0; }

/* float_pow (floatobject.c) on non-NaN operands; nullopt where CPython raises (zero to a negative power, a complex
 * result, ERANGE) and where its errno decides (a subnormal result). */
std::optional<double> float_pow(double base, double exponent)
{
    if (exponent == 0.0) return 1.0;
    if (std::isinf(exponent)) {
        double magnitude = std::fabs(base);
        if (magnitude == 1.0) return 1.0;
        if ((exponent > 0.0) == (magnitude > 1.0)) return std::fabs(exponent);
        return 0.0;
    }
    if (std::isinf(base)) {
        bool odd = odd_integer(exponent);
        if (exponent > 0.0) return odd ? base : std::fabs(base);
        return odd ? std::copysign(0.0, base) : 0.0;
    }
    if (base == 0.0) {
        if (exponent < 0.0) return std::nullopt;
        return odd_integer(exponent) ? base : 0.0;
    }
    bool negate = false;
    if (base < 0.0) {
        if (exponent != std::floor(exponent)) return std::nullopt;
        base = -base;
        negate = odd_integer(exponent);
    }
    if (base == 1.0) return negate ? -1.0 : 1.0;
    double result = cn_crt.pow(base, exponent);
    if (std::isinf(result) || (result != 0.0 && std::fabs(result) < DBL_MIN)) return std::nullopt;
    return negate ? -result : result;
}

/* math.log's loghelper for a float (math_1 over m_log); nullopt for the domain errors, whose text formats x. */
std::optional<double> log_helper(double x)
{
    if (std::isinf(x)) return x > 0.0 ? std::optional<double>(x) : std::nullopt;
    if (x > 0.0) return cn_crt.log(x);
    return std::nullopt;
}

/* math.floor / math.ceil of a float: the int as a double (+ 0.0: an int has no negative zero, so it converts back
 * to +0.0), or their errors. */
double integral(double x, bool ceiling)
{
    if (std::isinf(x)) throw NodeError("OverflowError", "cannot convert float infinity to integer");
    if (std::isnan(x)) throw NodeError("ValueError", "cannot convert float NaN to integer");
    return (ceiling ? std::ceil(x) : std::floor(x)) + 0.0;
}

[[noreturn]] void division_by_zero() { throw NodeError("ZeroDivisionError", "division by zero"); }

/* chainner:utility:math (math/math.py, utility_scalar.cpp math_node). */
void math(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    const std::string &op = std::get<std::string>(inputs[0]);
    double a = std::get<double>(inputs[1]);
    double b = std::get<double>(inputs[2]);
    double result;
    if (op == "add") {
        result = a + b;
    } else if (op == "sub") {
        result = a - b;
    } else if (op == "mul") {
        result = a * b;
    } else if (op == "div") {
        if (b == 0.0) division_by_zero();
        result = a / b;
    } else if (op == "pow") {
        std::optional<double> power = float_pow(a, b);
        if (!power) throw RunPython{};  // "{a}^{b} is not defined for real numbers." formats both floats
        result = *power;
    } else if (op == "log") {
        std::optional<double> numerator = log_helper(b);
        std::optional<double> denominator = log_helper(a);
        if (!numerator || !denominator) throw RunPython{};
        if (*denominator == 0.0) division_by_zero();
        result = *numerator / *denominator;
    } else if (op == "max") {
        result = b > a ? b : a;
    } else if (op == "min") {
        result = b < a ? b : a;
    } else if (op == "mod") {
        // _special_mod_numbers: 0.0 and the infinities match by ==; its NaN only by identity, and no input is NaN.
        auto special = [](double x) { return x == 0.0 || std::isinf(x); };
        if (special(a) || special(b)) {
            if (b == 0.0) division_by_zero();
            // floor's int converts back to float exactly, so b * rounded is b * floor(a / b) in doubles.
            result = a - b * integral(a / b, false);
        } else {
            // float_rem (floatobject.c); b is nonzero here.
            result = std::fmod(a, b);
            if (result != 0.0) {
                if ((b < 0.0) != (result < 0.0)) result += b;
            } else {
                result = std::copysign(0.0, b);
            }
        }
    } else if (op == "percent") {
        result = a * b / 100.0;
    } else {
        throw RunPython{};  // enforce admits MathOperation values only
    }
    outputs.push_back(result);
}

/* chainner:utility:compare (math/compare.py): Comparison values as the if-chain tests them; None for no match. */
void compare(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    double left = std::get<double>(inputs[1]);
    double right = std::get<double>(inputs[2]);
    switch (std::get<int64_t>(inputs[0])) {
    case 0: outputs.push_back(left == right); return;
    case 1: outputs.push_back(left != right); return;
    case 2: outputs.push_back(left > right); return;
    case 3: outputs.push_back(left < right); return;
    case 4: outputs.push_back(left >= right); return;
    case 5: outputs.push_back(left <= right); return;
    default: outputs.push_back(std::monostate{}); return;
    }
}

/* chainner:utility:math_round (math/round.py, utility_scalar.cpp round_node). */
void round_number(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    double a = std::get<double>(inputs[0]);
    const std::string &operation = std::get<std::string>(inputs[1]);
    const std::string &scale = std::get<std::string>(inputs[2]);
    double multiple = std::get<double>(inputs[3]);
    double power = std::get<double>(inputs[4]);
    if (operation != "Round down" && operation != "Round up" && operation != "Round") throw RunPython{};
    // math.floor, math.ceil or half_up's math.floor(value + 0.5): an int, as a double.
    auto apply = [&](double value) {
        if (operation == "Round") return integral(value + 0.5, false);
        return integral(value, operation == "Round up");
    };
    if (scale == "Integer") {
        double rounded = apply(a);
        if (!(rounded >= -9223372036854775808.0 && rounded < 9223372036854775808.0)) throw RunPython{};
        outputs.push_back(int64_t(rounded));
        return;
    }
    if (scale == "Multiple of...") {
        outputs.push_back(apply(a / multiple) * multiple);
        return;
    }
    if (scale == "Power of...") {
        std::optional<double> numerator = log_helper(a);
        std::optional<double> denominator = log_helper(power);
        if (!numerator || !denominator) throw RunPython{};
        if (*denominator == 0.0) division_by_zero();
        double exponent = apply(*numerator / *denominator);
        // float ** int converts the int with PyLong_AsDouble: exact here, since exponent is a double's floor.
        std::optional<double> result = float_pow(power, exponent);
        if (!result) throw RunPython{};  // OverflowError from errno
        outputs.push_back(*result);
        return;
    }
    throw RunPython{};  // enforce admits RoundScale values only
}

}  // namespace

CN_RUN_NODES_DECLARE(utility_math)
{
    registry.add("chainner:utility:compare", compare);
    registry.add("chainner:utility:math", math);
    registry.add("chainner:utility:math_round", round_number);
}
