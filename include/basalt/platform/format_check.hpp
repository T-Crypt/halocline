// include/basalt/platform/format_check.hpp - lets the compiler check the printf-style arguments of a logging function.
#pragma once

// Marks a function whose argument `fmt` (1-based) is a printf format and whose arguments start at `first`.
#define BASALT_PRINTF_FORMAT(fmt, first) __attribute__((format(printf, fmt, first)))
