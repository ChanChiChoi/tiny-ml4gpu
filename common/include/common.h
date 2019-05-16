/*
 * the file borrowed from pybind11
 * */

#pragma once

#if !defined(NAMESPACE_BEGIN)
#    define NAMESPACE_BEGIN(name) namespace name {
#endif

#if !defined(NAMESPACE_END)
#    define NAMESPACE_END(name) }
# endif

#define MAX(x,y) ((x)>(y)?(x):(y))
#define MIN(x,y) ((x)<(y)?(x):(y))


