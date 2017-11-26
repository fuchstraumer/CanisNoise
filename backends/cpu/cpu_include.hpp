#pragma once
#ifndef CPU_INCLUDE_HPP
#define CPU_INCLUDE_HPP

#include <algorithm>
#include <numeric>
#include <future>
#include <functional>
#include <valarray>

#ifdef BUILDING_DLL
#define API_CALL __declspec(dllexport)
#else
#define API_CALL __declspec(dllimport)
#endif

#endif //!CPU_INCLUDE_HPP