#pragma once

#if defined(__GNUC__)
    #define EXPORT __attribute__((visibility("default")))
    #define IMPORT
#else
    #define EXPORT __declspec(dllexport)
    #define IMPORT __declspec(dllimport)
#endif

#ifdef BUILDING_DLL
#define CN_API EXPORT
#else
#define CN_API IMPORT
#endif