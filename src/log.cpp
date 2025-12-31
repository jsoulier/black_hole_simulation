#include <savepoint/log.hpp>

#include <cstdio>
#include <string_view>

static void DefaultLogFunction(const std::string_view& string)
{
    std::fprintf(stderr, "%s\n", string.data());
}

static SavepointLogFunction logFunction = DefaultLogFunction;

void SavepointSetLogFunction(const SavepointLogFunction& function)
{
    logFunction = function;
}

void SavepointLog(const std::string_view& string)
{
    logFunction(string);
}
