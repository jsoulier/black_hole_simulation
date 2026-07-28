#include <savepoint/savepoint.hpp>

#include <cstddef>
#include <cstdio>
#include <format>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

#if SAVEPOINT_NULL
#include "null.hpp"
#endif
#if SAVEPOINT_SQLITE3
#include "sqlite3.hpp"
#endif

static void DefaultLogFunction(const std::string_view& string);

static SavepointLogFunction logFunction = DefaultLogFunction;

struct Hash
{
    using is_transparent = void;

    size_t operator()(const std::string_view& string) const
    {
        return std::hash<std::string_view>{}(string);
    }

    size_t operator()(const std::string& string) const
    {
        return std::hash<std::string_view>{}(string);
    }
};

static auto& GetPolyFunctions()
{
    // Avoiding SIOF
    static std::unordered_map<std::string, SavepointPolyFunction, Hash, std::equal_to<>> functions;
    return functions;
}

static void DefaultLogFunction(const std::string_view& string)
{
    std::fprintf(stderr, "%s\n", string.data());
}

void SavepointSetLogFunction(const SavepointLogFunction& function)
{
    logFunction = function;
}

void SavepointLog(const std::string_view& string)
{
    logFunction(string);
}

void SavepointAddPolyFunction(const std::string_view& string, const SavepointPolyFunction function)
{
    GetPolyFunctions().emplace(string, function);
}

SavepointPolyFunction SavepointGetPolyFunction(const std::string_view& string)
{
    auto it = GetPolyFunctions().find(string);
    if (it != GetPolyFunctions().end())
    {
        return it->second;
    }
    else
    {
        return nullptr;
    }
}

SavepointPoly* SavepointReadPoly(SavepointVisitor& visitor)
{
    std::string string;
    visitor(string);
    SavepointPolyFunction function = SavepointGetPolyFunction(string);
    if (function == nullptr)
    {
        SavepointLog(std::format("Failed to find poly string: {}", string));
        visitor.SetError();
        return nullptr;
    }
    SavepointPoly* poly = function();
    if (poly)
    {
        visitor(*poly);
    }
    else
    {
        SavepointLog(std::format("Failed to allocate poly: {}", string));
        visitor.SetError();
    }
    return poly;
}

void SavepointWritePoly(SavepointPoly* poly, SavepointVisitor& visitor)
{
    std::string_view string = poly->GetClassName();
    visitor(string);
    visitor(*poly);
}

void SavepointSkipString(SavepointVisitor& visitor)
{
    // TODO: There's a bug in MSVC. If I try this in SavepointVisit, the concept constraints explode.
    std::string string;
    visitor(string);
}

Savepoint::~Savepoint()
{
    if (Driver->IsOpen())
    {
        Close();
    }
}

SavepointStatus Savepoint::Open(SavepointDriver driver, const std::string_view& path, SavepointVersion version)
{
    switch (driver)
    {
#ifdef SAVEPOINT_NULL
    case SavepointDriver::Null:
        Driver = std::make_unique<SavepointDriverNull>();
        break;
#endif
#ifdef SAVEPOINT_SQLITE3
    case SavepointDriver::SQLite3:
        Driver = std::make_unique<SavepointDriverSQLite3>();
        break;
#endif
    default:
        SavepointLog(std::format("Unknown driver: {}", std::to_underlying(driver)));
        return SavepointStatus::Failed;
    }
    if (!Driver)
    {
        SavepointLog(std::format("Failed to create driver: {}", std::to_underlying(driver)));
        return SavepointStatus::Failed;
    }
    Version = version;
    return Driver->Open(path);
}

void Savepoint::Close()
{
    if (Driver->IsOpen())
    {
        Driver->Close();
    }
}

void Savepoint::Save()
{
    if (Driver->IsOpen())
    {
        Driver->Save();
    }
}

void Savepoint::Clear()
{
    if (Driver->IsOpen())
    {
        Driver->Clear();
    }
}
