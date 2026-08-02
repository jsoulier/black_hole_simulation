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

#include "null.hpp"
#if SAVEPOINT_SQLITE3
#include "sqlite3.hpp"
#endif

static void DefaultLogFunction(const std::string_view& string)
{
    std::fwrite(string.data(), sizeof(char), string.size(), stderr);
    std::fputc('\n', stderr);
}

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
    static std::unordered_map<std::string, SavepointPolyFunction, Hash, std::equal_to<>> functions;
    return functions;
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
    if (Driver && Driver->IsOpen())
    {
        Close();
    }
}

SavepointStatus Savepoint::Open(SavepointDriver driver, const std::string_view& path, SavepointVersion version, bool threadSafe, int maxWait)
{
    switch (driver)
    {
    case SavepointDriver::Null:
        Driver = std::make_unique<SavepointDriverNull>();
        break;
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
    SavepointStatus status = Driver->Open(path, threadSafe, maxWait);
    if (status == SavepointStatus::Failed && Driver->IsOpen())
    {
        Driver->Close();
    }
    return status;
}

void Savepoint::Close()
{
    if (Driver && Driver->IsOpen())
    {
        Driver->Close();
    }
}

void Savepoint::Save()
{
    if (Driver && Driver->IsOpen())
    {
        Driver->Save();
    }
}

void Savepoint::Clear()
{
    if (Driver && Driver->IsOpen())
    {
        Driver->Clear();
    }
}
