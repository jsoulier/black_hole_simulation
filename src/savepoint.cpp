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

static auto& GetDerivedFunctions()
{
    // Avoiding SIOF
    static std::unordered_map<std::string, SavepointDerivedFunction, Hash, std::equal_to<>> functions;
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

void SavepointAddDerivedFunction(const std::string_view& string, const SavepointDerivedFunction function)
{
    GetDerivedFunctions().emplace(string, function);
}

SavepointDerivedFunction SavepointGetDerivedFunction(const std::string_view& string)
{
    auto it = GetDerivedFunctions().find(string);
    if (it != GetDerivedFunctions().end())
    {
        return it->second;
    }
    else
    {
        return nullptr;
    }
}

bool SavepointWriteDerived(SavepointBase* base, SavepointVisitor& visitor)
{
    if (!base)
    {
        SavepointLog("Tried to write null base");
        return false;
    }
    std::string_view string = base->SavepointGetString();
    // TODO: Can this ever happen?
    SavepointDerivedFunction function = SavepointGetDerivedFunction(string);
    if (function == nullptr)
    {
        SavepointLog(std::format("Failed to find base string: {}", string));
        return false;
    }
    visitor(string);
    visitor(*base);
    return true;
}

SavepointBase* SavepointReadDerived(SavepointVisitor& visitor)
{
    std::string string;
    visitor(string);
    SavepointDerivedFunction function = SavepointGetDerivedFunction(string);
    if (function == nullptr)
    {
        SavepointLog(std::format("Failed to find base string: {}", string));
        return nullptr;
    }
    SavepointBase* base = function();
    if (!base)
    {
        SavepointLog(std::format("Failed to allocate base: {}", string));
        return nullptr;
    }
    visitor(*base);
    return base;
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
