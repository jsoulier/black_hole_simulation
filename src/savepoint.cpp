#include <savepoint/savepoint.hpp>

#include <format>
#include <memory>
#include <string_view>
#include <utility>

#include "null.hpp"
#include "sqlite3.hpp"

Savepoint::~Savepoint()
{
    if (IsOpen())
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
    return Driver->Open(path, version);
}

bool Savepoint::IsOpen() const
{
    return Driver->IsOpen();
}

void Savepoint::Close()
{
    if (IsOpen())
    {
        Driver->Close();
    }
}

void Savepoint::Save()
{
    if (IsOpen())
    {
        Driver->Save();
    }
}

void Savepoint::Clear()
{
    if (IsOpen())
    {
        Driver->Clear();
    }
}
