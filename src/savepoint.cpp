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
    case SavepointDriver::Null:
        Driver = std::make_unique<SavepointDriverNull>();
        break;
    case SavepointDriver::Sqlite3:
        Driver = std::make_unique<SavepointDriverSqlite3>();
        break;
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

void Savepoint::Delete(const SavepointID id)
{
    if (IsOpen())
    {
        Driver->Delete(id);
    }
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
