/*
 * This is free and unencumbered software released into the public domain.
 * 
 * Anyone is free to copy, modify, publish, use, compile, sell, or
 * distribute this software, either in source code form or as a compiled
 * binary, for any purpose, commercial or non-commercial, and by any
 * means.
 * 
 * In jurisdictions that recognize copyright laws, the author or authors
 * of this software dedicate any and all copyright interest in the
 * software to the public domain. We make this dedication for the benefit
 * of the public at large and to the detriment of our heirs and
 * successors. We intend this dedication to be an overt act of
 * relinquishment in perpetuity of all present and future rights to this
 * software under copyright law.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 * 
 * For more information, please refer to <https://unlicense.org>
 */

#include <savepoint/savepoint.hpp>

#include <format>
#include <memory>
#include <string_view>
#include <utility>

#include "drivers/null.hpp"
#include "drivers/sqlite3.hpp"

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

void Savepoint::Write(SavepointBase* base)
{
    if (!IsOpen())
    {
        return;
    }
    Driver->Write(base);
}

void Savepoint::Write(SavepointBase* base, SavepointID& id, int level)
{
    if (!IsOpen())
    {
        return;
    }
    Driver->Write(base, id, level);
}

void Savepoint::Write(SavepointBase* base, int x, int y, int level)
{
    if (IsOpen())
    {
        Driver->Write(base, x, y, level);
    }
}

void Savepoint::Write(SavepointBase* base, int x, int y, int z, int level)
{
    if (IsOpen())
    {
        Driver->Write(base, x, y, z, level);
    }
}

void Savepoint::Read(const SavepointReadBaseFunction& function)
{
    if (IsOpen())
    {
        Driver->Read(function);
    }
}

void Savepoint::Read(const SavepointReadBaseEntityFunction& function, int level)
{
    if (IsOpen())
    {
        Driver->Read(function, level);
    }
}

void Savepoint::Read(const SavepointReadBaseTile2DFunction& function, int level)
{
    if (IsOpen())
    {
        Driver->Read(function, level);
    }
}

void Savepoint::Read(const SavepointReadBaseTile3DFunction& function, int level)
{
    if (IsOpen())
    {
        Driver->Read(function, level);
    }
}

void Savepoint::Delete(const SavepointID id)
{
    if (IsOpen())
    {
        Driver->Delete(id);
    }
}

void Savepoint::Clear()
{
    if (IsOpen())
    {
        Driver->Clear();
    }
}
