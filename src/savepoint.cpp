#include <filesystem>
#include <functional>
#include <savepoint.hpp>
#include "sqlite3.h"

Savepoint::Savepoint()
    : Handle{nullptr}
{
}

bool Savepoint::Open(const std::filesystem::path& path)
{
    return true;
}

void Savepoint::Close(bool save)
{
}

void Savepoint::Save()
{
}

void Savepoint::Write(const SavepointSerializer& serializer)
{
}

void Savepoint::Write(const SavepointSerializer& serializer, SavepointID& id, int level)
{
}

void Savepoint::Write(const SavepointSerializer& serializer, int x, int y, int level)
{
}

void Savepoint::Write(const SavepointSerializer& serializer, int x, int y, int z, int level)
{
}

SavepointSerializer Savepoint::Read()
{
    return {};
}

void Savepoint::Read(const SavepointEntityFunc& func, int level)
{
}

void Savepoint::Read(const SavepointTile2DFunc& func, int level)
{
}

void Savepoint::Read(const SavepointTile3DFunc& func, int level)
{
}

void Savepoint::Delete(const SavepointID id)
{
}