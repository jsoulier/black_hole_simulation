#include <savepoint.hpp>
#include <cstdio>
#include <filesystem>
#include <string>
#include "sqlite3.h"

static SavepointLogFunction logFunction = SavepointDefaultLogFunction;

void SavepointSetLogFunction(const SavepointLogFunction& function)
{
    logFunction = function;
}

void SavepointDefaultLogFunction(const std::string& string)
{
    std::fputs(string.data(), stderr);
}

void SavepointLog(const std::string& string)
{
    logFunction(string);
}

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

void Savepoint::Read(const SavepointEntityFunction& function, int level)
{
}

void Savepoint::Read(const SavepointTile2DFunction& function, int level)
{
}

void Savepoint::Read(const SavepointTile3DFunction& function, int level)
{
}

void Savepoint::Delete(const SavepointID id)
{
}