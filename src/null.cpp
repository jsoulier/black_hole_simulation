#include <savepoint/savepoint.hpp>

#include <cstddef>
#include <cstdint>
#include <string_view>

#include "null.hpp"

SavepointStatus SavepointDriverNull::Open(const std::string_view& path)
{
    return SavepointStatus::New;
}

bool SavepointDriverNull::IsOpen() const
{
    return false;
}

void SavepointDriverNull::Write(const void* data, size_t size)
{
}

uint32_t SavepointDriverNull::Insert(const void* data, size_t size, int level)
{
    return SavepointID::kInvalidID; 
}

bool SavepointDriverNull::Update(const void* data, size_t size, uint32_t id, int level)
{
    return false;
}

void SavepointDriverNull::Write(const void* data, size_t size, int x, int y, int level)
{
}

void SavepointDriverNull::Write(const void* data, size_t size, int x, int y, int z, int level)
{
}

void SavepointDriverNull::Read(const SavepointReadDataFunction& function)
{
}

void SavepointDriverNull::Read(const SavepointReadAllEntityDataFunction& function, int level)
{
}

void SavepointDriverNull::Read(const SavepointReadAllTile2DDataFunction& function, int level)
{
}

void SavepointDriverNull::Read(const SavepointReadAllTile3DDataFunction& function, int level)
{
}

bool SavepointDriverNull::Read(const SavepointReadTile2DDataFunction& function, int level, int x, int y)
{
    return false;
}

bool SavepointDriverNull::Read(const SavepointReadTile3DDataFunction& function, int level, int x, int y, int z)
{
    return false;
}

void SavepointDriverNull::Read(const SavepointReadAllLevelsFunction& function)
{
}

void SavepointDriverNull::Delete(uint32_t id)
{
}

void SavepointDriverNull::Close()
{
}

void SavepointDriverNull::Save()
{
}

void SavepointDriverNull::Clear()
{
}
