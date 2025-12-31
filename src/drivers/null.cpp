#include "null.hpp"

SavepointStatus SavepointDriverNull::Open(const std::string_view& path, SavepointVersion version)
{
    return SavepointStatus::New;
}

bool SavepointDriverNull::IsOpen() const
{
    return false;
}

void SavepointDriverNull::Write(SavepointVisitor& visitor)
{
}

void SavepointDriverNull::Write(SavepointVisitor& visitor, SavepointID& id, int level)
{
}

void SavepointDriverNull::Write(SavepointVisitor& visitor, int x, int y, int level)
{
}

void SavepointDriverNull::Write(SavepointVisitor& visitor, int x, int y, int z, int level)
{
}

void SavepointDriverNull::Read(const SavepointReadVisitorFunction& function)
{
}

void SavepointDriverNull::Read(const SavepointReadVisitorEntityFunction& function, int level)
{
}

void SavepointDriverNull::Read(const SavepointReadVisitorTile2DFunction& function, int level)
{
}

void SavepointDriverNull::Read(const SavepointReadVisitorTile3DFunction& function, int level)
{
}

void SavepointDriverNull::Delete(const SavepointID id)
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
