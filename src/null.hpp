#pragma once

#include <savepoint/savepoint.hpp>

#include <cstdint>
#include <string_view>

class SavepointDriverNull : public ISavepointDriver
{
public:
    SavepointStatus Open(const std::string_view& path, SavepointVersion version) override;
    bool IsOpen() const override;
    void Write(SavepointVisitor& visitor) override;
    SavepointID Insert(SavepointVisitor& visitor, int level) override;
    SavepointID Update(SavepointVisitor& visitor, SavepointID id, int level) override;
    void Write(SavepointVisitor& visitor, int x, int y, int level) override;
    void Write(SavepointVisitor& visitor, int x, int y, int z, int level) override;
    void Read(const SavepointReadVisitorFunction& function) override;
    void Read(const SavepointReadVisitorEntityFunction& function, int level) override;
    void Read(const SavepointReadVisitorTile2DFunction& function, int level) override;
    void Read(const SavepointReadVisitorTile3DFunction& function, int level) override;
    void Read(const SavepointReadLevelFunction& function) override;
    void Delete(SavepointID id) override;
    void Close() override;
    void Save() override;
    void Clear() override;
};
