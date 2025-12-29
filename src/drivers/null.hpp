#pragma once

#include <savepoint/driver.hpp>

#include <string_view>

class SavepointDriverNull : public ISavepointDriver
{
public:
    SavepointStatus Open(const std::string_view& path, SavepointVersion version) override;
    bool IsOpen() const override;
    void Close() override;
    void Save() override;
    void Write(SavepointVisitor& visitor) override;
    void Write(SavepointVisitor& visitor, SavepointID& id, int level) override;
    void Write(SavepointVisitor& visitor, int x, int y, int level) override;
    void Write(SavepointVisitor& visitor, int x, int y, int z, int level) override;
    void Write(SavepointBase* base) override;
    void Write(SavepointBase* base, SavepointID& id, int level) override;
    void Write(SavepointBase* base, int x, int y, int level) override;
    void Write(SavepointBase* base, int x, int y, int z, int level) override;
    void Read(const SavepointReadVisitorFunction& function) override;
    void Read(const SavepointReadVisitorEntityFunction& function, int level) override;
    void Read(const SavepointReadVisitorTile2DFunction& function, int level) override;
    void Read(const SavepointReadVisitorTile3DFunction& function, int level) override;
    void Read(const SavepointReadBaseFunction& function) override;
    void Read(const SavepointReadBaseEntityFunction& function, int level) override;
    void Read(const SavepointReadBaseTile2DFunction& function, int level) override;
    void Read(const SavepointReadBaseTile3DFunction& function, int level) override;
    void Delete(const SavepointID id) override;
    void Clear() override;
};
