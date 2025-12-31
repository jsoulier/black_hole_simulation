#pragma once

#include <savepoint/driver.hpp>
#include <savepoint/fwd.hpp>
#include <savepoint/id.hpp>
#include <savepoint/status.hpp>
#include <savepoint/version.hpp>

#include <string_view>

class SavepointDriverNull : public ISavepointDriver
{
public:
    SavepointStatus Open(const std::string_view& path, SavepointVersion version) override;
    bool IsOpen() const override;
    void Write(SavepointVisitor& visitor) override;
    void Write(SavepointVisitor& visitor, SavepointID& id, int level) override;
    void Write(SavepointVisitor& visitor, int x, int y, int level) override;
    void Write(SavepointVisitor& visitor, int x, int y, int z, int level) override;
    void Read(const SavepointReadVisitorFunction& function) override;
    void Read(const SavepointReadVisitorEntityFunction& function, int level) override;
    void Read(const SavepointReadVisitorTile2DFunction& function, int level) override;
    void Read(const SavepointReadVisitorTile3DFunction& function, int level) override;
    void Delete(const SavepointID id) override;
    void Close() override;
    void Save() override;
    void Clear() override;
};
