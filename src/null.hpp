#pragma once

#include <savepoint/savepoint.hpp>

#include <cstddef>
#include <cstdint>
#include <string_view>

class SavepointDriverNull : public ISavepointDriver
{
public:
    SavepointStatus Open(const std::string_view& path) override;
    bool IsOpen() const override;
    void Write(const void* data, size_t size) override;
    uint32_t Insert(const void* data, size_t size, int level) override;
    bool Update(const void* data, size_t size, uint32_t id, int level) override;
    void Write(const void* data, size_t size, int x, int y, int level) override;
    void Write(const void* data, size_t size, int x, int y, int z, int level) override;
    void Read(const SavepointReadDataFunction& function) override;
    void Read(const SavepointReadAllEntityDataFunction& function, int level) override;
    void Read(const SavepointReadAllTile2DDataFunction& function, int level) override;
    void Read(const SavepointReadAllTile3DDataFunction& function, int level) override;
    bool Read(const SavepointReadTile2DDataFunction& function, int level, int x, int y) override;
    bool Read(const SavepointReadTile3DDataFunction& function, int level, int x, int y, int z) override;
    void Read(const SavepointReadAllLevelsFunction& function) override;
    void Delete(uint32_t id) override;
    void Close() override;
    void Save() override;
    void Clear() override;
};
