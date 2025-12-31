// [basic_upgraded]
#include <savepoint/savepoint.hpp>

#include <cassert>
#include <filesystem>

static constexpr SavepointVersion kVersion{0, 0, 0};

// The old entity from basic_usage.cpp
struct EntityV1
{
    int X;
    int Y;
    SavepointID ID;

    void Visit(SavepointVisitor& visitor)
    {
        visitor(X);
        visitor(Y);
    }
};

// Added new versions
static constexpr SavepointVersion kVersionAddedZ{0, 0, 1};
static constexpr SavepointVersion kVersionAddedW{0, 0, 2};

// The new entity
struct EntityV2
{
    int X;
    int Z; // Added a Z component in 0.0.1
    int Y;
    int W; // Added a W component in 0.0.2
    SavepointID ID;

    void Visit(SavepointVisitor& visitor)
    {
        visitor(X);
        visitor(Z, kVersionAddedZ, 0); // Deserialize if the version is >= 0.0.1 (otherwise default to 0)
        visitor(Y);
        visitor(W, kVersionAddedW, 1); // Deserialize if the version is >= 0.0.2 (otherwise default to 1) 
    }

    bool operator==(const EntityV1& other) const
    {
        return X == other.X && Y == other.Y;
    }
};

int main()
{
    std::filesystem::remove("savepoint.sqlite3");

    Savepoint savepoint;
    savepoint.Open(SavepointDriver::Sqlite3, "savepoint.sqlite3", kVersion);

    // Write a V1 entity
    EntityV1 inEntity{1, 2};
    savepoint.Write(inEntity, inEntity.ID, 0);
    
    // Read a V1 entity as a V2 entity
    savepoint.Read<EntityV2>([&](EntityV2& outEntity, SavepointID id)
    {
        // X and Y were read
        assert(outEntity == inEntity);
        
        // Z and W were not read
        assert(outEntity.Z == 0);
        assert(outEntity.W == 1);
    }, 0);

    savepoint.Close();
    return 0;
}
// [basic_upgraded]
