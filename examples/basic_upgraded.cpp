#include <savepoint/savepoint.hpp>

#include <cassert>
#include <filesystem>

static constexpr SavepointVersion kVersion1{0, 0, 0};

// Added new versions
static constexpr SavepointVersion kVersionAddedZ{0, 0, 1};
static constexpr SavepointVersion kVersionAddedW{0, 0, 2};

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

struct EntityV2
{
    int X;
    int Z; // Added in 0.0.1
    int Y;
    int W; // Added in 0.0.2
    SavepointID ID;

    void Visit(SavepointVisitor& visitor)
    {
        visitor(X);
        visitor(Z, kVersionAddedZ, 3); // Set the version and default value (optional)
        visitor(Y);
        visitor(W, kVersionAddedW, 5); // Set the version and default value (optional)
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
    savepoint.Open(SavepointDriver::Sqlite3, "savepoint.sqlite3", kVersion1);

    // Writing the V1 entity
    EntityV1 inEntity{1, 2};
    savepoint.Write(inEntity, inEntity.ID, 0);

    // Reading the V1 entity as a V2 entity
    savepoint.Read<EntityV2>([&](EntityV2& outEntity, SavepointID id)
    {
        assert(outEntity == inEntity); // Check old values
        assert(outEntity.Z == 3); // Check default value
        assert(outEntity.W == 5); // Check default value
    }, 0);

    savepoint.Close();
    return 0;
}
