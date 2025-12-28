#include <savepoint.hpp>

#include <cassert>
#include <filesystem>

static constexpr SavepointVersion kVersion1{0, 0, 0};

struct Entity
{
    int X;
    int Y;
    SavepointID ID;

    // All complex serialized objects should implement
    void Visit(SavepointVisitor& visitor)
    {
        visitor(X);
        visitor(Y);
    }

    bool operator==(const Entity& other) const
    {
        return X == other.X && Y == other.Y;
    }
};

int main()
{
    // Cleanup
    std::filesystem::remove("savepoint.sqlite3");

    Savepoint savepoint;
    
    // Open or create a savepoint connection
    switch (savepoint.Open("savepoint.sqlite3", kVersion1))
    {
    case SavepointStatus::Failed:
        assert(false);
        return 1;
    case SavepointStatus::Existing:
        // Opened an existing savepoint
        break;
    case SavepointStatus::New:
        // Created a new savepoint
        break;
    }

    // Writing the entity
    Entity inEntity{1, 2};
    savepoint.Write(inEntity, inEntity.ID, 0);

    // Reading the entity
    int reads = 0;
    savepoint.Read<Entity>([&](Entity& outEntity, SavepointID id)
    {
        outEntity.ID = id;
        assert(outEntity == inEntity);
        assert(id == inEntity.ID);
        reads++;
    }, 0);
    assert(reads == 1);

    // Commit transaction
    savepoint.Save();

    // Close the savepoint connection
    savepoint.Close();

    return 0;
}