#include <savepoint/savepoint.hpp>

#include <cassert>
#include <filesystem>

static constexpr SavepointVersion kVersion{0, 0, 0};

struct Entity
{
    int X;
    int Y;
    SavepointID ID;

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
    std::filesystem::remove("savepoint.sqlite3");

    Savepoint savepoint;
    switch (savepoint.Open(SavepointDriver::Sqlite3, "savepoint.sqlite3", kVersion))
    {
    case SavepointStatus::Failed:
        return 1;
    case SavepointStatus::Existing:
        break;
    case SavepointStatus::New:
        break;
    }

    Entity inEntity{1, 2};
    savepoint.Write(inEntity, inEntity.ID, 0);

    int reads = 0;
    savepoint.Read<Entity>([&](Entity& outEntity, SavepointID id)
    {
        outEntity.ID = id;
        assert(outEntity == inEntity);
        assert(id == inEntity.ID);
        reads++;
    }, 0);
    assert(reads == 1);

    savepoint.Save();
    savepoint.Close();
    return 0;
}