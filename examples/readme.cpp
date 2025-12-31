#include <savepoint/savepoint.hpp>

#include <cassert>

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
    Savepoint savepoint;
    savepoint.Open(SavepointDriver::Sqlite3, "savepoint.sqlite3", SavepointVersion{});

    Entity inEntity{1, 2};
    savepoint.Write(inEntity, inEntity.ID, 0);
    savepoint.Read<Entity>([&](Entity& outEntity, SavepointID id)
    {
        assert(outEntity == inEntity);
        assert(id == inEntity.ID);
    }, 0);
    
    savepoint.Close();
    return 0;
}
