#include <savepoint/savepoint.hpp>

#include <cassert>

struct Entity : SavepointEntity
{
    int X;
    int Y;

    Entity() = default;
    Entity(int x, int y)
        : X{x}
        , Y{y}
    {
    }

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
    savepoint.Open(SavepointDriver::SQLite3, "savepoint.sqlite3", SavepointVersion{});

    Entity inEntity{1, 2};
    savepoint.Write(inEntity, 0);
    savepoint.Read<Entity>([&](Entity& outEntity)
    {
        assert(outEntity == inEntity);
    }, 0);
    
    savepoint.Close();
    return 0;
}
