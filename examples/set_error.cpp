// [set_error]
#include <savepoint/savepoint.hpp>

#include <cassert>
#include <cstdint>
#include <filesystem>

static constexpr SavepointVersion kVersion{0, 0, 0};

struct Entity : SavepointEntity
{
    uint32_t Value;

    void OnCreate()
    {
        Value = 0xDEADBEEF;
    }

    void Visit(SavepointVisitor& visitor)
    {
        visitor(Value);
    }

    bool operator==(const Entity other) const
    {
        return Value == other.Value;
    }
};

struct ReadEntity : Entity
{
    void Visit(SavepointVisitor& visitor)
    {
        Entity::Visit(visitor);
        if (visitor.IsReading())
        {
            visitor.SetError();
        }
    }
};

struct WriteEntity : Entity
{
    void Visit(SavepointVisitor& visitor)
    {
        Entity::Visit(visitor);
        if (visitor.IsWriting())
        {
            visitor.SetError();
        }
    }
};

int main()
{
    std::filesystem::remove("savepoint.sqlite3");

    Savepoint savepoint;
    savepoint.Open(SavepointDriver::SQLite3, "savepoint.sqlite3", kVersion);

    Entity inEntity;
    inEntity.OnCreate();

    auto hasSingleEntity = [&]()
    {
        int reads = 0;
        savepoint.Read<Entity>([&](Entity& outEntity)
        {
            assert(outEntity == inEntity);
            reads++;
        }, 0);
        return reads == 1;
    };

    savepoint.Write(inEntity, 0);
    assert(hasSingleEntity());

    savepoint.Read<ReadEntity>([](ReadEntity& outReadEntity) { assert(false); }, 0);
    savepoint.Delete(inEntity);
    WriteEntity inWriteEntity;
    savepoint.Write(inWriteEntity, 0);
    savepoint.Read<ReadEntity>([&](Entity& outEntity) { assert(false); }, 0);

    savepoint.Write(inEntity, 0);
    assert(hasSingleEntity());

    savepoint.Close();
    return 0;
}
// [set_error]
