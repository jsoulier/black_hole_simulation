// [external_types]
// TODO: comments
#include <savepoint/savepoint.hpp>

#include <cassert>
#include <filesystem>

static constexpr SavepointVersion kVersion{0, 0, 0};

struct ExternalType
{
    int Member1;
    int Member2;
};

void SavepointVisit(SavepointVisitor& visitor, ExternalType& external)
{
    visitor(external.Member1);
    visitor(external.Member2);
}

bool operator==(const ExternalType& lhs, const ExternalType& rhs)
{
    return lhs.Member1 == rhs.Member1 && lhs.Member2 == rhs.Member2;
}

struct Entity
{
    ExternalType External;
    SavepointID ID;

    void Visit(SavepointVisitor& visitor)
    {
        visitor(External);
    }

    bool operator==(const Entity& other) const
    {
        return External == other.External;
    }
};

int main()
{
    std::filesystem::remove("savepoint.sqlite3");

    Savepoint savepoint;
    savepoint.Open(SavepointDriver::SQLite3, "savepoint.sqlite3", kVersion);

    Entity inEntity{1, 2};
    savepoint.Write(inEntity, inEntity.ID, 0);
    savepoint.Read<Entity>([&](Entity& outEntity, SavepointID id)
    {
        assert(outEntity == inEntity);
    }, 0);

    savepoint.Close();
    return 0;
}
// [external_types]
