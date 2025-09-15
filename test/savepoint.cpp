#include <savepoint.hpp>
#include <cassert>
#include <filesystem>
#include <memory>
#include <string>

static const std::string kFileName = "savepoint.sqlite3";

static constexpr SavepointVersion kVersion1{0, 0, 1};
static constexpr SavepointVersion kVersion2{0, 1, 0};
static constexpr SavepointVersion kVersion3{0, 2, 0};
static constexpr SavepointVersion kVersion4{1, 1, 0};

struct ItemV2
{
    int Durability = 1;
    int Damage = 2;

    void Visit(SavepointArchive& archive)
    {
        archive(Durability);
        archive(Damage);
    }

    bool operator==(const ItemV2& other) const
    {
        return Durability == other.Durability &&
            Damage == other.Damage;
    }
};

struct ItemV4
{
    int Durability = 1;
    int Rarity = 2;
    int Damage = 3;

    void Visit(SavepointArchive& archive)
    {
        archive(Durability);
        archive(Damage);
        archive(Rarity, kVersion4);
    }

    bool operator==(const ItemV2& other) const
    {
        return Durability == other.Durability &&
            Rarity == 2 &&
            Damage == other.Damage;
    }

    bool operator==(const ItemV4& other) const
    {
        return Durability == other.Durability &&
            Rarity == other.Rarity &&
            Damage == other.Damage;
    }
};

struct EntityV1
{
    float X = 1.0f;
    float Y = 2.0f;
    int Health = 3;

    void Visit(SavepointArchive& archive)
    {
        archive(X);
        archive(Y);
        archive(Health);
    }

    bool operator==(const EntityV1& other) const
    {
        return X == other.X &&
            Y == other.Y &&
            Health == other.Health;
    }
};

struct EntityV2
{
    int Strength = 4;
    float X = 1.0f;
    float Y = 2.0f;
    ItemV2 Item;
    int Health = 3;

    void Visit(SavepointArchive& archive)
    {
        archive(Strength, kVersion2);
        archive(X);
        archive(Y);
        archive(Item, kVersion2);
        archive(Health);
    }

    bool operator==(const EntityV1& other) const
    {
        return Strength == 4 &&
            X == other.X &&
            Y == other.Y &&
            Item == ItemV2{} &&
            Health == other.Health;
    }

    bool operator==(const EntityV2& other) const
    {
        return Strength == other.Strength &&
            X == other.X &&
            Y == other.Y &&
            Item == other.Item &&
            Health == other.Health;
    }
};

struct EntityV3
{
    int Strength = 4;
    float X = 1.0f;
    float Y = 2.0f;
    ItemV2 Item;
    int Health = 3;
    int Intelligence = 5;

    void Visit(SavepointArchive& archive)
    {
        archive(Strength, kVersion2);
        archive(X);
        archive(Y);
        archive(Item, kVersion2);
        archive(Health);
        archive(Intelligence, kVersion3);
    }

    bool operator==(const EntityV1& other) const
    {
        return Strength == 4 &&
            X == other.X &&
            Y == other.Y &&
            Item == ItemV2{} &&
            Health == other.Health &&
            Intelligence == 5;
    }

    bool operator==(const EntityV2& other) const
    {
        return Strength == other.Strength &&
            X == other.X &&
            Y == other.Y &&
            Item == other.Item &&
            Health == other.Health &&
            Intelligence == 5;
    }

    bool operator==(const EntityV3& other) const
    {
        return Strength == other.Strength &&
            X == other.X &&
            Y == other.Y &&
            Item == other.Item &&
            Health == other.Health &&
            Intelligence == other.Intelligence;
    }
};

struct EntityV4
{
    int Strength = 4;
    float X = 1.0f;
    float Z = 6.0f;
    float Y = 2.0f;
    ItemV4 Item;
    int Health = 3;
    int Intelligence = 5;

    void Visit(SavepointArchive& archive)
    {
        archive(Strength, kVersion2);
        archive(X);
        archive(Y);
        archive(Z, kVersion4);
        archive(Item, kVersion2);
        archive(Health);
        archive(Intelligence, kVersion3);
    }

    bool operator==(const EntityV1& other) const
    {
        return Strength == 4 &&
            X == other.X &&
            Z == 6.0f &&
            Y == other.Y &&
            Item == ItemV4{} &&
            Health == other.Health &&
            Intelligence == 5;
    }

    bool operator==(const EntityV2& other) const
    {
        return Strength == other.Strength &&
            X == other.X &&
            Z == 6.0f &&
            Y == other.Y &&
            Item == other.Item &&
            Health == other.Health &&
            Intelligence == 5;
    }

    bool operator==(const EntityV3& other) const
    {
        return Strength == other.Strength &&
            X == other.X &&
            Z == 6.0f &&
            Y == other.Y &&
            Item == other.Item &&
            Health == other.Health &&
            Intelligence == 5;
    }

    bool operator==(const EntityV4& other) const
    {
        return Strength == other.Strength &&
            X == other.X &&
            Z == other.Z &&
            Y == other.Y &&
            Item == other.Item &&
            Health == other.Health &&
            Intelligence == other.Intelligence;
    }
};

struct ZombieV1 : EntityV1
{
    float Speed = 1.0f;

    void Visit(SavepointArchive& archive)
    {
        EntityV1::Visit(archive);
        archive(Speed);
    }
};

struct ZombieV21 : EntityV1
{
    float Speed = 1.0f;
    float VelocityX = 2.0f;

    void Visit(SavepointArchive& archive)
    {
        EntityV1::Visit(archive);
        archive(Speed);
        archive(VelocityX, kVersion2);
    }
};

struct ZombieV22 : EntityV2
{
    float Speed = 1.0f;
    float VelocityX = 2.0f;

    void Visit(SavepointArchive& archive)
    {
        EntityV2::Visit(archive);
        archive(Speed);
        archive(VelocityX, kVersion2);
    }
};

struct ZombieV4 : EntityV4
{
    float Speed = 1.0f;
    float VelocityX = 2.0f;
    float VelocityY = 2.0f;
    float VelocityZ = 2.0f;

    void Visit(SavepointArchive& archive)
    {
        EntityV4::Visit(archive);
        archive(Speed);
        archive(VelocityX, kVersion2);
        archive(VelocityY, kVersion4);
        archive(VelocityZ, kVersion4);
    }
};

template<typename InT, typename OutT>
void TestUpgrade(Savepoint& savepoint, SavepointVersion inVersion)
{
    SavepointArchive inArchive{inVersion};
    SavepointID inEntityID;
    std::unique_ptr<InT> inEntity = std::make_unique<InT>();
    inEntity->Visit(inArchive);
    savepoint.Write(inArchive, inEntityID);
    int i = 0;
    savepoint.Read([&](SavepointArchive& outArchive, SavepointID outEntityID)
    {
        std::unique_ptr<OutT> outEntity = std::make_unique<OutT>();
        outEntity->Visit(outArchive);
        assert(*outEntity == *inEntity);
        assert(outEntityID == inEntityID);
        i++;
    });
    assert(i == 1);
    savepoint.Clear();
};

int main()
{
    std::filesystem::remove(kFileName);
    std::filesystem::remove(kFileName + "-journal");
    Savepoint savepoint;
    assert(savepoint.Open(kFileName));
    TestUpgrade<EntityV1, EntityV1>(savepoint, kVersion1);
    TestUpgrade<EntityV1, EntityV2>(savepoint, kVersion1);
    TestUpgrade<EntityV1, EntityV3>(savepoint, kVersion1);
    TestUpgrade<EntityV1, EntityV4>(savepoint, kVersion1);
    TestUpgrade<EntityV2, EntityV2>(savepoint, kVersion2);
    TestUpgrade<EntityV2, EntityV3>(savepoint, kVersion2);
    TestUpgrade<EntityV2, EntityV4>(savepoint, kVersion2);
    TestUpgrade<EntityV3, EntityV3>(savepoint, kVersion3);
    TestUpgrade<EntityV3, EntityV4>(savepoint, kVersion3);
    TestUpgrade<EntityV4, EntityV4>(savepoint, kVersion4);
    savepoint.Save();
    savepoint.Close();
    return 0;
}