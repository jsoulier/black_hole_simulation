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

    bool operator==(const EntityV3& other) const
    {
        return Strength == other.Strength &&
            X == other.X &&
            Y == other.Y &&
            Item == other.Item &&
            Health == other.Health;
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

    void Visit(SavepointArchive& archive)
    {
        archive(Strength, kVersion2);
        archive(X);
        archive(Y);
        archive(Z, kVersion4);
        archive(Item, kVersion2);
        archive(Health);
    }

    bool operator==(const EntityV1& other) const
    {
        return Strength == 4 &&
            X == other.X &&
            Z == 6.0f &&
            Y == other.Y &&
            Item == ItemV4{} &&
            Health == other.Health;
    }

    bool operator==(const EntityV2& other) const
    {
        return Strength == other.Strength &&
            X == other.X &&
            Z == 6.0f &&
            Y == other.Y &&
            Item == other.Item &&
            Health == other.Health;
    }

    bool operator==(const EntityV3& other) const
    {
        return Strength == other.Strength &&
            X == other.X &&
            Z == 6.0f &&
            Y == other.Y &&
            Item == other.Item &&
            Health == other.Health;
    }

    bool operator==(const EntityV4& other) const
    {
        return Strength == other.Strength &&
            X == other.X &&
            Z == other.Z &&
            Y == other.Y &&
            Item == other.Item &&
            Health == other.Health;
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

    bool operator==(const ZombieV1& other) const
    {
        return EntityV1::operator==(other) &&
            Speed == other.Speed;
    }
};

struct ZombieV2 : EntityV2
{
    float Speed = 1.0f;
    float VelocityX = 2.0f;

    void Visit(SavepointArchive& archive)
    {
        EntityV2::Visit(archive);
        archive(Speed);
        archive(VelocityX, kVersion2);
    }

    bool operator==(const ZombieV1& other) const
    {
        return EntityV2::operator==(other) &&
            Speed == other.Speed &&
            VelocityX == 2.0f;
    }

    bool operator==(const ZombieV2& other) const
    {
        return EntityV2::operator==(other) &&
            Speed == other.Speed &&
            VelocityX == other.VelocityX;
    }
};

struct ZombieV3 : EntityV3
{
    float Speed = 1.0f;
    float VelocityX = 2.0f;
    float VelocityY = 3.0f;
    float VelocityZ = 4.0f;

    void Visit(SavepointArchive& archive)
    {
        EntityV3::Visit(archive);
        archive(Speed);
        archive(VelocityX, kVersion2);
        archive(VelocityY, kVersion3);
        archive(VelocityZ, kVersion3);
    }

    bool operator==(const ZombieV1& other) const
    {
        return EntityV3::operator==(other) &&
            Speed == other.Speed &&
            VelocityX == 2.0f &&
            VelocityY == 3.0f &&
            VelocityZ == 4.0f;
    }

    bool operator==(const ZombieV2& other) const
    {
        return EntityV3::operator==(other) &&
            Speed == other.Speed &&
            VelocityX == other.VelocityX &&
            VelocityY == 3.0f &&
            VelocityZ == 4.0f;
    }

    bool operator==(const ZombieV3& other) const
    {
        return EntityV3::operator==(other) &&
            Speed == other.Speed &&
            VelocityX == other.VelocityX &&
            VelocityY == other.VelocityY &&
            VelocityZ == other.VelocityZ;
    }
};

struct ZombieV4 : EntityV4
{
    float Speed = 1.0f;
    float VelocityX = 2.0f;
    float VelocityY = 3.0f;
    float VelocityZ = 4.0f;

    void Visit(SavepointArchive& archive)
    {
        EntityV4::Visit(archive);
        archive(Speed);
        archive(VelocityX, kVersion2);
        archive(VelocityY, kVersion3);
        archive(VelocityZ, kVersion3);
    }

    bool operator==(const ZombieV1& other) const
    {
        return EntityV4::operator==(other) &&
            Speed == other.Speed &&
            VelocityX == 2.0f &&
            VelocityY == 3.0f &&
            VelocityZ == 4.0f;
    }

    bool operator==(const ZombieV2& other) const
    {
        return EntityV4::operator==(other) &&
            Speed == other.Speed &&
            VelocityX == other.VelocityX &&
            VelocityY == 3.0f &&
            VelocityZ == 4.0f;
    }

    bool operator==(const ZombieV3& other) const
    {
        return EntityV4::operator==(other) &&
            Speed == other.Speed &&
            VelocityX == other.VelocityX &&
            VelocityY == other.VelocityY &&
            VelocityZ == other.VelocityZ;
    }

    bool operator==(const ZombieV4& other) const
    {
        return EntityV4::operator==(other) &&
            Speed == other.Speed &&
            VelocityX == other.VelocityX &&
            VelocityY == other.VelocityY &&
            VelocityZ == other.VelocityZ;
    }
};

struct Tile2D
{
    int X;
    int Y;
};

struct Tile3D
{
    int X;
    int Y;
    int Z;

    void Visit(SavepointArchive& archive)
    {
        archive(X);
        archive(Y);
        archive(Z);
    }
};

template<typename InT, typename OutT>
void Test(Savepoint& savepoint, SavepointVersion inVersion)
{
    SavepointArchive inArchive{inVersion};
    SavepointID inEntityID;
    std::shared_ptr<InT> inEntity = std::make_shared<InT>();
    inArchive(*inEntity);
    savepoint.Write(inArchive, inEntityID, 0);
    int i = 0;
    savepoint.Read([&](SavepointArchive& outArchive, SavepointID outEntityID)
    {
        std::shared_ptr<OutT> outEntity = std::make_shared<OutT>();
        outArchive(*outEntity);
        assert(*outEntity == *inEntity);
        assert(outEntityID == inEntityID);
        i++;
    }, 0);
    assert(i == 1);
    savepoint.Clear();
};

int main()
{
    std::filesystem::remove(kFileName);
    std::filesystem::remove(kFileName + "-journal");
    Savepoint savepoint;
    assert(savepoint.Open(kFileName));

    // basic upgrade tests
    Test<EntityV1, EntityV1>(savepoint, kVersion1);
    Test<EntityV1, EntityV2>(savepoint, kVersion1);
    Test<EntityV2, EntityV2>(savepoint, kVersion2);
    Test<EntityV1, EntityV3>(savepoint, kVersion1);
    Test<EntityV2, EntityV3>(savepoint, kVersion2);
    Test<EntityV3, EntityV3>(savepoint, kVersion3);
    Test<EntityV1, EntityV4>(savepoint, kVersion1);
    Test<EntityV2, EntityV4>(savepoint, kVersion2);
    Test<EntityV3, EntityV4>(savepoint, kVersion3);
    Test<EntityV4, EntityV4>(savepoint, kVersion4);

    // inheritance upgrade tests
    Test<ZombieV1, ZombieV1>(savepoint, kVersion1);
    Test<ZombieV1, ZombieV2>(savepoint, kVersion1);
    Test<ZombieV2, ZombieV2>(savepoint, kVersion2);
    Test<ZombieV1, ZombieV3>(savepoint, kVersion1);
    Test<ZombieV2, ZombieV3>(savepoint, kVersion2);
    Test<ZombieV3, ZombieV3>(savepoint, kVersion3);
    Test<ZombieV1, ZombieV4>(savepoint, kVersion1);
    Test<ZombieV2, ZombieV4>(savepoint, kVersion2);
    Test<ZombieV3, ZombieV4>(savepoint, kVersion3);
    Test<ZombieV4, ZombieV4>(savepoint, kVersion4);

    // 2d spatial tests
    {
        SavepointArchive inArchive{SavepointVersion{}};
        for (int inX = 0; inX < 256; inX++)
        for (int inY = 0; inY < 256; inY++)
        {
            inArchive.Reset();
            Tile2D inTile2D{inX, inY};
            inArchive(inTile2D);
            savepoint.Write(inArchive, inX, inY, 0);
        }
        int i = 0;
        savepoint.Read([&](SavepointArchive& outArchive, int outX, int outY)
        {
            Tile2D outTile2D;
            outArchive(outTile2D);
            assert(outTile2D.X == outX);
            assert(outTile2D.Y == outY);
            i++;
        }, 0);
        assert(i == 256 * 256);
    }

    // 3d spatial tests
    {
        SavepointArchive inArchive{SavepointVersion{}};
        for (int inX = 0; inX < 32; inX++)
        for (int inY = 0; inY < 32; inY++)
        for (int inZ = 0; inZ < 32; inZ++)
        {
            inArchive.Reset();
            Tile3D inTile3D{inX, inY, inZ};
            inArchive(inTile3D);
            savepoint.Write(inArchive, inX, inY, inZ, 0);
        }
        int i = 0;
        savepoint.Read([&](SavepointArchive& outArchive, int outX, int outY, int outZ)
        {
            Tile3D outTile3D;
            outArchive(outTile3D);
            assert(outTile3D.X == outX);
            assert(outTile3D.Y == outY);
            assert(outTile3D.Z == outZ);
            i++;
        }, 0);
        assert(i == 32 * 32 * 32);
    }

    // 2d spatial replacement tests
    savepoint.Clear();
    {
        SavepointArchive inArchive{SavepointVersion{}};
        for (int inX = 0; inX < 32; inX++)
        for (int inY = 0; inY < 32; inY++)
        {
            inArchive.Reset();
            Tile2D inTile2D{5, 5};
            inArchive(inTile2D);
            savepoint.Write(inArchive, inX, inY, 0);
        }
        int i = 0;
        savepoint.Read([&](SavepointArchive& outArchive, int outX, int outY)
        {
            Tile2D outTile2D;
            outArchive(outTile2D);
            assert(outTile2D.X == 5);
            assert(outTile2D.Y == 5);
            i++;
        }, 0);
        assert(i == 32 * 32);
    }
    {
        SavepointArchive inArchive{SavepointVersion{}};
        for (int inX = 0; inX < 32; inX++)
        for (int inY = 0; inY < 32; inY++)
        {
            inArchive.Reset();
            Tile2D inTile2D{10, 10};
            inArchive(inTile2D);
            savepoint.Write(inArchive, inX, inY, 0);
        }
        int i = 0;
        savepoint.Read([&](SavepointArchive& outArchive, int outX, int outY)
        {
            Tile2D outTile2D;
            outArchive(outTile2D);
            assert(outTile2D.X == 10);
            assert(outTile2D.Y == 10);
            i++;
        }, 0);
        assert(i == 32 * 32);
    }

    savepoint.Save();
    savepoint.Close();
    return 0;
}