/*
 * Expected:
 * Tried to read into non-empty range
 * Fixed range is too small: %d < %d (x3)
 * Tried to read into non-empty range (x2)
 */

#include <savepoint/savepoint.hpp>

#include <array>
#include <cassert>
#include <filesystem>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

static const std::string kFileName = "savepoint.sqlite3";

static constexpr SavepointVersion kVersion1{0, 0, 1};
static constexpr SavepointVersion kVersion2{0, 1, 0};
static constexpr SavepointVersion kVersion3{0, 2, 0};
static constexpr SavepointVersion kVersion4{1, 1, 0};

struct Header
{
    int Major;
    int Minor;
    int Patch;
};

struct ItemV2
{
    int Durability = 1;
    int Damage = 2;

    void Visit(SavepointVisitor& visitor)
    {
        visitor(Durability);
        visitor(Damage);
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

    void Visit(SavepointVisitor& visitor)
    {
        visitor(Durability);
        visitor(Damage);
        visitor(Rarity, kVersion4);
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

    void Visit(SavepointVisitor& visitor)
    {
        visitor(X);
        visitor(Y);
        visitor(Health);
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

    void Visit(SavepointVisitor& visitor)
    {
        visitor(Strength, kVersion2);
        visitor(X);
        visitor(Y);
        visitor(Item, kVersion2);
        visitor(Health);
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

    void Visit(SavepointVisitor& visitor)
    {
        visitor(Strength, kVersion2);
        visitor(X);
        visitor(Y);
        visitor(Item, kVersion2);
        visitor(Health);
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

    void Visit(SavepointVisitor& visitor)
    {
        visitor(Strength, kVersion2);
        visitor(X);
        visitor(Y);
        visitor(Z, kVersion4);
        visitor(Item, kVersion2);
        visitor(Health);
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

    void Visit(SavepointVisitor& visitor)
    {
        EntityV1::Visit(visitor);
        visitor(Speed);
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

    void Visit(SavepointVisitor& visitor)
    {
        EntityV2::Visit(visitor);
        visitor(Speed);
        visitor(VelocityX, kVersion2);
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

    void Visit(SavepointVisitor& visitor)
    {
        EntityV3::Visit(visitor);
        visitor(Speed);
        visitor(VelocityX, kVersion2);
        visitor(VelocityY, kVersion3);
        visitor(VelocityZ, kVersion3);
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

    void Visit(SavepointVisitor& visitor)
    {
        EntityV4::Visit(visitor);
        visitor(Speed);
        visitor(VelocityX, kVersion2);
        visitor(VelocityY, kVersion3);
        visitor(VelocityZ, kVersion3);
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

    void Visit(SavepointVisitor& visitor)
    {
        visitor(X);
        visitor(Y);
        visitor(Z);
    }
};

struct Vector
{
    std::vector<int> Data = {1, 5, 3, 7};

    void Visit(SavepointVisitor& visitor)
    {
        visitor(Data);
    }

    bool operator==(const Vector& other) const
    {
        return Data == other.Data;
    }
};

struct ArrayV1
{
    std::array<int, 2> Data = {1, 3};

    void Visit(SavepointVisitor& visitor)
    {
        visitor(Data);
    }

    bool operator==(const ArrayV1& other) const
    {
        return Data == other.Data;
    }
};

struct ArrayV2
{
    std::array<int, 3> Data = {1, 3, 2};
    int Sentinel = 4;

    void Visit(SavepointVisitor& visitor)
    {
        visitor(Data);
        visitor(Sentinel, kVersion2);
    }

    bool operator==(const ArrayV1& other) const
    {
        return Data[0] == other.Data[0] &&
            Data[1] == other.Data[1] &&
            Data[2] == 2 &&
            Sentinel == 4;
    }

    bool operator==(const ArrayV2& other) const
    {
        return
            Data == other.Data &&
            Sentinel == other.Sentinel;
    }
};

struct ArrayV3
{
    std::array<int, 1> Data = {1};
    int Sentinel = 4;

    void Visit(SavepointVisitor& visitor)
    {
        visitor(Data);
        visitor(Sentinel, kVersion2);
    }

    bool operator==(const ArrayV1& other) const
    {
        return
            Data[0] == other.Data[0] &&
            Sentinel == 4;
    }

    bool operator==(const ArrayV2& other) const
    {
        return Data[0] == other.Data[0] &&
            Sentinel == other.Sentinel;
    }

    bool operator==(const ArrayV3& other) const
    {
        return Data[0] == other.Data[0] &&
            Sentinel == other.Sentinel;
    }
};

struct Map
{
    std::map<int, int> Data = {{1, 2}, {2, 3}};

    void Visit(SavepointVisitor& visitor)
    {
        visitor(Data);
    }

    bool operator==(const Map& other) const
    {
        return Data == other.Data;
    }
};

struct Set
{
    std::set<int> Data = {1, 2, 3};

    void Visit(SavepointVisitor& visitor)
    {
        visitor(Data);
    }

    bool operator==(const Set& other) const
    {
        return Data == other.Data;
    }
};

struct UniquePtr
{
    std::unique_ptr<int> Data = std::make_unique<int>(1);

    void Visit(SavepointVisitor& visitor)
    {
        visitor(Data);
    }

    bool operator==(const UniquePtr& other) const
    {
        return *Data == *(other.Data);
    }
};

struct SharedPtr
{
    std::shared_ptr<int> Data = std::make_shared<int>(1);

    void Visit(SavepointVisitor& visitor)
    {
        visitor(Data);
    }

    bool operator==(const SharedPtr& other) const
    {
        return *Data == *(other.Data);
    }
};

struct SharedPtrVector
{
    std::vector<std::shared_ptr<int>> Data;

    void Visit(SavepointVisitor& visitor)
    {
        if (visitor.IsWriting())
        {
            Data.push_back(std::make_shared<int>(1));
            Data.push_back(std::make_shared<int>(3));
            Data.push_back(std::make_shared<int>(2));
        }
        visitor(Data);
    }

    bool operator==(const SharedPtrVector& other) const
    {
        return *Data[0] == *(other.Data[0]) &&
            *Data[1] == *(other.Data[1]) &&
            *Data[2] == *(other.Data[2]);
    }
};

struct NullPtr
{
    std::shared_ptr<int> Data;

    void Visit(SavepointVisitor& visitor)
    {
        visitor(Data);
    }

    bool operator==(const NullPtr& other) const
    {
        return !Data && !other.Data;
    }
};

struct BaseEntity : public SavepointBase
{
    SavepointID ID;
    int X;
    int Y;

    void Visit(SavepointVisitor& visitor) override
    {
        visitor(X);
        visitor(Y);
    }

    bool operator==(const BaseEntity& other) const
    {
        return X == other.X &&
            Y == other.Y;
    }
};

struct BaseMob : BaseEntity
{
    int Health;
    int Damage;

    void Visit(SavepointVisitor& visitor) override
    {
        BaseEntity::Visit(visitor);
        visitor(Health);
        visitor(Damage);
    }

    bool operator==(const BaseMob& other) const
    {
        return BaseEntity::operator==(other) &&
            Health == other.Health &&
            Damage == other.Damage;
    }
};

struct DerivedItem : BaseEntity
{
    SAVEPOINT_DERIVED(DerivedItem)
};

struct DerivedZombie : BaseMob
{
    SAVEPOINT_DERIVED(DerivedZombie)
};

struct DerivedSkeleton : BaseMob
{
    SAVEPOINT_DERIVED(DerivedSkeleton)
};

struct DerivedSpider : BaseMob
{
    SAVEPOINT_DERIVED(DerivedSpider)

    int Eyes = 8;
    int Legs = 8;

    void Visit(SavepointVisitor& visitor) override
    {
        BaseMob::Visit(visitor);
        visitor(Eyes);
        visitor(Legs);
    }

    bool operator==(const DerivedSpider& other) const
    {
        return BaseMob::operator==(other) &&
            Eyes == other.Eyes &&
            Legs == other.Legs;
    }
};

template<typename InT, typename OutT>
static void TestReadWrite(SavepointDriver driver, SavepointVersion inVersion)
{
    std::filesystem::remove(kFileName);
    std::filesystem::remove(kFileName + "-journal");
    Savepoint savepoint;
    SavepointStatus status = savepoint.Open(driver, kFileName, inVersion);
    assert(status != SavepointStatus::Failed);
    SavepointID inID;
    InT inEntity;
    savepoint.Write(inEntity, inID, 0);
    int i = 0;
    savepoint.Read<OutT>([&](OutT& outEntity, SavepointID outID)
    {
        assert(outEntity == inEntity);
        assert(outID == inID);
        i++;
    }, 0);
    assert(i == 1);
    savepoint.Close();
}

static void Test(SavepointDriver driver)
{
    std::filesystem::remove(kFileName);
    std::filesystem::remove(kFileName + "-journal");
    Savepoint savepoint;
    SavepointStatus status;
    status = savepoint.Open(driver, kFileName, SavepointVersion{});
    assert(status == SavepointStatus::New);
    savepoint.Close();
    status = savepoint.Open(driver, kFileName, SavepointVersion{});
    assert(status == SavepointStatus::New);
    savepoint.Save();
    savepoint.Close();
    status = savepoint.Open(driver, kFileName, SavepointVersion{});
    assert(status == SavepointStatus::Existing);
    {
        Header header;
        header.Major = 1;
        header.Minor = 2;
        header.Patch = 3;
        savepoint.Write(header);
        savepoint.Read<Header>([&](Header& header)
        {
            assert(header.Major = 1);
            assert(header.Minor = 2);
            assert(header.Patch = 3);
        });
        header.Major = 4;
        header.Minor = 5;
        header.Patch = 6;
        savepoint.Write(header);
        savepoint.Read<Header>([&](Header& header)
        {
            assert(header.Major = 4);
            assert(header.Minor = 5);
            assert(header.Patch = 6);
        });
    }
    {
        for (int inX = 0; inX < 256; inX++)
        for (int inY = 0; inY < 256; inY++)
        {
            Tile2D inTile2D{inX, inY};
            savepoint.Write(inTile2D, inX, inY, 0);
        }
        int i = 0;
        savepoint.Read<Tile2D>([&](Tile2D& outTile2D, int outX, int outY)
        {
            assert(outTile2D.X == outX);
            assert(outTile2D.Y == outY);
            i++;
        }, 0);
        assert(i == 256 * 256);
    }
    {
        for (int inX = 0; inX < 32; inX++)
        for (int inY = 0; inY < 32; inY++)
        for (int inZ = 0; inZ < 32; inZ++)
        {
            Tile3D inTile3D{inX, inY, inZ};
            savepoint.Write(inTile3D, inX, inY, inZ, 0);
        }
        int i = 0;
        savepoint.Read<Tile3D>([&](Tile3D& outTile3D, int outX, int outY, int outZ)
        {
            assert(outTile3D.X == outX);
            assert(outTile3D.Y == outY);
            assert(outTile3D.Z == outZ);
            i++;
        }, 0);
        assert(i == 32 * 32 * 32);
    }
    savepoint.Clear();
    {
        for (int inX = 0; inX < 32; inX++)
        for (int inY = 0; inY < 32; inY++)
        {
            Tile2D inTile2D{5, 5};
            savepoint.Write(inTile2D, inX, inY, 0);
        }
        int i = 0;
        savepoint.Read<Tile2D>([&](Tile2D& outTile2D, int outX, int outY)
        {
            assert(outTile2D.X == 5);
            assert(outTile2D.Y == 5);
            i++;
        }, 0);
        assert(i == 32 * 32);
    }
    {
        for (int inX = 0; inX < 32; inX++)
        for (int inY = 0; inY < 32; inY++)
        {
            Tile2D inTile2D{10, 10};
            savepoint.Write(inTile2D, inX, inY, 0);
        }
        int i = 0;
        savepoint.Read<Tile2D>([&](Tile2D& outTile2D, int outX, int outY)
        {
            assert(outTile2D.X == 10);
            assert(outTile2D.Y == 10);
            i++;
        }, 0);
        assert(i == 32 * 32);
    }
    savepoint.Clear();
    {
        std::shared_ptr<DerivedItem> inItem = std::make_shared<DerivedItem>();
        std::shared_ptr<DerivedZombie> inZombie = std::make_shared<DerivedZombie>();
        std::shared_ptr<DerivedSkeleton> inSkeleton = std::make_shared<DerivedSkeleton>();
        std::shared_ptr<DerivedSpider> inSpider = std::make_shared<DerivedSpider>();
        savepoint.Write(inItem.get(), inItem->ID, 0);
        savepoint.Write(inZombie.get(), inZombie->ID, 0);
        savepoint.Write(inSkeleton.get(), inSkeleton->ID, 0);
        savepoint.Write(inSpider.get(), inSpider->ID, 0);
        int i = 0;
        savepoint.Read([&](SavepointBase* base, SavepointID outID)
        {
            if (outID == inItem->ID)
            {
                DerivedItem* outItem = dynamic_cast<DerivedItem*>(base);
                assert(outItem);
                assert(*outItem == *inItem);
            }
            else if (outID == inZombie->ID)
            {
                DerivedZombie* outZombie = dynamic_cast<DerivedZombie*>(base);
                assert(outZombie);
                assert(*outZombie == *inZombie);
            }
            else if (outID == inSkeleton->ID)
            {
                DerivedSkeleton* outSkeleton = dynamic_cast<DerivedSkeleton*>(base);
                assert(outSkeleton);
                assert(*outSkeleton == *inSkeleton);
            }
            else if (outID == inSpider->ID)
            {
                DerivedSpider* outSpider = dynamic_cast<DerivedSpider*>(base);
                assert(outSpider);
                assert(*outSpider == *inSpider);
            }
            else
            {
                assert(false);
            }
            i++;
        }, 0);
        assert(i == 4);
    }
    savepoint.Close();
    TestReadWrite<EntityV1, EntityV1>(driver, kVersion1);
    TestReadWrite<EntityV1, EntityV2>(driver, kVersion1);
    TestReadWrite<EntityV2, EntityV2>(driver, kVersion2);
    TestReadWrite<EntityV1, EntityV3>(driver, kVersion1);
    TestReadWrite<EntityV2, EntityV3>(driver, kVersion2);
    TestReadWrite<EntityV3, EntityV3>(driver, kVersion3);
    TestReadWrite<EntityV1, EntityV4>(driver, kVersion1);
    TestReadWrite<EntityV2, EntityV4>(driver, kVersion2);
    TestReadWrite<EntityV3, EntityV4>(driver, kVersion3);
    TestReadWrite<EntityV4, EntityV4>(driver, kVersion4);
    TestReadWrite<ZombieV1, ZombieV1>(driver, kVersion1);
    TestReadWrite<ZombieV1, ZombieV2>(driver, kVersion1);
    TestReadWrite<ZombieV2, ZombieV2>(driver, kVersion2);
    TestReadWrite<ZombieV1, ZombieV3>(driver, kVersion1);
    TestReadWrite<ZombieV2, ZombieV3>(driver, kVersion2);
    TestReadWrite<ZombieV3, ZombieV3>(driver, kVersion3);
    TestReadWrite<ZombieV1, ZombieV4>(driver, kVersion1);
    TestReadWrite<ZombieV2, ZombieV4>(driver, kVersion2);
    TestReadWrite<ZombieV3, ZombieV4>(driver, kVersion3);
    TestReadWrite<ZombieV4, ZombieV4>(driver, kVersion4);
    TestReadWrite<Vector, Vector>(driver, kVersion1);
    TestReadWrite<ArrayV1, ArrayV1>(driver, kVersion1);
    TestReadWrite<ArrayV1, ArrayV2>(driver, kVersion1);
    TestReadWrite<ArrayV1, ArrayV3>(driver, kVersion1);
    TestReadWrite<ArrayV2, ArrayV2>(driver, kVersion1);
    TestReadWrite<ArrayV2, ArrayV3>(driver, kVersion1);
    TestReadWrite<ArrayV3, ArrayV3>(driver, kVersion1);
    TestReadWrite<ArrayV2, ArrayV2>(driver, kVersion2);
    TestReadWrite<ArrayV2, ArrayV3>(driver, kVersion2);
    TestReadWrite<ArrayV3, ArrayV3>(driver, kVersion2);
    TestReadWrite<Map, Map>(driver, kVersion1);
    TestReadWrite<Set, Set>(driver, kVersion1);
    TestReadWrite<UniquePtr, UniquePtr>(driver, kVersion1);
    TestReadWrite<SharedPtr, SharedPtr>(driver, kVersion1);
    TestReadWrite<SharedPtrVector, SharedPtrVector>(driver, kVersion1);
    TestReadWrite<NullPtr, NullPtr>(driver, kVersion1);
}

int main()
{
    Test(SavepointDriver::Sqlite3);
    return 0;
}
