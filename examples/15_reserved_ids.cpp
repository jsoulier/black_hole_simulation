// [15_reserved_ids]
#include <savepoint/savepoint.hpp>

#include <cassert>
#include <filesystem>

static constexpr SavepointVersion kVersion{0, 0, 0};

// IDs (or levels) can be reserved
static constexpr int kPlayerID = 1001;
static constexpr int kBossID = 1002;

struct Entity
{
    int Score;

    Entity() = default;
    Entity(int score)
        : Score{score}
    {
    }

    void Visit(SavepointVisitor& visitor)
    {
        visitor(Score);
    }
};

int main()
{
    std::filesystem::remove("savepoint.sqlite3");

    Savepoint savepoint;
    SavepointStatus status = savepoint.Open(SavepointDriver::SQLite3, "savepoint.sqlite3", kVersion);
    assert(status == SavepointStatus::New);

    Entity inPlayer{10};
    Entity inBoss{20};
    savepoint.Write(inPlayer, kPlayerID);
    savepoint.Write(inBoss, kBossID);

    Entity outPlayer;
    assert(savepoint.Read(outPlayer, kPlayerID));
    assert(outPlayer.Score == inPlayer.Score);

    Entity outBoss;
    assert(savepoint.Read(outBoss, kBossID));
    assert(outBoss.Score == inBoss.Score);

    inBoss.Score = 30;
    savepoint.Write(inBoss, kBossID);
    assert(savepoint.Read(outBoss, kBossID));
    assert(outBoss.Score == inBoss.Score);
    assert(savepoint.Read(outPlayer, kPlayerID));
    assert(outPlayer.Score == inPlayer.Score);

    assert(!savepoint.Read(outBoss, 1003));

    savepoint.Clear();
    assert(!savepoint.Read(outPlayer, kPlayerID));
    assert(!savepoint.Read(outBoss, kBossID));

    savepoint.Close();
    return 0;
}
// [15_reserved_ids]
