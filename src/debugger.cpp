#include <savepoint/imgui.hpp>
#include <savepoint/savepoint.hpp>

#include <imgui.h>
#include <imgui_impl_sdl3.h>
#include <imgui_impl_sdlrenderer3.h>

#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>

#include <array>
#include <cstdint>
#include <format>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <vector>

static constexpr SavepointVersion kVersion{1, 0, 0};
static constexpr const char* kPath = "debugger.sqlite3";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

enum class Faction : uint8_t
{
    Neutral,
    Friendly,
    Hostile,
};

struct Gear
{
    int Durability = 100;
    float Weight = 1.5f;

    void Visit(SavepointVisitor& visitor)
    {
        visitor(Durability);
        visitor(Weight);
    }

    void Randomize(std::mt19937& rng)
    {
        Durability = std::uniform_int_distribution<int>{0, 100}(rng);
        Weight = std::uniform_real_distribution<float>{0.5f, 20.0f}(rng);
    }
};

struct WorldHeader
{
    std::string Seed = "seed";
    int Ticks = 0;
    std::mt19937 Rng;

    void Visit(SavepointVisitor& visitor)
    {
        visitor(Seed);
        visitor(Ticks);
        visitor(Rng);
    }

    void Randomize(std::mt19937& rng)
    {
        Seed = std::format("seed-{}", std::uniform_int_distribution<int>{1000, 9999}(rng));
        Ticks = std::uniform_int_distribution<int>{0, 100000}(rng);
        Rng = rng;
    }
};

struct LevelInfo
{
    std::string Name = "level";
    int Difficulty = 1;
    bool Visited = false;

    void Visit(SavepointVisitor& visitor)
    {
        visitor(Name);
        visitor(Difficulty);
        visitor(Visited);
    }

    void Randomize(std::mt19937& rng)
    {
        static constexpr std::array<const char*, 4> kNames = {"caves", "surface", "ruins", "depths"};
        Name = kNames[std::uniform_int_distribution<size_t>{0, kNames.size() - 1}(rng)];
        Difficulty = std::uniform_int_distribution<int>{1, 10}(rng);
        Visited = std::uniform_int_distribution<int>{0, 1}(rng) != 0;
    }
};

struct Player : SavepointEntity
{
    int Health = 100;
    float X = 0.0f;
    float Y = 0.0f;
    std::string Name = "player";
    Faction Team = Faction::Friendly;
    Gear Equipped;
    std::vector<int> Inventory;

    void Visit(SavepointVisitor& visitor)
    {
        visitor(Health);
        visitor(X);
        visitor(Y);
        visitor(Name);
        visitor(Team);
        visitor(Equipped);
        visitor(Inventory);
    }

    void Randomize(std::mt19937& rng)
    {
        static constexpr std::array<const char*, 4> kNames = {"alice", "bob", "carol", "dave"};
        Health = std::uniform_int_distribution<int>{1, 100}(rng);
        X = std::uniform_real_distribution<float>{-64.0f, 64.0f}(rng);
        Y = std::uniform_real_distribution<float>{-64.0f, 64.0f}(rng);
        Name = kNames[std::uniform_int_distribution<size_t>{0, kNames.size() - 1}(rng)];
        Team = static_cast<Faction>(std::uniform_int_distribution<int>{0, 2}(rng));
        Equipped.Randomize(rng);
        Inventory.clear();
        int count = std::uniform_int_distribution<int>{0, 3}(rng);
        for (int i = 0; i < count; i++)
        {
            Inventory.push_back(std::uniform_int_distribution<int>{1, 99}(rng));
        }
    }
};

enum class Terrain : uint8_t
{
    Empty,
    Grass,
    Stone,
    Water,
};

struct Floor
{
    Terrain Type = Terrain::Empty;
    int Variant = 0;

    void Visit(SavepointVisitor& visitor)
    {
        visitor(Type);
        visitor(Variant);
    }

    void Randomize(std::mt19937& rng)
    {
        Type = static_cast<Terrain>(std::uniform_int_distribution<int>{0, 3}(rng));
        Variant = std::uniform_int_distribution<int>{0, 5}(rng);
    }
};

struct Voxel
{
    uint8_t Material = 0;
    float Density = 0.0f;

    void Visit(SavepointVisitor& visitor)
    {
        visitor(Material);
        visitor(Density);
    }

    void Randomize(std::mt19937& rng)
    {
        Material = static_cast<uint8_t>(std::uniform_int_distribution<int>{0, 7}(rng));
        Density = std::uniform_real_distribution<float>{0.0f, 1.0f}(rng);
    }
};

//! [debugger_types]
SAVEPOINT_TYPE(WorldHeader)
SAVEPOINT_TYPE(LevelInfo)
SAVEPOINT_TYPE(Player)
SAVEPOINT_TYPE(Floor)
SAVEPOINT_TYPE(Voxel)
//! [debugger_types]

// ---------------------------------------------------------------------------
// Writing, which is the only part that knows the types
// ---------------------------------------------------------------------------

static constexpr int kSide = 8;

static void WriteDemo(Savepoint& savepoint, int level, std::mt19937& rng)
{
    WorldHeader header;
    header.Randomize(rng);
    savepoint.Write(header);

    for (int i = 0; i < 3; i++)
    {
        LevelInfo info;
        info.Randomize(rng);
        savepoint.Write(info, level + i);
    }
    for (int i = 0; i < 8; i++)
    {
        Player player;
        player.Randomize(rng);
        savepoint.Write(player, level);
    }
    for (int y = 0; y < kSide; y++)
    {
        for (int x = 0; x < kSide; x++)
        {
            Floor floor;
            floor.Randomize(rng);
            savepoint.Write(floor, x, y, level);
            for (int z = 0; z < 3; z++)
            {
                Voxel voxel;
                voxel.Randomize(rng);
                savepoint.Write(voxel, x, y, z, level);
            }
        }
    }
    savepoint.Save();
}

static void DrawWrite(Savepoint& savepoint, SavepointDebugger& debugger, std::mt19937& rng)
{
    ImGui::SeparatorText("Demo data");
    if (ImGui::Button("Write"))
    {
        WriteDemo(savepoint, debugger.Level, rng);
        debugger.Dirty = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("Clear"))
    {
        savepoint.Clear();
        savepoint.Save();
        debugger.Dirty = true;
    }
}

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

int main(int argc, char** argv)
{
    SavepointSetLogFunction([](const std::string_view& string)
    {
        SDL_Log("%.*s", static_cast<int>(string.size()), string.data());
    });

    if (!SDL_Init(SDL_INIT_VIDEO))
    {
        SDL_Log("SDL_Init: %s", SDL_GetError());
        return 1;
    }
    SDL_Window* window = SDL_CreateWindow("Savepoint Debugger", 1280, 800,
        SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY);
    if (!window)
    {
        SDL_Log("SDL_CreateWindow: %s", SDL_GetError());
        return 1;
    }
    SDL_Renderer* renderer = SDL_CreateRenderer(window, nullptr);
    if (!renderer)
    {
        SDL_Log("SDL_CreateRenderer: %s", SDL_GetError());
        return 1;
    }
    SDL_SetRenderVSync(renderer, 1);
    SDL_ShowWindow(window);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    ImGui::StyleColorsDark();
    ImGui_ImplSDL3_InitForSDLRenderer(window, renderer);
    ImGui_ImplSDLRenderer3_Init(renderer);

    const char* path = argc > 1 ? argv[1] : kPath;
    Savepoint savepoint;
    if (savepoint.Open(SavepointDriver::SQLite3, path, kVersion) == SavepointStatus::Failed)
    {
        SDL_Log("Failed to open %s", path);
        return 1;
    }

    SavepointDebugger debugger;
    std::mt19937 rng{std::random_device{}()};

    bool running = true;
    while (running)
    {
        SDL_Event event;
        while (SDL_PollEvent(&event))
        {
            ImGui_ImplSDL3_ProcessEvent(&event);
            if (event.type == SDL_EVENT_QUIT)
            {
                running = false;
            }
            if (event.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED &&
                event.window.windowID == SDL_GetWindowID(window))
            {
                running = false;
            }
        }
        if (SDL_GetWindowFlags(window) & SDL_WINDOW_MINIMIZED)
        {
            SDL_Delay(10);
            continue;
        }

        ImGui_ImplSDLRenderer3_NewFrame();
        ImGui_ImplSDL3_NewFrame();
        ImGui::NewFrame();

        const ImGuiViewport* viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(viewport->WorkPos);
        ImGui::SetNextWindowSize(viewport->WorkSize);
        ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration |
            ImGuiWindowFlags_NoMove |
            ImGuiWindowFlags_NoBringToFrontOnFocus;
        if (ImGui::Begin(path, nullptr, flags))
        {
            DrawWrite(savepoint, debugger, rng);
            debugger.Render(savepoint);
        }
        ImGui::End();

        ImGui::Render();
        SDL_SetRenderDrawColorFloat(renderer, 0.09f, 0.09f, 0.11f, 1.0f);
        SDL_RenderClear(renderer);
        ImGui_ImplSDLRenderer3_RenderDrawData(ImGui::GetDrawData(), renderer);
        SDL_RenderPresent(renderer);
    }

    savepoint.Close();
    ImGui_ImplSDLRenderer3_Shutdown();
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
