#pragma once

#include <imgui.h>

#include <savepoint/savepoint.hpp>

#include <array>
#include <cstdint>
#include <format>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

enum class SavepointDebuggerMode : uint8_t
{
    Singleton,
    Levels,
    Entities,
    Tiles2D,
    Tiles3D,
};

static constexpr std::array<std::string_view, 5> kSavepointDebuggerModes =
{
    "Singleton",
    "Levels",
    "Entities",
    "2D tiles",
    "3D tiles",
};

struct SavepointDebuggerTree
{
    SavepointDebuggerTree(SavepointID id, int x, int y, int z, int level, std::vector<SavepointDebugNode> nodes)
        : ID{id}
        , X{x}
        , Y{y}
        , Z{z}
        , Level{level}
        , Nodes{std::move(nodes)}
    {
    }

    void Render(SavepointDebuggerMode mode) const
    {
        static constexpr ImGuiTreeNodeFlags kLeafFlags = ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen | ImGuiTreeNodeFlags_SpanFullWidth;
        static constexpr ImGuiTreeNodeFlags kNodeFlags = ImGuiTreeNodeFlags_SpanFullWidth | ImGuiTreeNodeFlags_DefaultOpen;
        switch (mode)
        {
        case SavepointDebuggerMode::Singleton:
            ImGui::TextUnformatted("singleton");
            break;
        case SavepointDebuggerMode::Levels:
            ImGui::Text("level %d", Level);
            break;
        case SavepointDebuggerMode::Entities:
            ImGui::Text("id %u, level %d", ID.GetValue(), Level);
            break;
        case SavepointDebuggerMode::Tiles2D:
            ImGui::Text("x %d, y %d, level %d", X, Y, Level);
            break;
        case SavepointDebuggerMode::Tiles3D:
            ImGui::Text("x %d, y %d, z %d, level %d", X, Y, Z, Level);
            break;
        }
        ImGui::Separator();
        if (Nodes.empty())
        {
            ImGui::TextDisabled("Nothing was read.");
            return;
        }

        int pushed = 0;
        int collapsed = -1;
        for (const SavepointDebugNode& node : Nodes)
        {
            int depth = node.GetDepth();
            if (collapsed >= 0)
            {
                if (depth > collapsed)
                {
                    continue;
                }
                collapsed = -1;
            }
            while (pushed > depth)
            {
                ImGui::TreePop();
                pushed--;
            }
            std::string_view type = node.GetTypeName();
            if (node.GetIsLeaf())
            {
                ImGui::TreeNodeEx(&node, kLeafFlags, "%.*s = %s", int(type.size()), type.data(), node.GetValue().data());
            }
            else if (ImGui::TreeNodeEx(&node, kNodeFlags, "%.*s", int(type.size()), type.data()))
            {
                pushed++;
            }
            else
            {
                collapsed = depth;
            }
        }
        while (pushed > 0)
        {
            ImGui::TreePop();
            pushed--;
        }
    }

    std::vector<std::string> GetValues(SavepointDebuggerMode mode) const
    {
        switch (mode)
        {
        case SavepointDebuggerMode::Levels:
            return {std::format("{}", Level)};
        case SavepointDebuggerMode::Entities:
            return {std::format("{}", ID.GetValue())};
        case SavepointDebuggerMode::Tiles2D:
            return {std::format("{}", X), std::format("{}", Y)};
        case SavepointDebuggerMode::Tiles3D:
            return {std::format("{}", X), std::format("{}", Y), std::format("{}", Z)};
        default:
            return {};
        }
    }

    SavepointID ID;
    int X;
    int Y;
    int Z;
    int Level;
    std::vector<SavepointDebugNode> Nodes;
};

struct SavepointDebugger
{
    SavepointDebugger()
        : Mode{SavepointDebuggerMode::Entities}
        , CachedMode{SavepointDebuggerMode::Singleton}
        , Level{0}
        , CachedLevel{0}
        , Slice{0}
        , Selected{-1}
        , Dirty{true}
    {
    }

    void Clear()
    {
        Trees.clear();
        Selected = -1;
        CachedMode = Mode;
        CachedLevel = Level;
        Dirty = false;
    }

    void Refresh(Savepoint& savepoint)
    {
        Clear();

        std::vector<SavepointDebugNode> nodes;
        switch (Mode)
        {
        case SavepointDebuggerMode::Singleton:
            if (savepoint.ReadDebug(nodes))
            {
                Trees.emplace_back(SavepointID{}, 0, 0, 0, 0, std::move(nodes));
            }
            break;
        case SavepointDebuggerMode::Levels:
            if (savepoint.ReadDebug(nodes, Level))
            {
                Trees.emplace_back(SavepointID{}, 0, 0, 0, Level, std::move(nodes));
            }
            break;
        case SavepointDebuggerMode::Entities:
            savepoint.ReadDebug([this](const std::vector<SavepointDebugNode>& read, SavepointID id)
            {
                Trees.emplace_back(id, 0, 0, 0, Level, read);
            }, Level);
            break;
        case SavepointDebuggerMode::Tiles2D:
            savepoint.ReadDebug([this](const std::vector<SavepointDebugNode>& read, int x, int y)
            {
                Trees.emplace_back(SavepointID{}, x, y, 0, Level, read);
            }, Level);
            break;
        case SavepointDebuggerMode::Tiles3D:
            savepoint.ReadDebug([this](const std::vector<SavepointDebugNode>& read, int x, int y, int z)
            {
                Trees.emplace_back(SavepointID{}, x, y, z, Level, read);
            }, Level);
            break;
        }
    }

    std::vector<std::string_view> GetKeys() const
    {
        switch (Mode)
        {
        case SavepointDebuggerMode::Levels:
            return {"level"};
        case SavepointDebuggerMode::Entities:
            return {"id"};
        case SavepointDebuggerMode::Tiles2D:
            return {"x", "y"};
        case SavepointDebuggerMode::Tiles3D:
            return {"x", "y", "z"};
        default:
            return {};
        }
    }

    void Render(Savepoint& savepoint)
    {
        if (Dirty || Mode != CachedMode || Level != CachedLevel)
        {
            Refresh(savepoint);
        }
        if (Mode == SavepointDebuggerMode::Singleton && !Trees.empty())
        {
            Selected = 0;
        }

        if (!ImGui::BeginTable("savepoint", 3, ImGuiTableFlags_Resizable | ImGuiTableFlags_BordersInnerV))
        {
            return;
        }
        ImGui::TableSetupColumn("controls", ImGuiTableColumnFlags_WidthFixed, 240.0f);
        ImGui::TableSetupColumn("rows");
        ImGui::TableSetupColumn("contents", ImGuiTableColumnFlags_WidthFixed, 360.0f);
        ImGui::TableNextRow();

        ImGui::TableNextColumn();
        size_t mode = size_t(Mode);
        ImGui::SeparatorText("View");
        if (ImGui::BeginCombo("Mode", kSavepointDebuggerModes[mode].data()))
        {
            for (size_t i = 0; i < kSavepointDebuggerModes.size(); i++)
            {
                if (ImGui::Selectable(kSavepointDebuggerModes[i].data(), mode == i))
                {
                    Mode = SavepointDebuggerMode(i);
                }
            }
            ImGui::EndCombo();
        }
        if (Mode != SavepointDebuggerMode::Singleton)
        {
            ImGui::InputInt("Level", &Level);
        }
        if (Mode == SavepointDebuggerMode::Tiles3D && ImGui::InputInt("Z slice", &Slice))
        {
            Selected = -1;
        }
        ImGui::SeparatorText("Savepoint");
        if (ImGui::Button("Refresh"))
        {
            Dirty = true;
        }
        ImGui::Text("%zu rows", Trees.size());

        ImGui::TableNextColumn();
        ImGui::SeparatorText("Entries");
        std::vector<int> visible;
        for (int i = 0; i < static_cast<int>(Trees.size()); i++)
        {
            if (Mode != SavepointDebuggerMode::Tiles3D || Trees[i].Z == Slice)
            {
                visible.push_back(i);
            }
        }
        if (visible.empty())
        {
            ImGui::TextDisabled("Nothing to show.");
        }
        else if (Mode == SavepointDebuggerMode::Singleton)
        {
            ImGui::TextDisabled("Only one entry, shown on the right.");
        }
        else
        {
            std::vector<std::string_view> keys = GetKeys();
            std::vector<std::string> headers;
            for (int i : visible)
            {
                size_t leaf = 0;
                for (const SavepointDebugNode& node : Trees[i].Nodes)
                {
                    if (!node.GetIsLeaf())
                    {
                        continue;
                    }
                    if (leaf == headers.size())
                    {
                        headers.push_back(std::format("{} {}", node.GetTypeName(), leaf));
                    }
                    leaf++;
                }
            }

            int columns = static_cast<int>(keys.size() + headers.size());
            ImGuiTableFlags flags = ImGuiTableFlags_Borders |
                ImGuiTableFlags_RowBg |
                ImGuiTableFlags_Resizable |
                ImGuiTableFlags_ScrollX |
                ImGuiTableFlags_ScrollY;
            if (ImGui::BeginTable("rows", columns, flags))
            {
                ImGui::TableSetupScrollFreeze(static_cast<int>(keys.size()), 1);
                for (std::string_view key : keys)
                {
                    ImGui::TableSetupColumn(key.data(), ImGuiTableColumnFlags_WidthFixed, 50.0f);
                }
                for (const std::string& header : headers)
                {
                    ImGui::TableSetupColumn(header.c_str());
                }
                ImGui::TableHeadersRow();

                for (int i : visible)
                {
                    const SavepointDebuggerTree& tree = Trees[i];
                    std::vector<std::string> values = tree.GetValues(Mode);
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::PushID(i);
                    if (ImGui::Selectable(values[0].c_str(), Selected == i, ImGuiSelectableFlags_SpanAllColumns))
                    {
                        Selected = i;
                    }
                    ImGui::PopID();
                    for (size_t key = 1; key < values.size(); key++)
                    {
                        ImGui::TableNextColumn();
                        ImGui::TextUnformatted(values[key].c_str());
                    }

                    int column = static_cast<int>(values.size());
                    for (const SavepointDebugNode& node : tree.Nodes)
                    {
                        if (node.GetIsLeaf() && column < columns)
                        {
                            ImGui::TableNextColumn();
                            ImGui::TextUnformatted(node.GetValue().c_str());
                            column++;
                        }
                    }
                }
                ImGui::EndTable();
            }
        }

        ImGui::TableNextColumn();
        ImGui::SeparatorText("Contents");
        if (ImGui::BeginChild("contents"))
        {
            if (Selected < 0 || Selected >= static_cast<int>(Trees.size()))
            {
                ImGui::TextDisabled("Select a row.");
            }
            else
            {
                Trees[Selected].Render(Mode);
            }
        }
        ImGui::EndChild();

        ImGui::EndTable();
    }

    SavepointDebuggerMode Mode;
    SavepointDebuggerMode CachedMode;
    int Level;
    int CachedLevel;
    int Slice;
    int Selected;
    std::vector<SavepointDebuggerTree> Trees;
    bool Dirty;
};
