#include <savepoint/base.hpp>
#include <savepoint/visitor.hpp>

#include <cstddef>
#include <format>
#include <functional>
#include <string>
#include <string_view>
#include <unordered_map>

struct Hash
{
    using is_transparent = void;

    size_t operator()(const std::string_view& string) const
    {
        return std::hash<std::string_view>{}(string);
    }

    size_t operator()(const std::string& string) const
    {
        return std::hash<std::string_view>{}(string);
    }
};

using DerivedFunctions = std::unordered_map<std::string, SavepointDerivedFunction, Hash, std::equal_to<>>;

static DerivedFunctions& GetDerivedFunctions()
{
    // Required because of SIOF
    static DerivedFunctions functions;
    return functions;
}

void SavepointAddDerivedFunction(const std::string_view& string, const SavepointDerivedFunction function)
{
    GetDerivedFunctions().emplace(string, function);
}

bool SavepointWriteDerived(SavepointBase* base, SavepointVisitor& visitor)
{
    if (base)
    {
        std::string_view string = base->SavepointDerivedGetString();
        visitor(string);
        visitor(*base);
        return true;
    }
    else
    {
        SavepointLog("Tried to write null base");
        return false;
    }
}

SavepointBase* SavepointReadDerived(SavepointVisitor& visitor)
{
    std::string string;
    visitor(string);
    auto it = GetDerivedFunctions().find(string);
    if (it == GetDerivedFunctions().end())
    {
        SavepointLog(std::format("Failed to find base string: {}", string));
        visitor.Fail();
        return nullptr;
    }
    SavepointBase* base = it->second();
    if (!base)
    {
        SavepointLog(std::format("Failed to allocate base: {}", string));
        visitor.Fail();
        return nullptr;
    }
    visitor(*base);
    return base;
}
