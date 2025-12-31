/*
 * This is free and unencumbered software released into the public domain.
 * 
 * Anyone is free to copy, modify, publish, use, compile, sell, or
 * distribute this software, either in source code form or as a compiled
 * binary, for any purpose, commercial or non-commercial, and by any
 * means.
 * 
 * In jurisdictions that recognize copyright laws, the author or authors
 * of this software dedicate any and all copyright interest in the
 * software to the public domain. We make this dedication for the benefit
 * of the public at large and to the detriment of our heirs and
 * successors. We intend this dedication to be an overt act of
 * relinquishment in perpetuity of all present and future rights to this
 * software under copyright law.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 * 
 * For more information, please refer to <https://unlicense.org>
 */

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
