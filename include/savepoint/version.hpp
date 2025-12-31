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

#pragma once

#include <savepoint/fwd.hpp>

#include <cstdint>
#include <format>
#include <string>

/**
 * @brief 
 * 
 */
class SavepointVersion
{
public:
    /**
     * @brief 
     * 
     */
    constexpr SavepointVersion()
        : Value{0}
    {
    }

    /**
     * @brief 
     * 
     * @param major
     * @param minor
     * @param patch
     */
    constexpr SavepointVersion(uint32_t major, uint32_t minor, uint32_t patch)
        : Value{major << 24 | minor << 16 | patch}
    {
    }

    /**
     * @brief 
     * 
     * @return 
     */
    constexpr uint32_t GetMajor() const
    {
        return (Value >> 24) & 0xFF;
    }

    /**
     * @brief 
     * 
     * @return 
     */
    constexpr uint32_t GetMinor() const
    {
        return (Value >> 16) & 0xFF;
    }

    /**
     * @brief 
     * 
     * @return 
     */
    constexpr uint32_t GetPatch() const
    {
        return Value & 0xFFFF;
    }

    /**
     * @brief 
     * 
     * @return 
     */
    std::string GetString() const
    {
        return std::format("{}.{}.{}", GetMajor(), GetMinor(), GetPatch());
    }

    /**
     * @brief 
     * 
     * @param other
     * @return
     */
    constexpr auto operator<=>(const SavepointVersion other) const
    {
        return Value <=> other.Value;
    }

private:
    uint32_t Value;
};

static constexpr SavepointVersion kSavepointVersion{0, 0, 0};
