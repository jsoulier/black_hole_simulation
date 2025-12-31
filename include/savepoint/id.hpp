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
#include <limits>

/**
 * @brief Used to uniquely identify a Savepoint entry.
 * 
 * For objects that don't have unique locations, an ID class is provided to
 * ensure the object gets a unique entry. When users write the object to the
 * Savepoint, they can provide the ID alongside the object. Savepoint will use
 * (and potentially generate a new ID) to insert or update an entry.
 * 
 * @snippet examples/basic_usage.cpp basic_usage
 * @see Savepoint
 */
class SavepointID
{
public:
    /**
     * @brief Default initializes the ID to an invalid value.
     */
    constexpr SavepointID()
        : Value{std::numeric_limits<uint32_t>::max()}
    {
    }

    /**
     * @brief Check if an ID is the same as another ID.
     * 
     * @param other The other ID.
     * @return True if the IDs are the same.
     */
    constexpr bool operator==(const SavepointID other) const
    {
        return Value == other.Value;
    }

    /**
     * @brief Check if an ID is not the same as another ID.
     * 
     * @param other The other ID.
     * @return True if the IDs are not the same.
     */
    constexpr bool operator!=(const SavepointID other) const
    {
        return Value != other.Value;
    }

    /**
     * @brief Check if an ID is valid.
     * 
     * @return True if the ID is valid.
     */
    constexpr bool IsValid() const
    {
        return Value != SavepointID{}.Value;
    }

    /** @cond INTERNAL */

    void SetValue(uint32_t value)
    {
        Value = value;
    }

    uint32_t GetValue() const
    {
        return Value;
    }

    /** @endcond */

private:
    uint32_t Value;
};
