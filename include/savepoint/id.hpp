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

private:
    uint32_t Value;
};
