#pragma once

#include <savepoint/fwd.hpp>

#include <cstdint>
#include <limits>

/** @cond INTERNAL */

struct SavepointID
{
    constexpr SavepointID()
        : Value{std::numeric_limits<uint32_t>::max()}
    {
    }

    constexpr SavepointID(uint32_t value)
        : Value{value}
    {
    }

    constexpr bool IsValid() const
    {
        return Value != SavepointID{}.Value;
    }

    uint32_t Value;
};

/** @endcond */

/**
 * @brief Used to uniquely identify a Savepoint entry.
 * 
 * For objects that don't have unique locations, a base class is provided to
 * ensure the object gets a unique entry. When users write their object to the
 * Savepoint, Savepoint will use the base class to insert or update an entry.
 * Users should not modify the base class themselves.
 * 
 * @snippet examples/basic_usage.cpp basic_usage
 * @see Savepoint
 */

/**
 * @brief 
 */
class SavepointEntity
{
    friend class Savepoint;

private:
    SavepointID ID;
};
