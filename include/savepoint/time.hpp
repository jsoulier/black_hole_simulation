#pragma once

#include <savepoint/fwd.hpp>

#include <chrono>
#include <cstdint>
#include <format>
#include <string>

/** @cond INTERNAL */

template<typename ClockT, typename DurationT>
class SavepointTimeImpl;

/** @endcond */

/** The default time specialization. */
using SavepointTime = SavepointTimeImpl<std::chrono::system_clock, std::chrono::seconds>;

/**
 * @brief Used to represent a time.
 * 
 * @tparam ClockT The clock type.
 * @tparam DurationT The duration type.
 */
template<typename ClockT, typename DurationT>
class SavepointTimeImpl
{
public:
    /** @brief The underlying representation of the duration. */
    using RepT = typename DurationT::rep;

    /**
     * @brief Create a time with an optional value.
     * 
     * @tparam T The type of the clock.
     * @param value The time value.
     */
    template<typename T = ClockT>
    // Waiting on MacOS's clang to support is_clock_v
    // requires std::chrono::is_clock_v<T>
    SavepointTimeImpl(T::time_point value = T::now())
        : Value{std::chrono::duration_cast<DurationT>(value.time_since_epoch()).count()}
    {
    }

    /**
     * @brief Get the time as a string.
     * 
     * @return The time as a string.
     */
    std::string GetString() const
    {
        // https://en.cppreference.com/w/cpp/chrono/duration/formatter.html
        typename ClockT::time_point value = ClockT::time_point(DurationT(Value));
        value = std::chrono::floor<DurationT>(value);
        std::string string = std::format("{:%F %T}", value);
        // For some reason there are trailing zeroes (at least on MSVC)
        auto position = string.rfind('.');
        if (position != std::string::npos)
        {
            return string.substr(0, position);
        }
        else
        {
            return string;
        }
    }

    /**
     * @brief Compare the time to another time.
     * 
     * @param other The other time.
     * @return True if the comparison evaluated to true.
     */
    auto operator<=>(const SavepointTimeImpl& other) const = default;

private:
    RepT Value;
};
