#pragma once

#include <savepoint/fwd.hpp>
#include <savepoint/visitor.hpp>

#include <chrono>
#include <compare>
#include <cstdint>
#include <format>
#include <string>

/**
 * @brief Used to represent a time.
 */
class SavepointTime
{
public:
    /** @brief The type of clock used. */
    using ClockT = std::chrono::system_clock;

    /** @brief The representation for the clock duration. */
    using DurationT = std::chrono::seconds;

    /**
     * @brief Create a time with a specific value.
     * 
     * @param value The time value.
     */
    explicit constexpr SavepointTime(uint64_t value = 0)
        : Value{value}
        , Clock{false}
    {
    }

    /**
     * @brief Create a real date and time with a specific value.
     * 
     * @tparam T The type of the clock.
     * @param value The time value.
     * 
     * @return The real date and time.
     */
    template<typename T = ClockT> requires std::chrono::is_clock_v<T>
    static SavepointTime GetClock(T::time_point value = T::now())
    {
        SavepointTime time;
        time.Value = std::chrono::duration_cast<DurationT>(value.time_since_epoch()).count();
        time.Clock = true;
        return time;
    }

    /**
     * @brief The visitor implementation.
     * 
     * @param visitor The visitor.
     */
    void Visit(SavepointVisitor& visitor)
    {
        visitor(Value);
        visitor(Clock);
    }

    /**
     * @brief Get the value.
     * 
     * @return The value.
     */
    constexpr uint64_t GetValue() const 
    {
        return Value;
    }

    /**
     * @brief Check if the time is based on a world clock.
     * 
     * @return If the time is based on a world clock.
     */
    constexpr bool IsClock() const
    {
        return Clock;
    }

    /**
     * @brief Get the time as a string.
     * 
     * @return The time as a string.
     */
    std::string GetString() const
    {
        if (IsClock())
        {
            // https://en.cppreference.com/w/cpp/chrono/duration/formatter.html
            ClockT::time_point value = ClockT::time_point(DurationT(Value));
            value = std::chrono::floor<DurationT>(value);
            std::string string = std::format("{:%F %T}", value);
            // For some reason there are trailing zeroes (at least on MSVC)
            auto position = string.rfind('.');
            if (position != std::string::npos)
            {
                string.resize(position);
            }
            return string;
        }
        else
        {
            return std::to_string(Value);
        }
    }

    /**
     * @brief Compare the time to another time. If the types are different, returns less.
     * 
     * @param other The other time.
     * @return True if the comparison evaluated to true.
     */
    constexpr auto operator<=>(const SavepointTime& other) const
    {
        if (IsClock() == other.IsClock())
        {
            return Value <=> other.Value;
        }
        else
        {
            return std::strong_ordering::less;
        }
    }

private:
    uint64_t Value;
    bool Clock;
};
