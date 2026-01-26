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

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <format>
#include <functional>
#include <iterator>
#include <memory>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

/**
 * @brief The log function signature.
 * 
 * @param string The log message.
 */
using SavepointLogFunction = std::function<void(const std::string_view& string)>;

/**
 * @brief Set the log function used by SavepointLog. Defaults to stderr.
 * 
 * @param function The log function.
 */
void SavepointSetLogFunction(const SavepointLogFunction& function);

/** @cond INTERNAL */

void SavepointLog(const std::string_view& string);

/** @endcond */

/**
 * @brief Used to specify a Savepoint version.
 * 
 * For versioning members and performing quick comparisons, a version wrapper
 * is provided. The version consists of a major, minor, and patch version (with
 * decreasing significance). Versions are packed into a u32 so comparisons are
 * cheap. Versions can also be assigned and compared at compile time.
 * 
 * @snippet examples/basic_upgraded.cpp basic_upgraded
 * @see Savepoint
 */
class SavepointVersion
{
public:
    /**
     * @brief Default initializes the version to 0.0.0.
     */
    constexpr SavepointVersion()
        : Value{0}
    {
    }

    /**
     * @brief Initialize the version to major.minor.patch.
     * 
     * @param major The major version.
     * @param minor The minor version.
     * @param patch The patch version.
     */
    constexpr SavepointVersion(uint32_t major, uint32_t minor, uint32_t patch)
        : Value{major << 24 | minor << 16 | patch}
    {
    }

    /**
     * @brief Get the major version.
     * 
     * @return The major version.
     */
    constexpr uint32_t GetMajor() const
    {
        return (Value >> 24) & 0xFF;
    }

    /**
     * @brief Get the minor version.
     * 
     * @return The minor version.
     */
    constexpr uint32_t GetMinor() const
    {
        return (Value >> 16) & 0xFF;
    }

    /**
     * @brief Get the patch version.
     * 
     * @return The patch version.
     */
    constexpr uint32_t GetPatch() const
    {
        return Value & 0xFFFF;
    }

    /**
     * @brief Get the version as a string in the format major.minor.patch.
     * 
     * @return The version as a string.
     */
    std::string GetString() const
    {
        return std::format("{}.{}.{}", GetMajor(), GetMinor(), GetPatch());
    }

    /**
     * @brief Compare the version to another version.
     * 
     * @param other The other version.
     * @return True if the comparison evaluated to true.
     */
    constexpr auto operator<=>(const SavepointVersion& other) const = default;

private:
    uint32_t Value;
};

/**
 * @brief The current savepoint version.
 */
static constexpr SavepointVersion kSavepointVersion{0, 0, 0};

/**
 * @brief The default time specialization.
 */
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
    /** @brief The time point type used by the clock type. */
    using TimePointT = typename ClockT::time_point;

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
        TimePointT value{DurationT(Value)};
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
class SavepointEntity
{
    friend class Savepoint;

private:
    SavepointID ID;
};

/**
 * @brief Serves as the base class for the user's base class.
 * 
 * To support polymorphic types, Savepoint offers a base class you can use.
 * Savepoint checks if visited objects inherit from SavepointBase and if so, 
 * serializes the object alongside its type information. When Savepoint reads
 * the type information out, it knows to instantiate the derived class.
 * 
 * @snippet examples/polymorphic_types.cpp polymorphic_types
 * @see SAVEPOINT_DERIVED
 */
class SavepointBase
{
public:
    /**
     * @brief Default destructor.
     */
    virtual ~SavepointBase() = default;

    /**
     * @brief The Visit method to be called from SavepointVisitor.
     * 
     * @param visitor The visitor.
     * @see SavepointVisitor
     */
    virtual void Visit(SavepointVisitor& visitor) {}

    /** @cond INTERNAL */

    virtual std::string_view SavepointDerivedGetString() const = 0;

    /** @endcond */
};

/**
 * @brief Helper for concrete derived classes to implement SavepointBase methods.
 * 
 * Implements SavepointDerivedGetString and automatically registers a factory function
 * for the derived class. It allows a SavepointVisitor to to create an instance of
 * the derived class whilst only knowing its class name.
 * 
 * @param T The class type.
 * @see SavepointBase
 */
#define SAVEPOINT_DERIVED(T) \
    private: \
        struct SavepointDerivedFunctionRegistrar \
        { \
            static SavepointBase* Function() \
            { \
                return new T(); \
            } \
            SavepointDerivedFunctionRegistrar() \
            { \
                SavepointAddDerivedFunction(#T, Function); \
            } \
        }; \
        static inline SavepointDerivedFunctionRegistrar SavepointDerivedFunctionRegistrar; \
    public: \
        std::string_view SavepointDerivedGetString() const override \
        { \
            return #T; \
        } \

/** @cond INTERNAL */

using SavepointDerivedFunction = SavepointBase*(*)();

void SavepointAddDerivedFunction(const std::string_view& string, const SavepointDerivedFunction function);
SavepointDerivedFunction SavepointGetDerivedFunction(const std::string_view& string);
bool SavepointWriteDerived(SavepointBase* base, SavepointVisitor& visitor);
SavepointBase* SavepointReadDerived(SavepointVisitor& visitor);

template<typename T>
struct SavepointIsUniquePointerImpl : std::false_type {};

template<typename T, typename Deleter>
struct SavepointIsUniquePointerImpl<std::unique_ptr<T, Deleter>> : std::true_type {};

template<typename T>
concept SavepointIsUniquePointer = SavepointIsUniquePointerImpl<T>::value;

template<typename T>
struct SavepointIsSharedPointerImpl : std::false_type {};

template<typename T>
struct SavepointIsSharedPointerImpl<std::shared_ptr<T>> : std::true_type {};

template<typename T>
concept SavepointIsSharedPointer = SavepointIsSharedPointerImpl<T>::value;

template<typename T>
concept SavepointIsStdPointer = SavepointIsUniquePointer<T> || SavepointIsSharedPointer<T>;

template<typename T>
concept SavepointIsPointer = std::is_pointer_v<T> || SavepointIsStdPointer<T>;

template<typename T, typename Base>
concept SavepointIsPointerBaseOf = SavepointIsStdPointer<T> && std::is_base_of_v<Base, typename T::element_type>;

template<typename T>
struct SavepointIsPairImpl : std::false_type {};

template<typename First, typename Second>
struct SavepointIsPairImpl<std::pair<First, Second>> : std::true_type {};

template<typename T>
concept SavepointIsPair = SavepointIsPairImpl<T>::value;

template<typename T>
concept SavepointIsDynamicRange = requires(T item) { item.insert(std::ranges::end(item), std::declval<typename T::value_type>()); };

template<typename T>
concept SavepointIsStaticRange = !SavepointIsDynamicRange<T> && requires(T item) { item[0] = std::declval<typename T::value_type>(); };

template<typename T>
concept SavepointHasFreeVisit = requires(SavepointVisitor visitor, T item) { { Visit(visitor, item) }; };

template<typename T>
concept SavepointHasMemberVisit = requires(SavepointVisitor visitor, T item) { { item.Visit(visitor) }; };

template<typename T>
concept SavepointCanMemcpy = !SavepointIsPointer<T> && !SavepointHasFreeVisit<T> && !SavepointHasMemberVisit<T> && std::is_trivially_copyable_v<T>;

template<typename T>
concept SavepointCanVisit = !std::is_same_v<T, SavepointVisitor> && !std::is_base_of_v<SavepointBase, T>;

template<typename T>
struct SavepointIsEntityImpl : std::false_type {};

template<typename T>
concept SavepointIsEntity = std::is_base_of_v<SavepointEntity, T> || SavepointIsPointerBaseOf<T, SavepointEntity>;

/** @endcond */

/**
 * @brief Implementation of the Visitor pattern for serialization.
 * 
 * The visitor is used to serialize objects. It uses a simplified version of the
 * [pattern](https://refactoring.guru/design-patterns/visitor) with two operating modes:
 * 1. Reading from the Savepoint.
 * 2. Writing to the Savepoint.
 * 
 * Visitors are a structured blob of binary data. They consist of:
 * 1. A SavepointVersion representing the user's application build version.
 * 2. A SavepointVersion representing the Savepoint build version (reserved for future use).
 * 3. The object data.
 * 
 * When writing, we store the current build versions. When reading, we load the
 * versions used in the previous write. By comparing these versions to the build
 * versions, we can determine what members are safe to deserialize.
 * 
 * @snippet examples/nested_types.cpp nested_types
 */
class SavepointVisitor
{
private:
    static constexpr size_t kHeader = sizeof(SavepointVersion) * 2;

public:
    /**
     * @brief Serialize to/from an items's raw bytes.
     * 
     * Perform a simple memcpy from the visitor to the item's raw bytes (and
     * vice versa) using the size of the item. If the item cannot be deserialized,
     * it will be default initialized using args, assuming args are provided.
     * 
     * @tparam T The type to serialize.
     * @tparam Args The types of the args.
     * @param item The item to serialize.
     * @param version The version required to deserialize.
     * @param args The args for default initialization.
     */
    template<SavepointCanMemcpy T, typename... Args>
    void operator()(T& item, SavepointVersion version = {}, Args&&... args)
    {
        // For detecting bugs in MSVC concepts
        static_assert(!SavepointIsPointer<T>);
        static_assert(!std::is_base_of_v<SavepointBase, T>);
        static_assert(!std::ranges::range<T>);
        if (IsReading())
        {
            // Required for write-only containers (e.g. views)
            if constexpr (std::is_const_v<T>)
            {
                SavepointLog("Tried to read into a const");
                return;
            }
            else
            {
                if (Version < version)
                {
                    if constexpr (sizeof...(Args) > 0)
                    {
                        item = T{std::forward<Args>(args)...};
                    }
                    return;
                }
                if (sizeof(T) > GetSize())
                {
                    SavepointLog(std::format("Tried to read past visitor: {} -> {}", Version.GetString(), version.GetString()));
                    if constexpr (sizeof...(Args) > 0)
                    {
                        item = T{std::forward<Args>(args)...};
                    }
                    return;
                }
                std::memcpy(std::addressof(item), Reader.data() + Offset, sizeof(T));
                Offset += sizeof(T);
            }
        }
        else
        {
            Writer.resize(Writer.size() + sizeof(T));
            std::memcpy(Writer.data() + Writer.size() - sizeof(T), std::addressof(item), sizeof(T));
        }
    }

    /**
     * @brief Visit using the implementation from Visit.
     *
     * If the item cannot be deserialized, it will be default initialized using
     * args, assuming args are provided.
     * 
     * @tparam T The type to serialize.
     * @tparam Args The types of the args.
     * @param item The item to serialize.
     * @param version The version required to deserialize.
     * @param args The args for default initialization.
     */
    template<SavepointHasFreeVisit T, typename... Args>
    void operator()(T& item, SavepointVersion version = {}, Args&&... args)
    {
        if (IsReading())
        {
            if (Version < version || GetSize() == 0)
            {
                if constexpr (sizeof...(Args) > 0)
                {
                    item = T{std::forward<Args>(args)...};
                }
                return;
            }
        }
        Visit(*this, item);
    }

    /**
     * @brief Visit using the implementation from T::Visit.
     * 
     * If the item cannot be deserialized, it will be default initialized using
     * args, assuming args are provided.
     * 
     * @tparam T The type to serialize.
     * @tparam Args The types of the args.
     * @param item The item to serialize.
     * @param version The version required to deserialize.
     * @param args The args for default initialization.
     */
    template<SavepointHasMemberVisit T, typename... Args>
    void operator()(T& item, SavepointVersion version = {}, Args&&... args)
    {
        if (IsReading())
        {
            if (Version < version || GetSize() == 0)
            {
                if constexpr (sizeof...(Args) > 0)
                {
                    item = T{std::forward<Args>(args)...};
                }
                return;
            }
        }
        item.Visit(*this);
    }

    /**
     * @brief Skip bytes.
     * 
     * @tparam T The type to skip.
     */
    template<SavepointCanMemcpy T>
    void Skip()
    {
        if (IsReading())
        {
            if (sizeof(T) > GetSize())
            {
                SavepointLog(std::format("Tried to skip past visitor: {}", Version.GetString()));
                return;
            }
            Offset += sizeof(T);
        }
        else
        {
            Writer.resize(Writer.size() + sizeof(T));
        }
    }

    /**
     * @brief Disable deserialization.
     */
    void Fail()
    {
        if (!IsReading())
        {
            SavepointLog("Tried to fail while writing");
            return;
        }
        Offset = Reader.size();
    }

    /**
     * @brief Check if a visitor is reading.
     * 
     * @return True if the visitor is reading.
     */
    bool IsReading() const
    {
        return !Reader.empty();
    }

    /**
     * @brief Check if a visitor is writing.
     * 
     * @return True if the visitor is writing.
     */
    bool IsWriting() const
    {
        return !IsReading();
    }

    /**
     * @brief Get the application version.
     * 
     * @return The application version.
     */
    SavepointVersion GetVersion() const
    {
        return Version;
    }

    /**
     * @brief Get the number of bytes to write or the remaining to read.
     * 
     * @return The number of bytes.
     */
    size_t GetSize() const
    {
        if (IsReading())
        {
            return Reader.size() - std::min(Offset, Reader.size());
        }
        else
        {
            return Writer.size();
        }
    }

    /** @cond INTERNAL */

    void SavepointBeginReading(const void* data, size_t size)
    {
        void* pointer = const_cast<void*>(data);
        Reader = {static_cast<uint8_t*>(pointer), size};
        Offset = 0;
        operator()(Version);
        Skip<SavepointVersion>();
    }

    void SavepointBeginWriting(SavepointVersion version)
    {
        Reader = {};
        Writer.resize(kHeader);
        std::memcpy(Writer.data(), &version, sizeof(SavepointVersion));
        std::memcpy(Writer.data() + sizeof(SavepointVersion), &kSavepointVersion, sizeof(SavepointVersion));
        Version = version;
    }

    const void* SavepointGetData() const
    {
        return Writer.data();
    }

    /** @endcond */

private:
    SavepointVersion Version;
    std::vector<uint8_t> Writer;
    std::span<uint8_t> Reader;
    size_t Offset;
};

/**
 * @brief Visit implementation for serializing std::unique_ptr and std::shared_ptr.
 * 
 * Pointers data and whether they are null are serialized. As such, pointers are
 * allowed to be nullptr and will be handled accordingly. Polymorphics are also
 * supported by storing type information alongside the aforementioned data. When
 * reading, the correct derived type will be instantiated and deserialized.
 * 
 * Raw pointers are unsupported, not because they couldn't be, but because it's
 * not needed and avoids potential pitfalls.
 * 
 * @tparam T The type of the pointer.
 * @param visitor The visitor.
 * @param item The pointer.
 * @see SavepointBase
 * @see SAVEPOINT_DERIVED
 */
template<SavepointIsStdPointer T>
void Visit(SavepointVisitor& visitor, T& item)
{
    using ValueT = typename T::element_type;
    if (visitor.IsReading())
    {
        bool hasPointer = false;
        visitor(hasPointer);
        if (!hasPointer)
        {
            if (item)
            {
                SavepointLog("Nulled an allocated pointer since visitor contained a nullptr");
                item.reset();
            }
            return;
        }
        if (!item)
        {
            if constexpr (std::is_base_of_v<SavepointBase, ValueT>)
            {
                item.reset(dynamic_cast<ValueT*>(SavepointReadDerived(visitor)));
                return;
            }
            else if constexpr (std::is_default_constructible_v<ValueT>)
            {
                item.reset(new ValueT());
                if (!item)
                {
                    SavepointLog("Failed to allocate pointer");
                    visitor.Fail();
                    return;
                }
            }
            else
            {
                // Don't static_assert because it'll fail on already instantiated
                // derived classes with abstract parents
                SavepointLog("No method to create pointer");
                visitor.Fail();
                return;
            }
        }
        visitor(*item);
    }
    else
    {
        bool hasPointer = item.get() != nullptr;
        visitor(hasPointer);
        if (hasPointer)
        {
            if constexpr (std::is_base_of_v<SavepointBase, ValueT>)
            {
                SavepointWriteDerived(item.get(), visitor);
            }
            else
            {
                visitor(*item);
            }
        }
    }
}

/**
 * @brief Visit implementation for serializing an std::pair.
 * 
 * @tparam T The type of the pair.
 * @param visitor The visitor.
 * @param item The pair.
 */
template<SavepointIsPair T>
void Visit(SavepointVisitor& visitor, T& item)
{
    // Required because maps use const for value_type::first_type
    using First = std::remove_const_t<typename T::first_type>;
    First& first = const_cast<First&>(item.first);
    visitor(first);
    visitor(item.second);
}

/**
 * @brief Visit implementation for serializing containers.
 * 
 * @tparam T The type of the container.
 * @param visitor The visitor.
 * @param item The pointer.
 */
template<std::ranges::range T>
void Visit(SavepointVisitor& visitor, T& item)
{
    using ValueT = typename T::value_type;
    size_t size = item.size();
    if constexpr (SavepointIsDynamicRange<T>)
    {
        if (visitor.IsReading() && size)
        {
            SavepointLog("Tried to read into non-empty range");
            item.clear();
        }
    }
    visitor(size);
    if (visitor.IsReading())
    {
        // Can detect when we read garbage and would iterate forever
        if (size > visitor.GetSize())
        {
            SavepointLog("Tried to read past visitor");
            visitor.Fail();
            return;
        }
        if constexpr (SavepointIsDynamicRange<T>)
        {
            auto inserter = std::inserter(item, std::ranges::end(item));
            for (size_t i = 0; i < size; i++)
            {
                // TODO: mutable iterators
                ValueT element;
                visitor(element);
                *inserter++ = element;
            }
        }
        else if constexpr (SavepointIsStaticRange<T>)
        {
            size_t maxSize = std::ranges::size(item);
            if (size > maxSize)
            {
                SavepointLog(std::format("Fixed range is too small: {} < {}", maxSize, size));
            }
            if (size < maxSize)
            {
                SavepointLog(std::format("Fixed range will be truncated: {} < {}", maxSize, size));
            }
            maxSize = std::min(size, maxSize);
            for (size_t i = 0; i < maxSize; i++)
            {
                visitor(item[i]);
            }
            for (; maxSize < size; maxSize++)
            {
                visitor.Skip<ValueT>();
            }
        }
        else
        {
            // Required for write-only containers (e.g. views)
            SavepointLog("Unknown range");
        }
    }
    else
    {
        for (auto& element : item)
        {
            visitor(element);
        }
    }
}

/**
 * @brief The statuses returned by Savepoint.
 */
enum class SavepointStatus
{
    Failed,   /**< Failed to open a Savepoint for any reason */
    Existing, /**< Opened an existing Savepoint */
    New,      /**< Created a new Savepoint */
};

/**
 * @brief The implementation for Savepoint's file operations.
 */
enum class SavepointDriver : uint8_t
{
    SQLite3, /**< Backed by sqlite3. */
    Null,    /**< Noop. */
};

/** @cond INTERNAL */

using SavepointReadVisitorFunction = std::function<void(SavepointVisitor& visitor)>;
using SavepointReadVisitorEntityFunction = std::function<void(SavepointVisitor& visitor, SavepointID id)>;
using SavepointReadVisitorTile2DFunction = std::function<void(SavepointVisitor& visitor, int x, int y)>;
using SavepointReadVisitorTile3DFunction = std::function<void(SavepointVisitor& visitor, int x, int y, int z)>;
using SavepointReadLevelFunction = std::function<void(int level)>;

class ISavepointDriver
{
public:
    virtual SavepointStatus Open(const std::string_view& path, SavepointVersion version) = 0;
    virtual bool IsOpen() const = 0;
    virtual void Write(SavepointVisitor& visitor) = 0;
    virtual SavepointID Insert(SavepointVisitor& visitor, int level) = 0;
    virtual SavepointID Update(SavepointVisitor& visitor, SavepointID id, int level) = 0;
    virtual void Write(SavepointVisitor& visitor, int x, int y, int level) = 0;
    virtual void Write(SavepointVisitor& visitor, int x, int y, int z, int level) = 0;
    virtual void Read(const SavepointReadVisitorFunction& function) = 0;
    virtual void Read(const SavepointReadVisitorEntityFunction& function, int level) = 0;
    virtual void Read(const SavepointReadVisitorTile2DFunction& function, int level) = 0;
    virtual void Read(const SavepointReadVisitorTile3DFunction& function, int level) = 0;
    virtual void Read(const SavepointReadLevelFunction& function) = 0;
    virtual void Delete(SavepointID id) = 0;
    virtual void Close() = 0;
    virtual void Save() = 0;
    virtual void Clear() = 0;
};

/** @endcond */

/**
 * @brief The read function signature.
 * 
 * @tparam T The type to read.
 * @param item The read item.
 */
template<typename T>
using SavepointReadFunction = std::function<void(T& item)>;

/**
 * @brief The entity read function signature.
 * 
 * @tparam T The type to read.
 * @param item The read item.
 * @see SavepointID
 */
template<typename T>
using SavepointReadEntityFunction = std::function<void(T& item)>;

/**
 * @brief The 2D tile read function signature.
 * 
 * @tparam T The type to read.
 * @param item The read item.
 * @param x The x location.
 * @param y The y location.
 */
template<typename T>
using SavepointReadTile2DFunction = std::function<void(T& item, int x, int y)>;

/**
 * @brief The 3D tile read function signature.
 * 
 * @tparam T The type to read.
 * @param item The read item.
 * @param x The x location.
 * @param y The y location.
 * @param z The z location.
 */
template<typename T>
using SavepointReadTile3DFunction = std::function<void(T& item, int x, int y, int z)>;

/**
 * @brief The connection handle to a Savepoint file.
 * 
 * @snippet examples/basic_usage.cpp basic_usage
 */
class Savepoint
{
public:
    /**
     * @brief Default initializes the connection.
     */
    Savepoint() = default;

    /**
     * @brief If connected, closes the connection.
     */
    ~Savepoint();

    /**
     * @brief Deleted copy constructor.
     */
    Savepoint(const Savepoint& other) = delete;
    
    /**
     * @brief Deleted copy assignment operator.
     */
    Savepoint& operator=(const Savepoint& other) = delete;
    
    /**
     * @brief Deleted move constructor.
     */
    Savepoint(Savepoint&& other) = delete;
    
    /**
     * @brief Deleted move assignment operator.
     */
    Savepoint& operator=(Savepoint&& other) = delete;
    
    /**
     * @brief Opens a new connection.
     * 
     * @param driver The driver to use for file operations.
     * @param path The path to the Savepoint file.
     * @param version The user's application version.
     * @return The result of the attempt to open a connection.
     * @see Save
     * @see Close
     */
    SavepointStatus Open(SavepointDriver driver, const std::string_view& path, SavepointVersion version);

    /**
     * @brief Write a singleton to the Savepoint.
     * 
     * For storing information such as date and time, the user can write a
     * singleton with the assumption that only one entry exists.
     * 
     * @tparam T The type to write.
     * @param item The item to write.
     */
    template<SavepointCanVisit T>
    void Write(T& item)
    {
        if (!Driver->IsOpen())
        {
            return;
        }
        Visitor.SavepointBeginWriting(Version);
        Visitor(item);
        Driver->Write(Visitor);
    }

    /**
     * @brief Write an entity to the Savepoint.
     * 
     * For items without a unique location, an ID can be used to ensure the item
     * gets a unique entry. If the ID is invalid, the ID will be written to and
     * a new entry will be inserted. If the ID is valid, an existing entry will
     * be updated (including the level).
     * 
     * @tparam T The type to write.
     * @param item The item to write.
     * @param level The level.
     * @see SavepointEntity
     */
    template<SavepointIsEntity T>
    void Write(T& item, int level)
    {
        if (!Driver->IsOpen())
        {
            return;
        }
        Visitor.SavepointBeginWriting(Version);
        Visitor(item);
        SavepointID& id = GetID(item);
        // Not an error. Inserting a new entry
        if (!id.IsValid())
        {
            id = Driver->Insert(Visitor, level);
        }
        else
        {
            id = Driver->Update(Visitor, id, level);
            // Update failed so try inserting
            if (!id.IsValid())
            {
                id = Driver->Insert(Visitor, level);
            }
        }
    }

    /**
     * @brief Write a 2D tile to the Savepoint.
     * 
     * Writes a tile to a entry using a key of x, y, and level. If an entry
     * already exists, the entry will be overridden.
     * 
     * @tparam T The type to write.
     * @param item The item to write.
     * @param x The x location.
     * @param y The y location.
     * @param level The level.
     */
    template<SavepointCanVisit T>
    void Write(T& item, int x, int y, int level)
    {
        if (!Driver->IsOpen())
        {
            return;
        }
        Visitor.SavepointBeginWriting(Version);
        Visitor(item);
        Driver->Write(Visitor, x, y, level);
    }

    /**
     * @brief Write a 3D tile to the Savepoint.
     * 
     * Writes a tile to a entry using a key of x, y, z, and level. If an entry
     * already exists, the entry will be overridden.
     * 
     * @tparam T The type to write.
     * @param item The item to write.
     * @param x The x location.
     * @param y The y location.
     * @param z The z location.
     * @param level The level.
     */
    template<SavepointCanVisit T>
    void Write(T& item, int x, int y, int z, int level)
    {
        if (!Driver->IsOpen())
        {
            return;
        }
        Visitor.SavepointBeginWriting(Version);
        Visitor(item);
        Driver->Write(Visitor, x, y, z, level);
    }

    /**
     * @brief Read a singleton from the Savepoint.
     * 
     * @tparam T The type to read.
     * @param item The item to read.
     */
    template<SavepointCanVisit T>
    void Read(T& item)
    {
        if (!Driver->IsOpen())
        {
            return;
        }
        Driver->Read([&item](SavepointVisitor& visitor)
        {
            visitor(item);
        });
    }

    /**
     * @brief Read all entities in the specified level from the Savepoint.
     * 
     * @tparam T The type to read.
     * @param function The function to use.
     * @param level The level.
     * @see SavepointEntity
     */
    template<SavepointCanVisit T>
    void Read(const SavepointReadEntityFunction<T>& function, int level)
    {
        if (!Driver->IsOpen())
        {
            return;
        }
        Driver->Read([this, &function](SavepointVisitor& visitor, SavepointID id)
        {
            T item;
            visitor(item);
            GetID(item) = id;
            function(item);
        }, level);
    }

    /**
     * @brief Read all 2D tiles in the specified level from the Savepoint.
     * 
     * @tparam T The type to read.
     * @param function The function to use.
     * @param level The level.
     */
    template<SavepointCanVisit T>
    void Read(const SavepointReadTile2DFunction<T>& function, int level)
    {
        if (!Driver->IsOpen())
        {
            return;
        }
        Driver->Read([&function](SavepointVisitor& visitor, int x, int y)
        {
            T item;
            visitor(item);
            function(item, x, y);
        }, level);
    }

    /**
     * @brief Read all 3D tiles in the specified level from the Savepoint.
     * 
     * @tparam T The type to read.
     * @param function The function to use.
     * @param level The level.
     */
    template<SavepointCanVisit T>
    void Read(const SavepointReadTile3DFunction<T>& function, int level)
    {
        if (!Driver->IsOpen())
        {
            return;
        }
        Driver->Read([&function](SavepointVisitor& visitor, int x, int y, int z)
        {
            T item;
            visitor(item);
            function(item, x, y, z);
        }, level);
    }

    /**
     * @brief Get all the levels from the Savepoint. Duplicates are removed.
     * 
     * @return The levels.
     */
    std::vector<int> GetLevels()
    {
        if (!Driver->IsOpen())
        {
            return {};
        }
        std::vector<int> levels;
        Driver->Read([&levels](int level)
        {
            levels.push_back(level);
        });
        return levels;
    }

    /**
     * @brief Deletes an entity from the Savepoint.
     * 
     * @tparam T The type to delete
     * @param item The item to delete.
     * @see SavepointEntity
     */
    template<SavepointIsEntity T>
    void Delete(T& item)
    {
        if (!Driver->IsOpen())
        {
            return;
        }
        SavepointID& id = GetID(item);
        if (id.IsValid())
        {
            Driver->Delete(id);
        }
    }
    
    /**
     * @brief Closes the connection. Does NOT call Savepoint::Save.
     * 
     * @see Save
     * @see Close
     */
    void Close();
    
    /**
     * @brief Save all pending changes.
     * 
     * Commits the current transaction and starts a new one. The next time
     * Savepoint::Open is called, it will return SavepointStatus::Existing
     * instead of SavepointStatus::New.
     * 
     * @see Open
     */
    void Save();

    /**
     * @brief Remove all entities and tiles from the Savepoint.
     */
    void Clear();

private:
    template<SavepointIsEntity T>
    static constexpr SavepointID& GetID(T& item)
    {
        if constexpr (SavepointIsStdPointer<T>)
        {
            return item->ID;
        }
        else
        {
            return item.ID;
        }
    }

    SavepointVersion Version;
    SavepointVisitor Visitor;
    std::unique_ptr<ISavepointDriver> Driver;
};
