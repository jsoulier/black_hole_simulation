#pragma once

#include <savepoint/fwd.hpp>

#include <string_view>

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
bool SavepointWriteDerived(SavepointBase* base, SavepointVisitor& visitor);
SavepointBase* SavepointReadDerived(SavepointVisitor& visitor);

/** @endcond */