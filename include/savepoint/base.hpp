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

#include <string_view>

// Base class for user's base class (optional, only required for polymorphics)
class SavepointBase
{
public:
    virtual void Visit(SavepointVisitor& visitor) = 0;

    // For getting the class name at runtime for derived factory lookups
    virtual std::string_view SavepointDerivedGetString() const = 0;
};

using SavepointDerivedFunction = SavepointBase*(*)();

// Register a derived factory. Use SAVEPOINT_DERIVED instead
void SavepointAddDerivedFunction(const std::string_view& string, const SavepointDerivedFunction function);

// Create an object from the registered derived factories
SavepointBase* SavepointCreateDerived(const std::string_view& string);

// For user's concrete derived classes
#define SAVEPOINT_DERIVED(T) \
    private: \
        /* Automatically register derived factory */ \
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
            return #T;\
        } \
