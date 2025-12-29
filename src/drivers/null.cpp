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

#include "null.hpp"

SavepointStatus SavepointDriverNull::Open(const std::string_view& path, SavepointVersion version)
{
    return SavepointStatus::New;
}

bool SavepointDriverNull::IsOpen() const
{
    return false;
}

void SavepointDriverNull::Write(SavepointVisitor& visitor)
{
}

void SavepointDriverNull::Write(SavepointVisitor& visitor, SavepointID& id, int level)
{
}

void SavepointDriverNull::Write(SavepointVisitor& visitor, int x, int y, int level)
{
}

void SavepointDriverNull::Write(SavepointVisitor& visitor, int x, int y, int z, int level)
{
}

void SavepointDriverNull::Read(const SavepointReadVisitorFunction& function)
{
}

void SavepointDriverNull::Read(const SavepointReadVisitorEntityFunction& function, int level)
{
}

void SavepointDriverNull::Read(const SavepointReadVisitorTile2DFunction& function, int level)
{
}

void SavepointDriverNull::Read(const SavepointReadVisitorTile3DFunction& function, int level)
{
}

void SavepointDriverNull::Delete(const SavepointID id)
{
}

void SavepointDriverNull::Close()
{
}

void SavepointDriverNull::Save()
{
}

void SavepointDriverNull::Clear()
{
}
