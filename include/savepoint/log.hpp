#pragma once

#include <functional>
#include <string_view>

using SavepointLogFunction = std::function<void(const std::string_view& string)>;

void SavepointSetLogFunction(const SavepointLogFunction& function);
void SavepointLog(const std::string_view& string);
