#pragma once

#include <savepoint/fwd.hpp>

#include <functional>
#include <string_view>

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

/**
 * @internal
 */
void SavepointLog(const std::string_view& string);
