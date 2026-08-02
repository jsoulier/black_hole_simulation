#include <savepoint/savepoint.hpp>

#include <mutex>
#include <thread>

#include "mutex.hpp"

SavepointMutex::SavepointMutex()
    : Enabled{false}
    , Mutex{}
#ifndef NDEBUG
    , ThreadID{}
#endif
{
}

void SavepointMutex::SetEnabled(bool enabled)
{
    Enabled = enabled;
#ifndef NDEBUG
    if (!Enabled)
    {
        ThreadID = std::this_thread::get_id();
    }
#endif
}

void SavepointMutex::lock()
{
    if (Enabled)
    {
        Mutex.lock();
    }
#ifndef NDEBUG
    else if (ThreadID != std::this_thread::get_id())
    {
        SavepointLog("Savepoint used from multiple threads without thread safety enabled");
    }
#endif
}

void SavepointMutex::unlock()
{
    if (Enabled)
    {
        Mutex.unlock();
    }
}
