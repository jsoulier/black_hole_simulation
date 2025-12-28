#include <savepoint.hpp>

#include <array>
#include <map>
#include <memory>
#include <set>
#include <string_view>
#include <unordered_map>
#include <vector>

static_assert(SavepointPointer<void*>);
static_assert(SavepointPointer<void**>);
static_assert(SavepointPointer<const void*>);
static_assert(SavepointPointer<void const*>);
static_assert(SavepointPointer<const void* const>);
static_assert(!SavepointPointer<int>);
static_assert(SavepointPointer<std::shared_ptr<int>>);
static_assert(SavepointPointer<std::unique_ptr<int>>);

struct NoVisit {};
struct MemberVisit { void Visit(SavepointVisitor& visitor) {} };
struct FreeVisit {};
static void SavepointVisit(SavepointVisitor& visitor, FreeVisit& item) {}

static_assert(SavepointFreeVisit<FreeVisit>);
static_assert(!SavepointFreeVisit<MemberVisit>);
static_assert(!SavepointFreeVisit<NoVisit>);
static_assert(!SavepointMemberVisit<FreeVisit>);
static_assert(SavepointMemberVisit<MemberVisit>);
static_assert(!SavepointMemberVisit<NoVisit>);

static_assert(!SavepointDynamicRange<std::array<int, 1>>);
static_assert(SavepointDynamicRange<std::vector<int>>);
static_assert(!SavepointDynamicRange<std::string_view>);
static_assert(SavepointDynamicRange<std::map<int, int>>);
static_assert(SavepointDynamicRange<std::unordered_map<int, int>>);
static_assert(SavepointDynamicRange<std::set<int>>);

static_assert(SavepointStaticRange<std::array<int, 1>>);
static_assert(!SavepointStaticRange<std::vector<int>>);
static_assert(!SavepointStaticRange<std::string_view>);
static_assert(!SavepointStaticRange<std::map<int, int>>);
static_assert(!SavepointStaticRange<std::unordered_map<int, int>>);
static_assert(!SavepointStaticRange<std::set<int>>);

int main()
{
    return 0;
}
