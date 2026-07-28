#include <savepoint/savepoint.hpp>

#include <array>
#include <deque>
#include <list>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

static_assert(SavepointIsPointer<void*>);
static_assert(SavepointIsPointer<void**>);
static_assert(SavepointIsPointer<const void*>);
static_assert(SavepointIsPointer<void const*>);
static_assert(SavepointIsPointer<const void* const>);
static_assert(!SavepointIsPointer<int>);
static_assert(SavepointIsPointer<std::shared_ptr<int>>);
static_assert(SavepointIsPointer<std::unique_ptr<int>>);

struct NoVisit {};
struct MemberVisit { void Visit(SavepointVisitor& visitor) {} };
struct FreeVisit {};
static void Visit(SavepointVisitor& visitor, FreeVisit& item) {}

static_assert(SavepointHasFreeVisit<FreeVisit>);
static_assert(!SavepointHasFreeVisit<MemberVisit>);
static_assert(!SavepointHasFreeVisit<NoVisit>);
static_assert(!SavepointHasMemberVisit<FreeVisit>);
static_assert(SavepointHasMemberVisit<MemberVisit>);
static_assert(!SavepointHasMemberVisit<NoVisit>);

static_assert(!SavepointIsDynamicRange<std::array<int, 1>>);
static_assert(SavepointIsDynamicRange<std::vector<int>>);
static_assert(!SavepointIsDynamicRange<std::string_view>);
static_assert(SavepointIsDynamicRange<std::map<int, int>>);
static_assert(SavepointIsDynamicRange<std::unordered_map<int, int>>);
static_assert(SavepointIsDynamicRange<std::set<int>>);
static_assert(SavepointIsDynamicRange<std::unordered_set<int>>);
static_assert(SavepointIsDynamicRange<std::deque<int>>);
static_assert(SavepointIsDynamicRange<std::string>);
static_assert(SavepointIsDynamicRange<std::list<int>>);

static_assert(SavepointIsStaticRange<std::array<int, 1>>);
static_assert(!SavepointIsStaticRange<std::vector<int>>);
static_assert(!SavepointIsStaticRange<std::string_view>);
static_assert(!SavepointIsStaticRange<std::map<int, int>>);
static_assert(!SavepointIsStaticRange<std::unordered_map<int, int>>);
static_assert(!SavepointIsStaticRange<std::set<int>>);
static_assert(!SavepointIsStaticRange<std::unordered_set<int>>);
static_assert(!SavepointIsStaticRange<std::string>);
static_assert(!SavepointIsStaticRange<std::deque<int>>);

static_assert(!SavepointHasMemberVisit<std::string>);
static_assert(SavepointHasFreeVisit<std::string>);

static_assert(SavepointIsTuple<std::tuple<int>>);
static_assert(SavepointIsTuple<std::tuple<int, int>>);
static_assert(SavepointIsTuple<std::tuple<int, int, int>>);
static_assert(SavepointIsTuple<std::pair<int, int>>);

static_assert(SavepointIsOptional<std::optional<int>>);
static_assert(!SavepointIsOptional<std::tuple<int>>);

int main()
{
    return 0;
}
