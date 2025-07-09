/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * Licence, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef VARITENSOR_INDEX_H
#define VARITENSOR_INDEX_H

#include <format>
#include <string>
#include <utility>

#include "deny.h"

namespace varitensor {

class Index;

namespace impl {
constexpr int MAX_INTERVAL = -1;
} // namespace impl

struct Interval {
    const Index& origin;
    const int first;
    const int last;
};

// for nice initialisation of indices with standard sizes
enum IndexSizes {LATIN = 3, GREEK = 4};

class Index {
public:
// =================================================================================================
//                                                                                          c/dtor |
// =================================================================================================

    explicit Index(const int size):
        m_size{size},
        m_id{s_next_id++}
    {
        impl::soft_deny(m_size < 2, "Cannot initialize unnamed index with size < 2!");
    }

    explicit Index(std::string  name, const int size):
        m_size{size},
        m_id{s_next_id++},
        m_name{std::move(name)}
    {
        impl::soft_deny(m_size < 2, "Cannot initialize index with size < 2!");
    }

    Index(const Interval& interval): // NOLINT - we want implicit conversion
        m_size{interval.last - interval.first + 1},
        m_id{interval.origin.m_id},
        m_name{interval.origin.m_name},
        m_interval_start{interval.first}
    {}

// =================================================================================================
//                                                                                      comparison |
// =================================================================================================

    bool operator==(const Index& other) const {
        if (m_name.empty()) return m_id == other.m_id && m_size == other.m_size;
        return m_name == other.m_name && m_size == other.m_size;
    }

// =================================================================================================
//                                                                               getters / setters |
// =================================================================================================

    [[nodiscard]] const int& size() const { return m_size; }

    [[nodiscard]] const int& id() const { return m_id; }

    [[nodiscard]] std::string name() const {
        auto name = m_name.empty() ? std::format("idx{}", m_id) : m_name;
        if (m_interval_start != -1) name += std::format("({}-{})", m_interval_start, m_interval_start + m_size - 1);
        return name;
    }

    void set_name(const std::string& name) {
        m_name = name;
    }

// =================================================================================================
//                                                                               interval creation |
// =================================================================================================

    [[nodiscard]] Interval operator()(const int first, const int last=impl::MAX_INTERVAL) const {
        return interval(first, last);
    }

    [[nodiscard]] Interval interval(const int first, int last=impl::MAX_INTERVAL) const {
        if (last == impl::MAX_INTERVAL) last = m_size - 1;

        impl::soft_deny(first < 0, "Interval cannot have negative start!");
        impl::soft_deny(first >= last, "Interval must end after it starts!");
        impl::soft_deny(last >= m_size, "Interval cannot overflow index size!");
        return {*this, first, last};
    }

private:
// =================================================================================================
//                                                                                member variables |
// =================================================================================================

    int m_size;
    int m_id;
    std::string m_name{};
    int m_interval_start{-1};

    inline static int s_next_id{0};
};

} // namespace varitensor

#endif
