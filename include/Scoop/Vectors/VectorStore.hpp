#pragma once

#include <Scoop/Vectors/Vector.hpp>
#include <Scoop/Vectors/Serialize.hpp>

#include <span>
#include <vector>
#include <algorithm>
#include <iterator>
#include <shared_mutex>

namespace Scoop::Vectors
{
    struct SearchOptions
    {
        size_t topK = 10;
        float scoreThreshold = -1.0f;
        std::span<const size_t> excludedIndices {};
    };

    struct SearchResult
    {
        size_t index;
        float score;
    };

    template <std::floating_point T, size_t N>
    class VectorStore {
    public:
        /* Data Type */

        using DataT = Vector<T, N>;

        /* Constructor */

        VectorStore() = default;

        /* Storage */

        void Reserve(size_t capacity)
        {
            std::unique_lock<std::shared_mutex> lock(m_Mutex);

            m_Entries.reserve(capacity);
            m_Magnitudes.reserve(capacity);
        }

        size_t Size() const
        {
            std::shared_lock<std::shared_mutex> lock(m_Mutex);

            return m_Entries.size();
        }

        void Clear()
        {
            std::unique_lock<std::shared_mutex> lock(m_Mutex);

            m_Entries.clear();
            m_Magnitudes.clear();
        }

        /* Add Entries */

        void Add(DataT entry)
        {
            std::unique_lock<std::shared_mutex> lock(m_Mutex);

            m_Magnitudes.emplace_back(entry.Magnitude());
            m_Entries.emplace_back(std::move(entry));
        }

        void Add(const std::vector<DataT> &entries)
        {
            std::unique_lock<std::shared_mutex> lock(m_Mutex);

            m_Entries.reserve(m_Entries.size() + entries.size());
            m_Magnitudes.reserve(m_Magnitudes.size() + entries.size());

            for (const DataT &entry : entries)
                m_Magnitudes.emplace_back(entry.Magnitude());

            m_Entries.insert(m_Entries.end(), entries.begin(), entries.end());
        }

        void Add(std::vector<DataT> &&entries)
        {
            std::unique_lock<std::shared_mutex> lock(m_Mutex);

            m_Entries.reserve(m_Entries.size() + entries.size());
            m_Magnitudes.reserve(m_Magnitudes.size() + entries.size());

            for (const DataT &entry : entries)
                m_Magnitudes.emplace_back(entry.Magnitude());

            m_Entries.insert(
                m_Entries.end(),
                std::make_move_iterator(entries.begin()),
                std::make_move_iterator(entries.end())
            );
        }

        std::expected<void, std::string> Remove(size_t index)
        {
            std::unique_lock<std::shared_mutex> lock(m_Mutex);

            if (index >= m_Entries.size())
                return std::unexpected(std::format("Index {} exceeds max ({}).", index, m_Entries.size()));

            m_Entries.erase(m_Entries.begin() + index);
            m_Magnitudes.erase(m_Magnitudes.begin() + index);
        }

        /* Search */

        std::vector<SearchResult> Search(const DataT &query, const SearchOptions &options) const
        {
            if (options.topK == 0)
                return {};

            std::shared_lock<std::shared_mutex> lock(m_Mutex);

            std::vector<SearchResult> results;
            results.reserve(options.topK);

            T queryMagnitude = query.Magnitude();

            if (queryMagnitude == 0)
                return results;

            T inverseQueryMagnitude = 1 / queryMagnitude;

            size_t count = m_Entries.size();

            for (size_t entryIndex = 0; entryIndex < count; entryIndex++)
            {
                auto excludeIt = std::find(options.excludedIndices.begin(), options.excludedIndices.end(), entryIndex);

                if (excludeIt != options.excludedIndices.end())
                    continue;

                const DataT &entry = m_Entries[entryIndex];
                T entryMagnitude = m_Magnitudes[entryIndex];

                T dot = entry.Dot(query);

                if (entryMagnitude == 0)
                    continue;

                T entryScore = (dot / entryMagnitude) * inverseQueryMagnitude;

                if (entryScore < options.scoreThreshold)
                    continue;

                if (results.size() == options.topK && entryScore <= results.back().score)
                    continue;

                auto it = results.begin();

                while (it != results.end())
                {
                    if (entryScore > it->score)
                        break;

                    it++;
                }

                size_t resultCount = it - results.begin();

                if (resultCount < options.topK)
                {
                    if (entryScore < -1.0f)
                        entryScore = -1.0f;

                    if (entryScore > 1.0f)
                        entryScore = 1.0f;

                    if (results.size() == options.topK)
                        results.pop_back();

                    results.insert(it, SearchResult {.index = entryIndex, .score = entryScore});
                }
            }

            return results;
        }

        /* Persistence */

        std::string Save() const
        {
            std::shared_lock<std::shared_mutex> lock(m_Mutex);

            std::string serialized;
            serialized.reserve(m_Entries.size() * (N * 12 + 1));

            for (const DataT &entry : m_Entries)
            {
                serialized += EncodeVector(entry);
                serialized += '\n';
            }

            return serialized;
        }

        std::expected<void, std::string> Load(std::string_view serialized)
        {
            std::unique_lock<std::shared_mutex> lock(m_Mutex);

            std::vector<DataT> loadedData;
            std::vector<T> loadedMagnitudes;

            loadedData.reserve(std::count(serialized.begin(), serialized.end(), '\n') + 1);
            loadedMagnitudes.reserve(loadedData.capacity());

            size_t lineStart = 0;
            size_t length = serialized.length();

            while (lineStart <= length)
            {
                size_t lineEnd = serialized.find('\n', lineStart);

                if (lineEnd == std::string_view::npos)
                    lineEnd = length;

                std::string_view line = serialized.substr(lineStart, lineEnd - lineStart);

                if (!line.empty() && line.back() == '\r')
                    line.remove_suffix(1);

                if (line.empty())
                {
                    if (lineEnd == length)
                        break;

                    lineStart = lineEnd + 1;
                    continue;
                }

                std::expected<DataT, std::string> entry = DecodeVector<N>(line);

                if (!entry)
                    return std::unexpected(entry.error());

                loadedMagnitudes.emplace_back(entry.value().Magnitude());
                loadedData.emplace_back(std::move(entry.value()));

                if (lineEnd == length)
                    break;

                lineStart = lineEnd + 1;
            }

            m_Entries = std::move(loadedData);
            m_Magnitudes = std::move(loadedMagnitudes);

            return {};
        }

    private:
        std::vector<Vector<T, N>> m_Entries;
        std::vector<T> m_Magnitudes;
        mutable std::shared_mutex m_Mutex;
    };
}
