#pragma once

#include <Scoop/Vectors/Vector.hpp>

#include <string>
#include <format>
#include <charconv>
#include <expected>

namespace Scoop::Vectors
{
    template <std::floating_point T, size_t N>
    std::string EncodeVector(const Vector<T, N> &vector)
    {
        std::string str;
        bool first = true;

        for (size_t i = 0; i < N; i++)
        {
            if (first)
                first = false;
            else
                str += ',';

            str += std::to_string(vector.data[i]);
        }

        return str;
    }

    template <std::floating_point T, size_t N>
    std::expected<Vector<T, N>, std::string> DecodeVector(std::string_view str)
    {
        Vector<T, N> vector;

        size_t i = 0;
        size_t start = 0;
        size_t length = str.length();

        while (start <= length)
        {
            if (i >= N)
                return std::unexpected(std::format("Input dimensions exceeds vector size ({}).", N));

            size_t end = str.find(',', start);

            if (end == std::string::npos)
                end = length;

            std::string_view element(str.data() + start, end - start);

            T value = T();
            auto [ptr, ec] = std::from_chars(element.data(), element.data() + element.size(), value);

            if (ec != std::errc() || ptr != element.data() + element.size())
                return std::unexpected(std::format("Invalid float at element {}: '{}'.", i, std::string(element)));

            vector.data[i++] = value;

            if (end == length)
                break;

            start = end + 1;
        }

        if (i != N)
            return std::unexpected(std::format("Expected {}-d vector, got {}.", N, i));

        return vector;
    }
}
