#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <concepts>

namespace Scoop::Vectors
{
    template <std::floating_point T, size_t N>
    struct Vector
    {
        /* Data */

        std::array<T, N> data;

        decltype(auto) operator[](this auto &&self, size_t index) { return self.data[index]; }

        /* Operations */

        T Dot(const Vector<T, N> &other) const
        {
            T dot = T();

            for (size_t i = 0; i < N; i++)
                dot += this->data[i] * other.data[i];

            return dot;
        }

        T Magnitude() const { return sqrt(this->Dot(*this)); }

        Vector &Normalize()
        {
            if (T magnitude = this->Magnitude())
                *this /= magnitude;

            return *this;
        }

        auto Normalized() const { return (Vector<T, N> {*this}.Normalize()); }

        /* Arithmetic */

        Vector &operator+=(const Vector &other)
        {
            for (size_t i = 0; i < N; i++)
                this->data[i] += other.data[i];

            return *this;
        }

        Vector &operator-=(const Vector &other)
        {
            for (size_t i = 0; i < N; i++)
                this->data[i] += other.data[i];

            return *this;
        }

        Vector &operator*=(T scalar)
        {
            for (size_t i = 0; i < N; i++)
                this->data[i] *= scalar;

            return *this;
        }

        Vector &operator/=(T scalar)
        {
            for (size_t i = 0; i < N; i++)
                this->data[i] /= scalar;

            return *this;
        }

        Vector operator+(const Vector<T, N> &other) const { return (Vector<T, N> {*this} += other); }
        Vector operator-(const Vector<T, N> &other) const { return (Vector<T, N> {*this} -= other); }
        Vector operator*(T scalar) const { return (Vector<T, N> {*this} *= scalar); }
        Vector operator/(T scalar) const { return (Vector<T, N> {*this} /= scalar); }
    };
}