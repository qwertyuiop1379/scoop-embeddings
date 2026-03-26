#include <Scoop/Vectors/VectorStore.hpp>

#include <array>
#include <atomic>
#include <cerrno>
#include <cctype>
#include <cstring>
#include <expected>
#include <filesystem>
#include <fstream>
#include <format>
#include <iostream>
#include <mutex>
#include <optional>
#include <print>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/types.h>
#include <sys/wait.h>
#include <thread>
#include <unordered_set>
#include <unistd.h>
#include <utility>
#include <vector>

namespace
{
    constexpr size_t N_DIM = 768;
    std::string DIM_STR = std::format("{}", N_DIM);
    constexpr std::string_view kModelName = "embeddinggemma";

    using Store = Scoop::Vectors::VectorStore<float, N_DIM>;
    using Vector = Store::DataT;

    std::string_view Trim(std::string_view value);

    struct QueryRecipe
    {
        std::string label;
        Vector vector;
        std::vector<std::string_view> excludedLabels;
    };

    struct WeightedLabel
    {
        std::string_view label;
        float weight;
    };

    struct Manifest
    {
        std::string model;
        size_t dimensions = 0;
        size_t inputCount = 0;
        uint64_t inputHash = 0;
        std::string storeFile;
    };

    std::expected<std::string, std::string> RunOllama(std::string_view input)
    {
        int outputPipe[2];

        if (pipe(outputPipe) != 0)
            return std::unexpected(std::format("pipe() failed: {}", std::strerror(errno)));

        pid_t pid = fork();

        if (pid < 0)
        {
            close(outputPipe[0]);
            close(outputPipe[1]);
            return std::unexpected(std::format("fork() failed: {}", std::strerror(errno)));
        }

        if (pid == 0)
        {
            dup2(outputPipe[1], STDOUT_FILENO);
            dup2(outputPipe[1], STDERR_FILENO);
            close(outputPipe[0]);
            close(outputPipe[1]);

            std::string prompt(input);
            execlp(
                "ollama",
                "ollama",
                "run",
                kModelName.data(),
                "--dimensions",
                DIM_STR.c_str(),
                prompt.c_str(),
                static_cast<char *>(nullptr)
            );
            _exit(127);
        }

        close(outputPipe[1]);

        std::string output;
        std::array<char, 4096> buffer {};

        while (true)
        {
            ssize_t bytesRead = read(outputPipe[0], buffer.data(), buffer.size());

            if (bytesRead < 0)
            {
                if (errno == EINTR)
                    continue;

                close(outputPipe[0]);
                return std::unexpected(std::format("read() failed: {}", std::strerror(errno)));
            }

            if (bytesRead == 0)
                break;

            output.append(buffer.data(), static_cast<size_t>(bytesRead));
        }

        close(outputPipe[0]);

        int status = 0;

        if (waitpid(pid, &status, 0) < 0)
            return std::unexpected(std::format("waitpid() failed: {}", std::strerror(errno)));

        if (WIFSIGNALED(status))
            return std::unexpected(std::format("ollama terminated by signal {}:\n{}", WTERMSIG(status), output));

        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
            return std::unexpected(std::format("ollama exited with status {}:\n{}", WEXITSTATUS(status), output));

        return output;
    }

    bool IsAnsiSequenceChar(char ch) { return (ch >= '@' && ch <= '~'); }

    std::string StripAnsi(std::string_view text)
    {
        std::string stripped;
        stripped.reserve(text.size());

        for (size_t i = 0; i < text.size(); i++)
        {
            if (text[i] == '\x1b' && i + 1 < text.size() && text[i + 1] == '[')
            {
                i += 2;

                while (i < text.size() && !IsAnsiSequenceChar(text[i]))
                    i++;

                continue;
            }

            stripped += text[i];
        }

        return stripped;
    }

    std::expected<Vector, std::string> ParseEmbedding(std::string_view rawOutput)
    {
        std::string output = StripAnsi(rawOutput);
        size_t start = output.find('[');
        size_t end = output.rfind(']');

        if (start == std::string::npos || end == std::string::npos || end <= start)
            return std::unexpected(std::format("Could not find vector payload in ollama output:\n{}", output));

        auto decoded =
            Scoop::Vectors::DecodeVector<float, N_DIM>(std::string_view(output.data() + start + 1, end - start - 1));

        if (!decoded)
            return std::unexpected(std::format("Failed to decode ollama vector: {}", decoded.error()));

        return decoded.value();
    }

    std::expected<Vector, std::string> Embed(std::string_view input)
    {
        auto output = RunOllama(input);

        if (!output)
            return std::unexpected(output.error());

        return ParseEmbedding(*output);
    }

    void SaveStore(const Store &store, const std::filesystem::path &path)
    {
        std::ofstream file(path);

        if (!file.is_open())
            throw std::runtime_error(std::format("Failed to open '{}' for writing.", path.string()));

        file << store.Save();
    }

    uint64_t HashInputs(std::span<const std::string> inputs)
    {
        constexpr uint64_t kOffsetBasis = 1469598103934665603ull;
        constexpr uint64_t kPrime = 1099511628211ull;

        uint64_t hash = kOffsetBasis;

        for (const std::string &input : inputs)
        {
            for (unsigned char ch : input)
            {
                hash ^= ch;
                hash *= kPrime;
            }

            hash ^= '\n';
            hash *= kPrime;
        }

        return hash;
    }

    std::string ReadFile(const std::filesystem::path &path)
    {
        std::ifstream file(path);

        if (!file.is_open())
            throw std::runtime_error(std::format("Failed to open '{}' for reading.", path.string()));

        return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
    }

    void SaveManifest(const Manifest &manifest, const std::filesystem::path &path)
    {
        std::ofstream file(path);

        if (!file.is_open())
            throw std::runtime_error(std::format("Failed to open '{}' for writing.", path.string()));

        file << "model=" << manifest.model << '\n';
        file << "dimensions=" << manifest.dimensions << '\n';
        file << "input_count=" << manifest.inputCount << '\n';
        file << "input_hash=" << manifest.inputHash << '\n';
        file << "store_file=" << manifest.storeFile << '\n';
    }

    std::optional<Manifest> LoadManifest(const std::filesystem::path &path)
    {
        if (!std::filesystem::exists(path))
            return std::nullopt;

        std::ifstream file(path);

        if (!file.is_open())
            throw std::runtime_error(std::format("Failed to open '{}' for reading.", path.string()));

        Manifest manifest;
        std::string line;

        while (std::getline(file, line))
        {
            std::string_view trimmed = Trim(line);

            if (trimmed.empty())
                continue;

            size_t separator = trimmed.find('=');

            if (separator == std::string_view::npos)
                throw std::runtime_error(std::format("Invalid manifest line in '{}': '{}'.", path.string(), line));

            std::string key(trimmed.substr(0, separator));
            std::string value(trimmed.substr(separator + 1));

            if (key == "model")
                manifest.model = value;
            else if (key == "dimensions")
                manifest.dimensions = static_cast<size_t>(std::stoull(value));
            else if (key == "input_count")
                manifest.inputCount = static_cast<size_t>(std::stoull(value));
            else if (key == "input_hash")
                manifest.inputHash = std::stoull(value);
            else if (key == "store_file")
                manifest.storeFile = value;
        }

        if (manifest.model.empty() || manifest.dimensions == 0 || manifest.storeFile.empty())
            throw std::runtime_error(std::format("Manifest '{}' is missing required fields.", path.string()));

        return manifest;
    }

    bool IsManifestValid(
        const Manifest &manifest, std::span<const std::string> inputs, const std::filesystem::path &storePath
    )
    {
        return manifest.model == kModelName && manifest.dimensions == N_DIM && manifest.inputCount == inputs.size()
               && manifest.inputHash == HashInputs(inputs) && manifest.storeFile == storePath.filename().string();
    }

    std::expected<std::vector<Vector>, std::string> LoadVectors(const std::filesystem::path &path, size_t expectedCount)
    {
        std::ifstream file(path);

        if (!file.is_open())
            return std::unexpected(std::format("Failed to open '{}' for reading.", path.string()));

        std::vector<Vector> vectors;
        std::string line;

        while (std::getline(file, line))
        {
            std::string_view trimmed = Trim(line);

            if (trimmed.empty())
                continue;

            auto decoded = Scoop::Vectors::DecodeVector<float, N_DIM>(trimmed);

            if (!decoded)
                return std::unexpected(std::format("Failed to decode cached vector: {}", decoded.error()));

            vectors.push_back(*decoded);
        }

        if (vectors.size() != expectedCount)
        {
            return std::unexpected(
                std::format("Cached store has {} vectors, expected {}.", vectors.size(), expectedCount)
            );
        }

        return vectors;
    }

    Vector Mean(std::span<const Vector> vectors)
    {
        Vector out {};

        if (vectors.empty())
            return out;

        for (const Vector &vector : vectors)
            out += vector;

        return out / static_cast<float>(vectors.size());
    }

    Vector BlendNormalized(std::initializer_list<std::pair<Vector, float>> terms)
    {
        Vector out {};

        for (const auto &[vector, weight] : terms)
            out += vector.Normalized() * weight;

        return out.Normalize();
    }

    std::string_view Trim(std::string_view value)
    {
        size_t start = 0;
        size_t end = value.size();

        while (start < end && std::isspace(static_cast<unsigned char>(value[start])))
            start++;

        while (end > start && std::isspace(static_cast<unsigned char>(value[end - 1])))
            end--;

        return value.substr(start, end - start);
    }

    std::filesystem::path ResolveInputPath(const char *argv0)
    {
        std::vector<std::filesystem::path> candidates {
            std::filesystem::path(argv0).parent_path() / "inputs.txt",
            "examples/ollama-word-play/data/inputs.txt",
            "inputs.txt"
        };

        for (const auto &candidate : candidates)
        {
            if (std::filesystem::exists(candidate))
                return candidate;
        }

        throw std::runtime_error("Could not find inputs.txt. Pass a path as the second argument.");
    }

    std::vector<std::string> LoadInputs(const std::filesystem::path &path)
    {
        std::ifstream file(path);

        if (!file.is_open())
            throw std::runtime_error(std::format("Failed to open '{}' for reading.", path.string()));

        std::vector<std::string> inputs;
        std::unordered_set<std::string> seen;
        std::string line;

        while (std::getline(file, line))
        {
            std::string_view trimmed = Trim(line);

            if (trimmed.empty() || trimmed.starts_with('#'))
                continue;

            if (seen.emplace(trimmed).second)
                inputs.emplace_back(trimmed);
        }

        if (inputs.empty())
            throw std::runtime_error(std::format("No inputs found in '{}'.", path.string()));

        return inputs;
    }

    std::expected<std::vector<Vector>, std::string> EmbedInputs(std::span<const std::string> inputs)
    {
        if (inputs.empty())
            return {};

        const unsigned int hardwareThreads = std::thread::hardware_concurrency();

        const size_t workerCount = std::max<size_t>(
            1,
            std::min<size_t>(inputs.size(), hardwareThreads == 0 ? 4 : std::min(hardwareThreads, 12u))
        );

        std::vector<Vector> vectors(inputs.size());
        std::vector<std::string> errors(inputs.size());

        std::atomic<size_t> nextIndex = 0;
        std::atomic<bool> failed = false;
        std::mutex printMutex;

        std::vector<std::jthread> workers;
        workers.reserve(workerCount);

        {
            std::scoped_lock lock(printMutex);
            std::println(
                "Embedding {} inputs with `ollama run {} ...` using {} worker(s)",
                inputs.size(),
                kModelName,
                workerCount
            );
        }

        for (size_t worker = 0; worker < workerCount; worker++)
        {
            workers.emplace_back(
                [&]()
                {
                    while (!failed.load(std::memory_order_relaxed))
                    {
                        size_t index = nextIndex.fetch_add(1, std::memory_order_relaxed);

                        if (index >= inputs.size())
                            return;

                        {
                            std::scoped_lock lock(printMutex);
                            std::println("  - [{} / {}] {}", index + 1, inputs.size(), inputs[index]);
                        }

                        auto embedding = Embed(inputs[index]);

                        if (!embedding)
                        {
                            errors[index] = embedding.error();
                            failed.store(true, std::memory_order_relaxed);
                            return;
                        }

                        vectors[index] = *embedding;
                    }
                }
            );
        }

        workers.clear();

        for (size_t i = 0; i < errors.size(); i++)
        {
            if (!errors[i].empty())
                return std::unexpected(std::format("Embedding failed for '{}':\n{}", inputs[i], errors[i]));
        }

        return vectors;
    }

    float PhraseWeight(std::string_view label)
    {
        float weight = 1.0f;

        for (char ch : label)
        {
            if (ch == ' ')
                weight += 1.35f;
            else if (ch == '-')
                weight += 0.5f;
        }

        if (label.size() > 12)
            weight += 0.35f;

        if (label.size() > 18)
            weight += 0.35f;

        return weight;
    }

    std::string Describe(std::span<const WeightedLabel> terms)
    {
        std::string label;
        bool first = true;

        for (const auto &[term, weight] : terms)
        {
            float magnitude = std::abs(weight);

            if (first)
            {
                if (weight < 0.0f)
                    label += "-";

                first = false;
            }
            else
            {
                label += weight < 0.0f ? " - " : " + ";
            }

            if (magnitude != 1.0f)
                label += std::format("{:.1f} * ", magnitude);

            label += term;
        }

        label += " = ?";
        return label;
    }

    QueryRecipe BuildRecipe(const auto &find, std::span<const WeightedLabel> terms)
    {
        Vector vector {};
        std::vector<std::string_view> excludedLabels;
        excludedLabels.reserve(terms.size());

        for (const auto &[label, weight] : terms)
        {
            vector += find(label).Normalized() * weight;
            excludedLabels.push_back(label);
        }

        QueryRecipe recipe {
            .label = Describe(terms),
            .vector = vector.Normalized(),
            .excludedLabels = std::move(excludedLabels)
        };

        return recipe;
    }

    std::vector<QueryRecipe>
    GenerateExperiments(const std::vector<std::string> &labels, const auto &find, uint32_t seed)
    {
        if (labels.size() < 6)
            throw std::runtime_error("Need at least 6 unique inputs to generate random experiments.");

        std::mt19937 rng(seed);
        std::vector<double> labelWeights;
        labelWeights.reserve(labels.size());

        for (const std::string &label : labels)
            labelWeights.push_back(PhraseWeight(label));

        std::discrete_distribution<size_t> labelDistribution(labelWeights.begin(), labelWeights.end());
        std::uniform_int_distribution<int> termCountDistribution(4, 6);
        std::uniform_int_distribution<int> negativeCountDistribution(1, 2);
        std::uniform_real_distribution<float> positiveWeightDistribution(0.75f, 1.45f);
        std::uniform_real_distribution<float> negativeWeightDistribution(0.35f, 0.95f);

        auto sampleDistinctTerms = [&](int termCount)
        {
            std::vector<std::string_view> terms;
            std::unordered_set<std::string_view> seen;
            terms.reserve(static_cast<size_t>(termCount));

            while (static_cast<int>(terms.size()) < termCount)
            {
                std::string_view candidate = labels[labelDistribution(rng)];

                if (seen.emplace(candidate).second)
                    terms.push_back(candidate);
            }

            return terms;
        };

        std::vector<QueryRecipe> experiments;
        experiments.reserve(30);
        std::unordered_set<std::string> seenDescriptions;
        size_t attempts = 0;

        while (experiments.size() < 30 && attempts < 200)
        {
            attempts++;

            int termCount = termCountDistribution(rng);
            int negativeCount = std::min(termCount - 1, negativeCountDistribution(rng));
            int positiveCount = termCount - negativeCount;

            std::vector<std::string_view> sampledTerms = sampleDistinctTerms(termCount);
            std::vector<WeightedLabel> terms;

            terms.reserve(static_cast<size_t>(termCount));

            for (int i = 0; i < positiveCount; i++)
                terms.push_back({sampledTerms[static_cast<size_t>(i)], positiveWeightDistribution(rng)});

            for (int i = 0; i < negativeCount; i++)
                terms.push_back(
                    {sampledTerms[static_cast<size_t>(positiveCount + i)], -negativeWeightDistribution(rng)}
                );

            std::string description = Describe(terms);

            if (!seenDescriptions.emplace(description).second)
                continue;

            experiments.push_back(BuildRecipe(find, terms));
        }

        if (experiments.empty())
            throw std::runtime_error("Failed to generate random experiments from the corpus.");

        return experiments;
    }

    void PrintResults(
        const Store &store, std::span<const std::string> labels, std::string_view queryLabel, const Vector &query,
        std::span<const std::string_view> excludedLabels = {}
    )
    {
        std::vector<size_t> excludedIndices;
        excludedIndices.reserve(excludedLabels.size());

        for (std::string_view label : excludedLabels)
        {
            for (size_t i = 0; i < labels.size(); i++)
            {
                if (labels[i] == label)
                {
                    excludedIndices.push_back(i);
                    break;
                }
            }
        }

        auto results = store.Search(query, {.topK = 5, .scoreThreshold = -1.0f, .excludedIndices = excludedIndices});

        std::println("\n{}", queryLabel);

        for (const auto &result : results)
        {
            std::println(
                "  {:>2}. {:<30} score={:.4f}",
                &result - results.data() + 1,
                labels[result.index],
                result.score
            );
        }
    }
}

int main(int argc, char **argv)
{
    try
    {
        std::filesystem::path outputDir = argc > 1 ? argv[1] : "build/generated/ollama-word-play";
        std::filesystem::create_directories(outputDir);

        std::filesystem::path inputPath = argc > 2 ? argv[2] : ResolveInputPath(argv[0]);
        std::vector<std::string> inputs = LoadInputs(inputPath);

        std::filesystem::path storePath = outputDir / "store.csv";
        std::filesystem::path manifestPath = outputDir / "manifest.txt";

        Store store;
        store.Reserve(inputs.size());

        std::vector<std::string> labels = inputs;
        constexpr uint32_t kExperimentSeed = 1337;

        std::println("Loading inputs from {}", inputPath.string());
        std::vector<Vector> vectors;

        std::optional<Manifest> manifest = LoadManifest(manifestPath);

        if (manifest && std::filesystem::exists(storePath) && IsManifestValid(*manifest, inputs, storePath))
        {
            std::println("Reusing cached store from {}", storePath.string());
            auto cachedVectors = LoadVectors(storePath, inputs.size());

            if (!cachedVectors)
            {
                std::println("Cache read failed, rebuilding store:\n{}", cachedVectors.error());
                manifest.reset();
            }
            else
            {
                vectors = std::move(*cachedVectors);
            }
        }

        if (vectors.empty())
        {
            auto embedded = EmbedInputs(inputs);

            if (!embedded)
            {
                std::println(stderr, "\n{}", embedded.error());
                return 1;
            }

            vectors = std::move(*embedded);

            for (const Vector &vector : vectors)
                store.Add(vector);

            SaveStore(store, storePath);
            SaveManifest(
                Manifest {
                    .model = std::string(kModelName),
                    .dimensions = N_DIM,
                    .inputCount = inputs.size(),
                    .inputHash = HashInputs(inputs),
                    .storeFile = storePath.filename().string()
                },
                manifestPath
            );

            std::println("\nSaved store to {}", storePath.string());
            std::println("Saved manifest to {}", manifestPath.string());
        }
        else
        {
            for (const Vector &vector : vectors)
                store.Add(vector);
        }

        auto find = [&](std::string_view label) -> const Vector &
        {
            for (size_t i = 0; i < labels.size(); i++)
            {
                if (labels[i] == label)
                    return vectors[i];
            }

            throw std::runtime_error(std::format("Missing label '{}'.", label));
        };

        std::println("Generating experiments with seed {}", kExperimentSeed);
        std::vector<QueryRecipe> experiments = GenerateExperiments(labels, find, kExperimentSeed);

        for (const auto &experiment : experiments)
            PrintResults(store, labels, experiment.label, experiment.vector, experiment.excludedLabels);

        return 0;
    }
    catch (const std::exception &exception)
    {
        std::println(stderr, "{}", exception.what());
        return 1;
    }
}
