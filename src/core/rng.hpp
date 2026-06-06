#ifndef __SD_CORE_RNG_HPP__
#define __SD_CORE_RNG_HPP__

#include <random>
#include <vector>

/**
 * @brief Abstract base class for random number generators (RNG).
 *
 * This class defines the interface for random number generation
 * that is required by the stable diffusion framework. All RNG
 * implementations must inherit from this class and implement
 * the pure virtual methods.
 */
class RNG {
public:
    virtual void manual_seed(uint64_t seed)      = 0;
    virtual std::vector<float> randn(uint32_t n) = 0;
};

/**
 * @brief Default random number generator implementation using C++ standard library.
 *
 * This class provides a concrete implementation of the RNG interface
 * using the standard library's default random engine. It generates
 * random numbers with a normal distribution (mean=0, stddev=1) which
 * is commonly used in diffusion models.
 *
 * @see RNG
 */
class STDDefaultRNG : public RNG {
private:
    std::default_random_engine generator;

public:
    void manual_seed(uint64_t seed) override {
        generator.seed((unsigned int)seed);
    }

    std::vector<float> randn(uint32_t n) override {
        std::vector<float> result;
        float mean   = 0.0;
        float stddev = 1.0;
        std::normal_distribution<float> distribution(mean, stddev);
        for (uint32_t i = 0; i < n; i++) {
            float random_number = distribution(generator);
            result.push_back(random_number);
        }
        return result;
    }
};

#endif  // __SD_CORE_RNG_HPP__