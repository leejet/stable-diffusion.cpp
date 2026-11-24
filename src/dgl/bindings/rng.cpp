#include "dgl/bindings/rng.h"

#include <utility>

#include "core/option_types.h"
#include "core/rng.hpp"
#include "core/rng_mt19937.hpp"
#include "core/rng_philox.hpp"

namespace sd::dgl {

    static std::shared_ptr<RNG> make_rng(rng_type_t type) {
        if (type == STD_DEFAULT_RNG)
            return std::make_shared<STDDefaultRNG>();
        if (type == CPU_RNG)
            return std::make_shared<MT19937RNG>();
        return std::make_shared<PhiloxRNG>();
    }

    Value execute_make_rng(Values& args, RunState&) {
        const auto algorithm = get_string(args, "algorithm");
        auto rng             = make_rng(algorithm == "std" ? STD_DEFAULT_RNG : str_to_rng_type(algorithm.c_str()));
        rng->manual_seed(get_int(args, "seed"));
        return Value::object(std::move(rng));
    }

}  // namespace sd::dgl
