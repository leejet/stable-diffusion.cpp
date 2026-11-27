#include "dgl/bindings/video_windows.h"

#include "dgl/bindings/conversion.h"
#include "dgl/ops/video_windows.h"

namespace sd::dgl {
    Value split_video_frames(Values& args, RunState&) {
        std::vector<Value> windows;
        for (auto window : ops::split_video_frames(object<sd_tensor_t>(args, "frames"), size_t(get_int(args, "size")), size_t(get_int(args, "overlap"))))
            windows.push_back(Value::object(std::move(window)));
        return Value::list(std::move(windows), Kind::Tensor);
    }

    Value join_video_windows(Values& args, RunState&) {
        std::vector<std::shared_ptr<sd_tensor_t>> windows;
        for (const auto& window : args.at("windows").items)
            windows.push_back(window.as<sd_tensor_t>());
        return Value::object(ops::join_video_windows(windows, size_t(get_int(args, "overlap"))));
    }
}
