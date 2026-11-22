#ifndef __SD_DGL_MODELS_CIRCULAR_AXES_H__
#define __SD_DGL_MODELS_CIRCULAR_AXES_H__

#include <functional>

namespace sd::dgl::models {
    class CircularAxesScope {
        std::function<void()> restore_;

    public:
        template <class T>
        CircularAxesScope(T& model, bool x, bool y) {
            const bool old_x = model.circular_x;
            const bool old_y = model.circular_y;
            if (old_x != x || old_y != y) {
                restore_ = [&model, old_x, old_y] { model.apply_circular_axes(old_x, old_y); };
                model.apply_circular_axes(x, y);
            }
        }

        ~CircularAxesScope() {
            if (restore_)
                restore_();
        }
        CircularAxesScope(const CircularAxesScope&)            = delete;
        CircularAxesScope& operator=(const CircularAxesScope&) = delete;
    };
}

#endif  // __SD_DGL_MODELS_CIRCULAR_AXES_H__
