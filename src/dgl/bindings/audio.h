#ifndef __SD_DGL_BINDINGS_AUDIO_H__
#define __SD_DGL_BINDINGS_AUDIO_H__

#include "dgl/ops/audio.h"
#include "dgl/registry.h"

namespace sd::dgl {
    OutputSignature audio_signature();
    Value audio_result(ops::AudioReadResult audio);
    ops::AudioReadResult read_audio_dict(const Value& value, bool allow_empty = false);
    Value execute_load_audio(Values& args, RunState&);
    Value execute_save_audio(Values& args, RunState&);
}  // namespace sd::dgl

#endif  // __SD_DGL_BINDINGS_AUDIO_H__
