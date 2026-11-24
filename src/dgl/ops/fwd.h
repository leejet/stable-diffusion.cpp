#ifndef __SD_DGL_OPS_FWD_H__
#define __SD_DGL_OPS_FWD_H__

#include "dgl/models/fwd.h"

namespace sd::dgl::ops {
    struct ApgPredictor;
    struct CfgPredictor;
    struct DetailCrop;
    struct DetailMask;
    struct DetectionFilter;
    enum class DetectionSort;
    struct GuidanceBranches;
    struct IPAdapterBinding;
    struct IPAdapterBindings;
    struct ModelPredictor;
    class PredictionExecution;
    struct PredictionExecutionOptions;
    struct Predictor;
    class PreviewObserver;
    class PuLIDAttention;
    struct PuLIDBinding;
    struct PuLIDBindings;
    class SamplingObservation;
    class SamplingObserver;
    class SamplingObserverSession;
    struct SamplingOptions;
    struct SamplingRun;
    struct SamplingSolver;
    struct SamplingStep;
    struct SigmaWindow;
    struct SkipLayerPredictor;
    struct StepWindow;
    struct SwitchPredictor;
}

#endif
