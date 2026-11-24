#ifndef __SD_DGL_MODELS_FWD_H__
#define __SD_DGL_MODELS_FWD_H__

namespace sd::dgl::models {
    class AnimateDiff;
    class AudioEncoder;
    class AudioVAE;
    class CircularAxesScope;
    struct ControlBinding;
    struct ControlBindings;
    class ControlNet;
    struct DiffusionModel;
    struct DetailDetection;
    struct DetailDetector;
    struct DetailDetectorParams;
    struct DetectionOptions;
    class DiT;
    class DenoisingParameterization;
    struct DiTLoadParams;
    struct DiffusionPrediction;
    struct DiffusionPredictionOptions;
    struct DiffusionPredictionStep;
    class IPAdapter;
    class ImageUpscaleModel;
    struct ImageUpscaler;
    struct ImageUpscalerParams;
    class LatentUpscaler;
    struct LoraBinding;
    struct LoraConfig;
    class LoraExecution;
    struct LoraFile;
    class Model;
    struct ModelExecutionOptions;
    struct ModelLoadParams;
    class ModelRuntime;
    enum class OperatorMode;
    class PhotoMaker;
    struct PhotoMakerPrompt;
    struct PredictionParameters;
    class PuLID;
    class TextEncoder;
    struct TextEncoderLoadParams;
    struct UNetControls;
    class VAE;
    struct VAELoadParams;
    class VisionEncoder;
}

#endif
