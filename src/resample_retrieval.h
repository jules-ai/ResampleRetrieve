#ifndef RESAMPLE_RETRIEVAL_H
#define RESAMPLE_RETRIEVAL_H
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <type_traits>

namespace jules
{

    typedef struct
    {
        std::string path;
        std::string name;
        double similarity;
    } Result;

    enum class ModelType : uint8_t
    {
        MOBILENET_V4_S = 0,
        MOBILENET_V4_M = 1,
        MOBILENET_V4_L = 2,
    };

    class ResampleRetrieval
    {
    public:
        ResampleRetrieval() = default;
        ~ResampleRetrieval() = default;

        int Init(ModelType model_type);
        int Push(const std::string &image_path);
        int Query(const std::string &image_path, std::vector<Result> &results, size_t topk = 3);

    private:
        cv::Mat extract_feature(cv::Mat &blob);
        cv::Mat prepare_input(const std::string &image_path);

        bool m_initialized{false};
        cv::dnn::Net m_dnn_net;

        const cv::Scalar m_model_input_scale{1.f / 0.229, 1.f / 0.224, 1.f / 0.225};
        const cv::Scalar m_model_input_mean{0.485 * 255.f, 0.456 * 255.f, 0.406 * 255.f};

        int m_model_input_size;
        int m_model_resize_size;
        std::vector<std::string> m_lib_image_paths;
        std::vector<cv::Mat> m_lib_features;
    };

}

#endif // RESAMPLE_RETRIEVAL_H
