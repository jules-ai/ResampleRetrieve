#include "resample_retrieval.h"
#include <unordered_map>
#include <filesystem>
#include <fstream>
#include <iostream>
namespace fs = std::filesystem;

namespace std
{
    template <>
    struct hash<jules::ModelType>
    {
        size_t operator()(jules::ModelType t) const noexcept
        {
            return static_cast<size_t>(t);
        }
    };
}
namespace jules
{
    const std::unordered_map<ModelType, std::pair<std::string, int>, std::hash<ModelType>> allowed_models =
        {
            {ModelType::MOBILENET_V4_S, {"mobilenetv4_s", 224}},
            {ModelType::MOBILENET_V4_M, {"mobilenetv4_m", 384}},
            {ModelType::MOBILENET_V4_L, {"mobilenetv4_l", 384}},
    };

    int ResampleRetrieval::Init(ModelType model_type)
    {
        std::string model_path = "models/" + allowed_models.at(model_type).first + ".Opset17.onnx";

        if (!fs::exists(model_path))
        {
            std::cerr << "model not exist: " << model_path << std::endl;
            return -1;
        }

        m_model_input_size = allowed_models.at(model_type).second;
        m_model_resize_size = static_cast<int>(std::round(m_model_input_size * 0.925 + 48.8));
        m_dnn_net = cv::dnn::readNetFromONNX(model_path);
        m_lib_image_paths.clear();
        m_lib_features.clear();
        std::cout << "model init done: " << m_model_input_size << ", " << m_model_resize_size << std::endl;

        m_initialized = true;
        return 0;
    }

    int ResampleRetrieval::Push(const std::string &image_path)
    {
        if (!m_initialized)
        {
            std::cerr << "model not initialized!" << std::endl;
            return -1;
        }
        auto blob = prepare_input(image_path);
        auto feature = extract_feature(blob).clone();
        m_lib_features.push_back(feature);
        m_lib_image_paths.push_back(image_path);

        return 0;
    }

    int ResampleRetrieval::Query(const std::string &image_path, std::vector<Result> &results, size_t topk)
    {
        if (!m_initialized)
        {
            std::cerr << "model not initialized!" << std::endl;
            return -1;
        }
        if (topk == 0 || topk > m_lib_features.size())
        {
            std::cerr << "topk should be in [1, " << m_lib_features.size() << "]" << std::endl;
            return -1;
        }
        results.resize(m_lib_features.size());

        auto blob = prepare_input(image_path);
        auto query_feature = extract_feature(blob);

        for (size_t i = 0; i < m_lib_features.size(); ++i)
        {
            double similarity = query_feature.dot(m_lib_features[i]);
            results[i] = {m_lib_image_paths[i], fs::path(m_lib_image_paths[i]).filename().string(), similarity};
        }

        std::partial_sort(results.begin(), results.begin() + topk, results.end(), [](const Result &a, const Result &b)
                          { return a.similarity > b.similarity; });

        results.resize(topk);
        return 0;
    }

    cv::Mat ResampleRetrieval::extract_feature(cv::Mat &blob)
    {
        m_dnn_net.setInput(blob);
        auto feature = m_dnn_net.forward();
        feature = feature.reshape(1, 1);
        feature /= cv::norm(feature);
        return feature;
    }
    cv::Mat ResampleRetrieval::prepare_input(const std::string &image_path)
    {
        auto image = cv::imread(image_path);

        double image_height = image.rows;
        double image_width = image.cols;
        auto scale = static_cast<double>(m_model_resize_size) / std::min(image_height, image_width);

        int new_height = static_cast<int>(std::round(image_height * scale));
        int new_width = static_cast<int>(std::round(image_width * scale));

        cv::Mat resized;
        cv::resize(image, resized, cv::Size(new_width, new_height), 0, 0, cv::INTER_CUBIC);

        int y0 = (new_height - m_model_input_size) / 2;
        int x0 = (new_width - m_model_input_size) / 2;
        cv::Rect roi(x0, y0, m_model_input_size, m_model_input_size);

        cv::Mat blob = cv::dnn::blobFromImage(resized(roi), 1.0 / 255.0, cv::Size(m_model_input_size, m_model_input_size), m_model_input_mean, true, false, CV_32F);

        float *data = reinterpret_cast<float *>(blob.data);
        int channelSize = m_model_input_size * m_model_input_size;
        for (int c = 0; c < 3; ++c)
        {
            cv::Mat channel_mat(1, channelSize, CV_32F, data + c * channelSize);
            channel_mat.convertTo(channel_mat, -1, m_model_input_scale[c]);
        }

        return blob;
    }
}
