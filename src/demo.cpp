#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <iomanip>
#include "resample_retrieval.h"
#include "helper.h"

namespace fs = std::filesystem;
constexpr size_t MAX_QUERY = 8;
constexpr size_t TOP_K = 3;

int main(int argc, char *argv[])
{
    std::string target = "target";
    std::string library = "lib";

    std::vector<std::string> library_files = jules::get_images("lib");
    std::cout << "library_files.size() = " << library_files.size() << std::endl;

    std::vector<std::string> target_files = jules::get_images("target");
    if (target_files.size() > MAX_QUERY)
    {
        target_files.resize(MAX_QUERY);
    }
    std::cout << "target_files.size() = " << target_files.size() << std::endl;

    jules::ResampleRetrieval retriever;

    for (auto model_type : {jules::ModelType::MOBILENET_V4_S, jules::ModelType::MOBILENET_V4_M, jules::ModelType::MOBILENET_V4_L})
    {
        if (0 != retriever.Init(model_type))
        {
            std::cerr << "Failed to initialize ResampleRetrieval with model type " << static_cast<size_t>(model_type) << std::endl;
            return -1;
        }

        auto start = std::chrono::high_resolution_clock::now();
        for (const auto &file : library_files)
        {
            retriever.Push(file);
        }
        auto finish = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start);
        std::cout << "Model " << static_cast<size_t>(model_type) << ", Indexed " << library_files.size()
                  << " images at " << static_cast<double>(elapsed.count()) / library_files.size() << " ms per image." << std::endl;

        std::vector<std::vector<jules::Result>> all_results;

        for (int t = 0; t < target_files.size(); t++)
        {
            std::vector<jules::Result> results;
            if (0 != retriever.Query(target_files[t], results, TOP_K))
            {
                std::cerr << "Failed to query image: " << target_files[t] << std::endl;
                continue;
            }
            all_results.emplace_back(std::move(results));
        }

        auto visualize_result = jules::visualize(all_results, target_files, TOP_K);
        std::string output_filename = "result-" + std::to_string(static_cast<size_t>(model_type)) + ".png";
        cv::imwrite(output_filename, visualize_result);
        std::cout << "Saved result table to " << output_filename << std::endl;
    }

    return 0;
}