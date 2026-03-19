#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>
#include <vector>
#include <unordered_map>
#include "resample_retrieval.h"
#include "helper.h"

namespace fs = std::filesystem;

static double get_config(const std::string &key, double default_val)
{
    std::ifstream infile("config.ini");
    if (!infile.is_open())
        return default_val;
    std::string line;
    while (std::getline(infile, line))
    {
        if (line.find(key) != std::string::npos)
        {
            auto pos = line.find('=');
            if (pos != std::string::npos)
            {
                try
                {
                    return std::stod(line.substr(pos + 1));
                }
                catch (...)
                {
                    return default_val;
                }
            }
        }
    }
    return default_val;
}

int main()
{
    double threshold = get_config("cluster_threshold", 0.6);
    double speed_up = get_config("speed_up", 0.0);
    std::cout << "Cluster threshold: " << threshold << std::endl;
    std::cout << "Speedup: " << speed_up << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    auto model_type = jules::ModelType::DINO_V2_VITB8;
    if (speed_up >= 1.0)
    {
        model_type = jules::ModelType::MOBILENET_V4_L;
    }
    else
        model_type = jules::ModelType::DINO_V2_VITB8;

    std::string images_dir = "images";
    if (!fs::exists(images_dir))
    {
        std::cerr << "Directory 'images' does not exist." << std::endl;
        return 1;
    }

    std::vector<std::string> images = jules::get_images(images_dir);
    if (images.empty())
    {
        std::cerr << "No images found in 'images' directory." << std::endl;
        return 1;
    }

    std::string clusters_dir = "clusters";
    fs::create_directories(clusters_dir);

    jules::ResampleRetrieval retriever;
    if (0 != retriever.Init(model_type))
    {
        std::cerr << "Failed to init model." << std::endl;
        return 1;
    }

    for (const auto &img : images)
    {
        retriever.Push(img);
    }

    std::unordered_map<std::string, bool> visited;
    for (const auto &img : images)
    {
        visited[img] = false;
    }

    int cluster_id = 0;
    for (const auto &img : images)
    {
        if (visited[img])
            continue;

        std::vector<jules::Result> results;
        if (0 != retriever.Query(img, results, images.size()))
        {
            std::cerr << "Query failed for " << img << std::endl;
            continue;
        }

        std::vector<std::string> cluster_members;
        for (const auto &res : results)
        {
            if (res.similarity >= threshold && !visited[res.path])
            {
                cluster_members.push_back(res.path);
                visited[res.path] = true;
            }
        }

        if (!cluster_members.empty())
        {
            std::string current_cluster_dir = clusters_dir + "/cluster_" + std::to_string(cluster_id);
            fs::create_directories(current_cluster_dir);
            for (const auto &member : cluster_members)
            {
                fs::copy_file(member, current_cluster_dir + "/" + fs::path(member).filename().string(), fs::copy_options::overwrite_existing);
            }
            cluster_id++;
        }
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    auto unit_cost = static_cast<double>(duration.count()) / images.size();
    unit_cost = std::round(unit_cost * 100) / 100.0;
    std::cout << "Clustered " << images.size() << " images into " << cluster_id << " clusters." << std::endl;
    std::cout << "Time taken: " << duration.count() << " seconds, thus " << unit_cost << " seconds per image." << std::endl;
    return 0;
}
