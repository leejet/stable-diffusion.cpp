#ifndef IMATRIX_HPP
#define IMATRIX_HPP
#include <fstream>
#include <mutex>
#include <unordered_map>
#include <string>

/*Stolen from llama.cpp (credits: Kawrakow)*/

struct Stats {
    std::vector<float> values{};
    std::vector<int> counts{};
    int ncall = 0;
};

class IMatrixCollector {
public:
    IMatrixCollector() = default;
    bool collect_imatrix(struct ggml_tensor* t, bool ask, void* user_data);
    void save_imatrix(std::string fname, int ncall = -1) const;
    bool load_imatrix(const char* fname);
    std::vector<float> get_values(const std::string& key) const {
        auto it = m_stats.find(key);
        if (it != m_stats.end()) {
            return it->second.values;
        } else {
            return {};
        }
    }
private:
    std::unordered_map<std::string, Stats> m_stats = {};
    std::mutex m_mutex;
    int m_last_call = 0;
    std::vector<float> m_src1_data;
    std::vector<char> m_ids;  // the expert ids from ggml_mul_mat_id
};

#endif