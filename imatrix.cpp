#include "imatrix.hpp"

/*Stolen from llama.cpp (credits: Kawrakow)*/

#include "ggml-backend.h"
#include "ggml.h"
#include "util.h"

#include <cmath>

// remove any prefix and suffixes from the name
// CUDA0#blk.0.attn_k.weight#0 => blk.0.attn_k.weight
static std::string filter_tensor_name(const char* name) {
    std::string wname;
    const char* p = strchr(name, '#');
    if (p != NULL) {
        p             = p + 1;
        const char* q = strchr(p, '#');
        if (q != NULL) {
            wname = std::string(p, q - p);
        } else {
            wname = p;
        }
    } else {
        wname = name;
    }
    return wname;
}

bool IMatrixCollector::collect_imatrix(struct ggml_tensor* t, bool ask, void* user_data) {
    GGML_UNUSED(user_data);
    const struct ggml_tensor* src0 = t->src[0];
    const struct ggml_tensor* src1 = t->src[1];
    std::string wname              = filter_tensor_name(src0->name);

    // when ask is true, the scheduler wants to know if we are interested in data from this tensor
    // if we return true, a follow-up call will be made with ask=false in which we can do the actual collection
    if (ask) {
        if (t->op == GGML_OP_MUL_MAT_ID)
            return true;  // collect all indirect matrix multiplications
        if (t->op != GGML_OP_MUL_MAT)
            return false;
        // why are small batches ignored (<16 tokens)?
        // if (src1->ne[1] < 16 || src1->type != GGML_TYPE_F32) return false;
        if (!(wname.substr(0, 6) == "model." || wname.substr(0, 17) == "cond_stage_model." || wname.substr(0,14) == "text_encoders."))
            return false;
        return true;
    }
    // LOG_DEBUG("%s", wname.c_str());

    std::lock_guard<std::mutex> lock(m_mutex);

    // copy the data from the GPU memory if needed
    const bool is_host = ggml_backend_buffer_is_host(src1->buffer);

    if (!is_host) {
        m_src1_data.resize(ggml_nelements(src1));
        ggml_backend_tensor_get(src1, m_src1_data.data(), 0, ggml_nbytes(src1));
    }

    const float* data = is_host ? (const float*)src1->data : m_src1_data.data();

    // this has been adapted to the new format of storing merged experts in a single 3d tensor
    // ref: https://github.com/ggml-org/llama.cpp/pull/6387
    if (t->op == GGML_OP_MUL_MAT_ID) {
        //   ids  -> [n_experts_used, n_tokens]
        //   src1 -> [cols, n_expert_used, n_tokens]
        const ggml_tensor* ids = t->src[2];
        const int n_as         = src0->ne[2];
        const int n_ids        = ids->ne[0];

        // the top-k selected expert ids are stored in the ids tensor
        // for simplicity, always copy ids to host, because it is small
        // take into account that ids is not contiguous!

        GGML_ASSERT(ids->ne[1] == src1->ne[2]);

        m_ids.resize(ggml_nbytes(ids));
        ggml_backend_tensor_get(ids, m_ids.data(), 0, ggml_nbytes(ids));

        auto& e = m_stats[wname];

        ++e.ncall;

        if (e.values.empty()) {
            e.values.resize(src1->ne[0] * n_as, 0);
            e.counts.resize(src1->ne[0] * n_as, 0);
        } else if (e.values.size() != (size_t)src1->ne[0] * n_as) {
            LOG_ERROR("inconsistent size for %s (%d vs %d)\n", wname.c_str(), (int)e.values.size(), (int)src1->ne[0] * n_as);
            exit(1);  // GGML_ABORT("fatal error");
        }
        // LOG_DEBUG("%s[%d]: %32s, %s, %5d x %5d, %d\n", m_last_call, wname.c_str(), ggml_op_name(t->op), (int)src1->ne[0], (int)src1->ne[2], (int)src1->type);
        // loop over all possible experts, regardless if they are used or not in the batch
        for (int ex = 0; ex < n_as; ++ex) {
            size_t e_start = ex * src1->ne[0];

            for (int idx = 0; idx < n_ids; ++idx) {
                for (int row = 0; row < (int)src1->ne[2]; ++row) {
                    const int excur = *(const int32_t*)(m_ids.data() + row * ids->nb[1] + idx * ids->nb[0]);

                    GGML_ASSERT(excur >= 0 && excur < n_as);  // sanity check

                    if (excur != ex)
                        continue;

                    const int64_t i11 = idx % src1->ne[1];
                    const int64_t i12 = row;
                    const float* x    = (const float*)((const char*)data + i11 * src1->nb[1] + i12 * src1->nb[2]);

                    for (int j = 0; j < (int)src1->ne[0]; ++j) {
                        e.values[e_start + j] += x[j] * x[j];
                        e.counts[e_start + j]++;
                        if (!std::isfinite(e.values[e_start + j])) {
                            printf("\n");
                            LOG_ERROR("%f detected in %s\n", e.values[e_start + j], wname.c_str());
                            exit(1);
                        }
                    }
                }
            }
        }
    } else {
        auto& e = m_stats[wname];
        if (e.values.empty()) {
            e.values.resize(src1->ne[0], 0);
            e.counts.resize(src1->ne[0], 0);
        } else if (e.values.size() != (size_t)src1->ne[0]) {
            LOG_WARN("inconsistent size for %s (%d vs %d)\n", wname.c_str(), (int)e.values.size(), (int)src1->ne[0]);
            exit(1);  // GGML_ABORT("fatal error");
        }

        ++e.ncall;
        // LOG_DEBUG("%s[%d]: %32s, %s, %5d x %5d, %d\n", m_last_call, wname.c_str(), ggml_op_name(t->op), (int)src1->ne[0], (int)src1->ne[1], (int)src1->type);
        for (int row = 0; row < (int)src1->ne[1]; ++row) {
            const float* x = data + row * src1->ne[0];
            for (int j = 0; j < (int)src1->ne[0]; ++j) {
                e.values[j] += x[j] * x[j];
                e.counts[j]++;
                if (!std::isfinite(e.values[j])) {
                    LOG_WARN("%f detected in %s\n", e.values[j], wname.c_str());
                    exit(1);
                }
            }
        }
    }
    return true;

}

void IMatrixCollector::save_imatrix(std::string fname,int ncall) const {
    LOG_INFO("SAVING_IMATRIX to %s\n", fname.c_str());

    if (ncall > 0) {
        fname += ".at_";
        fname += std::to_string(ncall);
    }
    // avoid writing imatrix entries that do not have full data
    // this can happen with MoE models where some of the experts end up not being exercised by the provided training data

    int n_entries = 0;
    std::vector<std::string> to_store;

    bool is_first = true;  // for printing
    for (const auto& kv : m_stats) {
        const int n_all = kv.second.counts.size();

        if (n_all == 0) {
            continue;
        }

        int n_zeros = 0;
        for (const int c : kv.second.counts) {
            if (c == 0) {
                n_zeros++;
            }
        }

        if (n_zeros != 0 && is_first) {
            printf("\n");
            is_first = false;
        }

        if (n_zeros == n_all) {
            LOG_WARN("entry '%40s' has no data - skipping\n", kv.first.c_str());
            continue;
        }

        if (n_zeros > 0) {
            LOG_WARN("entry '%40s' has partial data (%.2f%%) - skipping\n", kv.first.c_str(), 100.0f * (n_all - n_zeros) / n_all);
            continue;
        }

        n_entries++;
        to_store.push_back(kv.first);
    }

    if (to_store.size() < m_stats.size()) {
        LOG_WARN("storing only %zu out of %zu entries\n", to_store.size(), m_stats.size());
    }

    std::ofstream out(fname, std::ios::binary);
    out.write((const char*)&n_entries, sizeof(n_entries));
    for (const auto& name : to_store) {
        const auto& stat = m_stats.at(name);
        int len          = name.size();
        out.write((const char*)&len, sizeof(len));
        out.write(name.c_str(), len);
        out.write((const char*)&stat.ncall, sizeof(stat.ncall));
        int nval = stat.values.size();
        out.write((const char*)&nval, sizeof(nval));
        if (nval > 0) {
            std::vector<float> tmp(nval);
            for (int i = 0; i < nval; i++) {
                tmp[i] = (stat.values[i] / static_cast<float>(stat.counts[i])) * static_cast<float>(stat.ncall);
            }
            out.write((const char*)tmp.data(), nval * sizeof(float));
        }
    }

    // Write the number of call the matrix was computed with
    out.write((const char*)&m_last_call, sizeof(m_last_call));

    // LOG_DEBUG("\n");
    // LOG_DEBUG("stored collected data after %d chunks in %s\n", m_last_call, fname.c_str());
}

bool IMatrixCollector::load_imatrix(const char* fname) {
    std::ifstream in(fname, std::ios::binary);
    if (!in) {
        LOG_ERROR("failed to open %s\n", fname);
        return false;
    }
    int n_entries;
    in.read((char*)&n_entries, sizeof(n_entries));
    if (in.fail() || n_entries < 1) {
        LOG_ERROR("no data in file %s\n", fname);
        return false;
    }
    for (int i = 0; i < n_entries; ++i) {
        int len;
        in.read((char*)&len, sizeof(len));
        std::vector<char> name_as_vec(len + 1);
        in.read((char*)name_as_vec.data(), len);
        if (in.fail()) {
            LOG_ERROR("failed reading name for entry %d from %s\n", i + 1, fname);
            return false;
        }
        name_as_vec[len] = 0;
        std::string name{name_as_vec.data()};
        auto& e = m_stats[std::move(name)];
        int ncall;
        in.read((char*)&ncall, sizeof(ncall));
        int nval;
        in.read((char*)&nval, sizeof(nval));
        if (in.fail() || nval < 1) {
            LOG_ERROR("failed reading number of values for entry %d\n", i);
            m_stats = {};
            return false;
        }

        if (e.values.empty()) {
            e.values.resize(nval, 0);
            e.counts.resize(nval, 0);
        }

        std::vector<float> tmp(nval);
        in.read((char*)tmp.data(), nval * sizeof(float));
        if (in.fail()) {
            LOG_ERROR("failed reading data for entry %d\n", i);
            m_stats = {};
            return false;
        }

        // Recreate the state as expected by save_imatrix(), and correct for weighted sum.
        for (int i = 0; i < nval; i++) {
            e.values[i] += tmp[i];
            e.counts[i] += ncall;
        }
        e.ncall += ncall;
    }
    return true;
}