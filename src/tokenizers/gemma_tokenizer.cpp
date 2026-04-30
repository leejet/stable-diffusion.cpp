#include "gemma_tokenizer.h"

#include <climits>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

#include "json.hpp"
#include "util.h"

namespace {

// Parse "<0xAB>" -> byte value 0xAB. Returns -1 if not a byte token.
int parse_byte_token(const std::string& piece) {
    if (piece.size() != 6) return -1;
    if (piece[0] != '<' || piece[1] != '0' || piece[2] != 'x' || piece[5] != '>') return -1;
    auto hex = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'A' && c <= 'F') return 10 + c - 'A';
        if (c >= 'a' && c <= 'f') return 10 + c - 'a';
        return -1;
    };
    int hi = hex(piece[3]), lo = hex(piece[4]);
    if (hi < 0 || lo < 0) return -1;
    return (hi << 4) | lo;
}

}  // namespace

GemmaTokenizer::GemmaTokenizer() {
    byte_fallback_ids_.fill(-1);
    BOS_TOKEN = "<bos>";
    EOS_TOKEN = "<eos>";
    PAD_TOKEN = "<pad>";
    UNK_TOKEN = "<unk>";
    BOS_TOKEN_ID = 2;
    EOS_TOKEN_ID = 1;
    PAD_TOKEN_ID = 0;
    UNK_TOKEN_ID = 3;
    add_bos_token = true;   // Gemma post-processor prepends <bos>.
    add_eos_token = false;
    pad_left      = true;   // Gemma uses left padding.
}

std::string GemmaTokenizer::decode_token(int token_id) const {
    if (token_id >= 0 && token_id < (int)id_to_piece_.size()) {
        return id_to_piece_[token_id];
    }
    return "";
}

// HF normalizer: Replace " " -> "▁" (U+2581). All other chars untouched.
std::string GemmaTokenizer::normalize(const std::string& text) const {
    static const std::string metaspace = "\xe2\x96\x81";  // UTF-8 for U+2581
    std::string out;
    out.reserve(text.size() + text.size() / 8);
    for (char c : text) {
        if (c == ' ') {
            out.append(metaspace);
        } else {
            out.push_back(c);
        }
    }
    return out;
}

std::vector<std::string> GemmaTokenizer::split_utf8_chars(const std::string& s) {
    std::vector<std::string> out;
    size_t i = 0;
    while (i < s.size()) {
        unsigned char b = (unsigned char)s[i];
        size_t len;
        if (b < 0x80)        len = 1;
        else if (b < 0xC0)   len = 1;  // malformed continuation; treat as 1-byte
        else if (b < 0xE0)   len = 2;
        else if (b < 0xF0)   len = 3;
        else                 len = 4;
        if (i + len > s.size()) len = s.size() - i;
        out.emplace_back(s.substr(i, len));
        i += len;
    }
    return out;
}

void GemmaTokenizer::byte_fallback(const std::string& ch, std::vector<std::string>& out) const {
    for (unsigned char b : ch) {
        int id = byte_fallback_ids_[b];
        if (id >= 0) {
            out.push_back(id_to_piece_[id]);  // "<0xNN>"
        } else {
            out.push_back(UNK_TOKEN);
        }
    }
}

std::vector<std::string> GemmaTokenizer::bpe(std::vector<std::string> pieces) const {
    // Greedy BPE: at each step find the adjacent pair with lowest merge rank and apply it.
    // O(N^2 * merges_lookup) per encode. N here is chars in a single chunk — a few hundred
    // at most for our use. Good enough.
    while (pieces.size() > 1) {
        int best_rank = INT_MAX;
        int best_i    = -1;
        for (size_t i = 0; i + 1 < pieces.size(); ++i) {
            std::string key = pieces[i];
            key.push_back('\t');
            key.append(pieces[i + 1]);
            auto it = merge_ranks_.find(key);
            if (it != merge_ranks_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_i    = (int)i;
            }
        }
        if (best_i < 0) break;
        pieces[best_i] = pieces[best_i] + pieces[best_i + 1];
        pieces.erase(pieces.begin() + best_i + 1);
    }
    return pieces;
}

std::vector<int> GemmaTokenizer::encode(const std::string& text, on_new_token_cb_t /*cb*/) {
    if (!loaded_) {
        LOG_ERROR("GemmaTokenizer::encode called before load_from_file()");
        return {};
    }

    std::string normalized = normalize(text);

    // ignore_merges=true: if the entire (post-normalization) chunk is directly in vocab,
    // emit it as a single token without running BPE.
    if (ignore_merges_) {
        auto it = vocab_.find(normalized);
        if (it != vocab_.end()) {
            return {it->second};
        }
    }

    std::vector<std::string> pieces;
    pieces.reserve(normalized.size());
    for (const auto& ch : split_utf8_chars(normalized)) {
        auto it = vocab_.find(ch);
        if (it != vocab_.end()) {
            pieces.push_back(ch);
        } else {
            byte_fallback(ch, pieces);
        }
    }

    pieces = bpe(std::move(pieces));

    std::vector<int> ids;
    ids.reserve(pieces.size());
    for (const auto& p : pieces) {
        auto it = vocab_.find(p);
        if (it != vocab_.end()) {
            ids.push_back(it->second);
        } else {
            ids.push_back(UNK_TOKEN_ID);
        }
    }
    return ids;
}

bool GemmaTokenizer::load_from_file(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        LOG_ERROR("GemmaTokenizer: cannot open %s", path.c_str());
        return false;
    }
    nlohmann::json j;
    try {
        f >> j;
    } catch (const nlohmann::json::parse_error& e) {
        LOG_ERROR("GemmaTokenizer: JSON parse error in %s: %s", path.c_str(), e.what());
        return false;
    }

    if (!j.contains("model") || !j["model"].contains("vocab") || !j["model"].contains("merges")) {
        LOG_ERROR("GemmaTokenizer: JSON missing model.vocab or model.merges");
        return false;
    }
    const auto& model = j["model"];

    // Vocab: HF tokenizer.json for BPE stores vocab as an object {piece: id}.
    const auto& vocab = model["vocab"];
    id_to_piece_.clear();
    id_to_piece_.resize(vocab.size());
    vocab_.reserve(vocab.size() * 2);
    for (auto it = vocab.begin(); it != vocab.end(); ++it) {
        const std::string piece = it.key();
        int id                  = it.value().get<int>();
        vocab_.emplace(piece, id);
        if (id >= 0 && id < (int)id_to_piece_.size()) {
            id_to_piece_[id] = piece;
        }
    }

    // Merges: ordered list; earlier entries have higher priority (lower rank).
    const auto& merges = model["merges"];
    merge_ranks_.reserve(merges.size() * 2);
    int rank = 0;
    for (const auto& m : merges) {
        // tokenizers >=0.20 stores each merge as a [left, right] array; older versions used a
        // single space-separated string. Accept both for robustness.
        std::string left, right;
        if (m.is_array() && m.size() == 2) {
            left  = m[0].get<std::string>();
            right = m[1].get<std::string>();
        } else if (m.is_string()) {
            const std::string s = m.get<std::string>();
            auto pos            = s.find(' ');
            if (pos == std::string::npos) continue;
            left  = s.substr(0, pos);
            right = s.substr(pos + 1);
        } else {
            continue;
        }
        std::string key = left;
        key.push_back('\t');
        key.append(right);
        merge_ranks_.emplace(std::move(key), rank++);
    }

    // Locate byte-fallback IDs. Every byte value 0..255 should have a "<0xNN>" entry.
    for (int id = 0; id < (int)id_to_piece_.size(); ++id) {
        int b = parse_byte_token(id_to_piece_[id]);
        if (b >= 0) {
            byte_fallback_ids_[b] = id;
        }
    }

    // Special token IDs: honor what's actually in the JSON if model is unusual; otherwise
    // keep the Gemma-3 defaults from the ctor.
    if (model.contains("unk_token") && model["unk_token"].is_string()) {
        auto it = vocab_.find(model["unk_token"].get<std::string>());
        if (it != vocab_.end()) UNK_TOKEN_ID = it->second;
    }
    if (j.contains("added_tokens")) {
        for (const auto& at : j["added_tokens"]) {
            if (!at.contains("content") || !at.contains("id")) continue;
            const std::string c = at["content"].get<std::string>();
            int id              = at["id"].get<int>();
            if (c == "<bos>") BOS_TOKEN_ID = id;
            else if (c == "<eos>") EOS_TOKEN_ID = id;
            else if (c == "<pad>") PAD_TOKEN_ID = id;
            else if (c == "<unk>") UNK_TOKEN_ID = id;
        }
    }

    ignore_merges_ = model.value("ignore_merges", true);
    loaded_        = true;
    LOG_DEBUG("GemmaTokenizer loaded: vocab=%zu merges=%zu", vocab_.size(), merge_ranks_.size());
    return true;
}
