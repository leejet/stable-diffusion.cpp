#ifndef __SD_TOKENIZERS_GEMMA_TOKENIZER_H__
#define __SD_TOKENIZERS_GEMMA_TOKENIZER_H__

#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "tokenizer.h"

// Gemma 3 tokenizer. BPE with byte-fallback + Metaspace-style normalization
// (space → U+2581 "▁"). Loads a HuggingFace tokenizer.json produced by
// `AutoTokenizer.from_pretrained("google/gemma-3-12b-it").backend_tokenizer.save()`.
//
// Not embeddable as a header like the other tokenizers — the raw JSON is ~33 MB
// and the vocab alone is 262144 pieces plus 514906 merges. Expected workflow:
// ship the tokenizer.json file alongside the weights, pass its path at runtime.
class GemmaTokenizer : public Tokenizer {
protected:
    std::unordered_map<std::string, int> vocab_;      // piece -> id
    std::vector<std::string> id_to_piece_;            // id -> piece
    std::unordered_map<std::string, int> merge_ranks_;  // "left\tright" -> rank (lower = higher priority)
    std::array<int, 256> byte_fallback_ids_{};        // byte value -> piece id for <0xXX>
    bool loaded_ = false;
    bool ignore_merges_ = true;

    std::string decode_token(int token_id) const override;
    std::string normalize(const std::string& text) const override;

    // Split a UTF-8 string into its individual code-point-sized chunks.
    static std::vector<std::string> split_utf8_chars(const std::string& s);

    // Byte-fallback a character that isn't in vocab: produce UTF-8 byte tokens.
    void byte_fallback(const std::string& ch, std::vector<std::string>& out) const;

    // Run BPE merging until no more merges apply.
    std::vector<std::string> bpe(std::vector<std::string> pieces) const;

public:
    GemmaTokenizer();

    bool load_from_file(const std::string& path);
    bool is_loaded() const { return loaded_; }
    int vocab_size() const { return (int)id_to_piece_.size(); }

    std::vector<int> encode(const std::string& text, on_new_token_cb_t on_new_token_cb = nullptr) override;
};

#endif  // __SD_TOKENIZERS_GEMMA_TOKENIZER_H__
