// Gemma-3 SentencePiece BPE tokenizer.
//
// Reads a raw `tokenizer.model` protobuf file (same format HuggingFace
// transformers and llama.cpp consume) and performs byte-level BPE encoding
// using the piece scores as merge priorities.
//
// SentencePiece vocab layout (for Gemma-3-12B):
//   262208 total pieces. First 4 are special control/unknown tokens
//   (<pad> id=0, <eos> id=1, <bos> id=2, <unk> id=3, <mask> id=4).
//   Most pieces are normal sub-word tokens; a small number are CONTROL
//   or USER_DEFINED (BOS/EOS/pad/mask/turn markers).
//
// For LTX-2.3, we tokenise the raw prompt with BOS prepended (Gemma
// convention) and EOS appended; we do NOT apply chat templates — LTX
// uses the Gemma base text encoder on raw text.

#ifndef __SD_TOKENIZERS_GEMMA3_TOKENIZER_H__
#define __SD_TOKENIZERS_GEMMA3_TOKENIZER_H__

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

class Gemma3Tokenizer {
public:
    enum TokenType : uint8_t {
        NORMAL       = 1,
        UNKNOWN      = 2,
        CONTROL      = 3,
        USER_DEFINED = 4,
        BYTE         = 5,
        UNUSED       = 6,
    };

    struct Piece {
        std::string text;
        float score = 0.0f;
        uint8_t type = NORMAL;
    };

    // Load vocab + scores from a SentencePiece protobuf (*.model) file.
    // Returns true on success. On failure, `error` holds a message.
    bool load_from_spm(const std::string& path, std::string* error = nullptr);

    // Encode `text` into token ids. If `add_bos` is true, prepends the BOS
    // id; if `add_eos`, appends EOS.
    //
    // Algorithm: byte-level pre-tokenization with the Gemma meta-space
    // prefix ("▁"), then BPE merges driven by piece scores. Highest-score
    // pair wins at each step.
    std::vector<int32_t> encode(const std::string& text,
                                bool add_bos = true,
                                bool add_eos = false) const;

    // Decoding is not required for LTX use, but trivial enough to expose.
    std::string decode(const std::vector<int32_t>& ids) const;

    int32_t bos_id() const { return bos_id_; }
    int32_t eos_id() const { return eos_id_; }
    int32_t pad_id() const { return pad_id_; }
    int32_t unk_id() const { return unk_id_; }
    int32_t vocab_size() const { return (int32_t)pieces_.size(); }

    const std::vector<Piece>& pieces() const { return pieces_; }

private:
    std::vector<Piece> pieces_;
    std::unordered_map<std::string, int32_t> piece_to_id_;
    int32_t bos_id_ = 2;
    int32_t eos_id_ = 1;
    int32_t pad_id_ = 0;
    int32_t unk_id_ = 3;

    // Encodes a single pre-tokenised word into the BPE sequence.
    void bpe_encode_word(const std::string& word, std::vector<int32_t>& out) const;
};

#endif  // __SD_TOKENIZERS_GEMMA3_TOKENIZER_H__
