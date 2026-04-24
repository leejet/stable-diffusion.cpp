// Gemma-3 SentencePiece BPE tokenizer — implementation.
//
// Protobuf wire format for sentencepiece.ModelProto (only the fields we
// care about):
//   message ModelProto {
//     repeated SentencePiece pieces = 1;
//     // ... unused fields
//   }
//   message SentencePiece {
//     string  piece = 1;
//     float   score = 2;
//     Type    type  = 3;  // enum, wire-type = varint
//   }
//
// We parse exactly this subset — everything else (trainer_spec,
// normalizer_spec, etc.) is skipped via tag/length walks.

#include "gemma3_tokenizer.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>

namespace {

// --- protobuf wire format helpers ------------------------------------------

struct Reader {
    const uint8_t* p;
    const uint8_t* end;

    bool eof() const { return p >= end; }

    bool read_varint(uint64_t& out) {
        out = 0;
        int shift = 0;
        while (p < end) {
            uint8_t b = *p++;
            out |= (uint64_t)(b & 0x7f) << shift;
            if ((b & 0x80) == 0) return true;
            shift += 7;
            if (shift >= 64) return false;
        }
        return false;
    }

    bool read_fixed32(uint32_t& out) {
        if (end - p < 4) return false;
        out = (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
              ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
        p += 4;
        return true;
    }

    bool read_fixed64(uint64_t& out) {
        if (end - p < 8) return false;
        uint64_t v = 0;
        for (int i = 0; i < 8; ++i) v |= (uint64_t)p[i] << (8 * i);
        p += 8;
        out = v;
        return true;
    }

    // Skip a field of the given wire type (for unused sections).
    bool skip_field(int wire_type) {
        if (wire_type == 0) {        // varint
            uint64_t tmp;
            return read_varint(tmp);
        } else if (wire_type == 1) { // fixed64
            uint64_t tmp;
            return read_fixed64(tmp);
        } else if (wire_type == 2) { // length-delimited
            uint64_t len;
            if (!read_varint(len)) return false;
            if ((uint64_t)(end - p) < len) return false;
            p += len;
            return true;
        } else if (wire_type == 5) { // fixed32
            uint32_t tmp;
            return read_fixed32(tmp);
        }
        return false;
    }
};

// Parse one SentencePiece message from `sub` (length-delimited sub-view).
bool parse_piece(const uint8_t* data, size_t len, Gemma3Tokenizer::Piece& out) {
    Reader r{data, data + len};
    out = {};
    out.type  = Gemma3Tokenizer::NORMAL;
    out.score = 0.0f;
    while (!r.eof()) {
        uint64_t tag;
        if (!r.read_varint(tag)) return false;
        int field = (int)(tag >> 3);
        int wire  = (int)(tag & 0x07);
        if (field == 1 && wire == 2) {
            uint64_t slen;
            if (!r.read_varint(slen)) return false;
            if ((uint64_t)(r.end - r.p) < slen) return false;
            out.text.assign((const char*)r.p, slen);
            r.p += slen;
        } else if (field == 2 && wire == 5) {
            uint32_t bits;
            if (!r.read_fixed32(bits)) return false;
            float f;
            std::memcpy(&f, &bits, 4);
            out.score = f;
        } else if (field == 3 && wire == 0) {
            uint64_t v;
            if (!r.read_varint(v)) return false;
            out.type = (uint8_t)v;
        } else {
            if (!r.skip_field(wire)) return false;
        }
    }
    return true;
}

}  // namespace

bool Gemma3Tokenizer::load_from_spm(const std::string& path, std::string* error) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        if (error) *error = "cannot open " + path;
        return false;
    }
    std::vector<uint8_t> buf((std::istreambuf_iterator<char>(f)),
                             std::istreambuf_iterator<char>());

    Reader r{buf.data(), buf.data() + buf.size()};
    pieces_.clear();
    piece_to_id_.clear();

    while (!r.eof()) {
        uint64_t tag;
        if (!r.read_varint(tag)) {
            if (error) *error = "truncated protobuf";
            return false;
        }
        int field = (int)(tag >> 3);
        int wire  = (int)(tag & 0x07);

        if (field == 1 && wire == 2) {
            uint64_t slen;
            if (!r.read_varint(slen)) return false;
            if ((uint64_t)(r.end - r.p) < slen) return false;
            Piece p;
            if (!parse_piece(r.p, slen, p)) {
                if (error) *error = "malformed SentencePiece";
                return false;
            }
            piece_to_id_[p.text] = (int32_t)pieces_.size();
            pieces_.push_back(std::move(p));
            r.p += slen;
        } else {
            if (!r.skip_field(wire)) {
                if (error) *error = "cannot skip unknown field";
                return false;
            }
        }
    }

    // Locate special tokens by name. Gemma's convention matches llama.cpp.
    auto find = [&](const std::string& s, int32_t fallback) -> int32_t {
        auto it = piece_to_id_.find(s);
        return it == piece_to_id_.end() ? fallback : it->second;
    };
    pad_id_ = find("<pad>", 0);
    eos_id_ = find("<eos>", 1);
    bos_id_ = find("<bos>", 2);
    unk_id_ = find("<unk>", 3);

    return true;
}

// Gemma's meta-space prefix byte sequence: U+2581 (LOWER ONE EIGHTH BLOCK)
// encoded as UTF-8: 0xE2 0x96 0x81 (three bytes).
static const std::string kMetaSpace = "\xE2\x96\x81";

// Byte-level fallback: SentencePiece encodes unknown bytes as
// "<0xHH>" pieces (tokens 6..261 cover 0x00..0xFF). Gemma uses the same.
static std::string byte_piece(uint8_t b) {
    static const char hex[] = "0123456789ABCDEF";
    char buf[8];
    buf[0] = '<'; buf[1] = '0'; buf[2] = 'x';
    buf[3] = hex[(b >> 4) & 0xf];
    buf[4] = hex[b & 0xf];
    buf[5] = '>'; buf[6] = 0;
    return std::string(buf, 6);
}

// Classic SentencePiece BPE: split input into unicode chars prefixed with
// meta-space for word boundaries, then repeatedly merge adjacent pairs
// using piece scores (higher = earlier) until no merge is possible.
//
// `word` here is the raw UTF-8 string including its leading meta-space.
void Gemma3Tokenizer::bpe_encode_word(const std::string& word,
                                      std::vector<int32_t>& out) const {
    if (word.empty()) return;

    // 1) Break into individual unicode code points (1-4 byte UTF-8 runs),
    //    each represented by its piece string. If the codepoint has no
    //    direct piece, fall back to its bytes as <0xHH> pieces.
    struct Sym {
        std::string text;
        int32_t id;
    };
    std::vector<Sym> syms;
    size_t i = 0;
    while (i < word.size()) {
        // Figure out UTF-8 run length at byte i.
        uint8_t c   = (uint8_t)word[i];
        size_t len  = 1;
        if ((c & 0x80) == 0)       len = 1;
        else if ((c & 0xE0) == 0xC0) len = 2;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xF8) == 0xF0) len = 4;
        if (i + len > word.size()) len = 1;

        std::string cp = word.substr(i, len);
        auto it = piece_to_id_.find(cp);
        if (it != piece_to_id_.end()) {
            syms.push_back({cp, it->second});
        } else {
            // Fallback to byte-level pieces.
            for (size_t b = 0; b < len; ++b) {
                std::string bp = byte_piece((uint8_t)word[i + b]);
                auto bit = piece_to_id_.find(bp);
                int32_t id = bit == piece_to_id_.end() ? unk_id_ : bit->second;
                syms.push_back({bp, id});
            }
        }
        i += len;
    }

    // 2) Merge adjacent pairs. Use a priority-based loop: at each step,
    //    scan for the best merge (highest score), apply it, repeat.
    // This is O(N^2) in word length but prompts are short (<300 tokens),
    //    so it's fine.
    while (syms.size() > 1) {
        float best_score    = -std::numeric_limits<float>::infinity();
        int   best_idx      = -1;
        int32_t best_id     = -1;
        std::string best_text;
        for (size_t k = 0; k + 1 < syms.size(); ++k) {
            std::string merged = syms[k].text + syms[k + 1].text;
            auto it = piece_to_id_.find(merged);
            if (it == piece_to_id_.end()) continue;
            float score = pieces_[it->second].score;
            if (score > best_score) {
                best_score = score;
                best_idx   = (int)k;
                best_id    = it->second;
                best_text  = std::move(merged);
            }
        }
        if (best_idx < 0) break;
        syms[best_idx] = {std::move(best_text), best_id};
        syms.erase(syms.begin() + best_idx + 1);
    }

    for (auto& s : syms) out.push_back(s.id);
}

std::vector<int32_t> Gemma3Tokenizer::encode(const std::string& text,
                                             bool add_bos,
                                             bool add_eos) const {
    std::vector<int32_t> ids;
    if (add_bos) ids.push_back(bos_id_);

    // Pre-tokenisation: SentencePiece replaces spaces with the meta-space
    // character. Gemma-3 disables the "add_dummy_prefix" behaviour, so the
    // FIRST word is encoded *without* a leading meta-space (the first chunk
    // has no prefix), but subsequent words get one.
    std::string normalised;
    normalised.reserve(text.size() + kMetaSpace.size() * 4);
    for (size_t i = 0; i < text.size(); ++i) {
        char c = text[i];
        if (c == ' ') normalised += kMetaSpace;
        else          normalised += c;
    }

    // Split into chunks. The first chunk (before any meta-space) is encoded
    // as-is; subsequent chunks (starting at a meta-space boundary) include
    // their leading meta-space as part of the word.
    if (!normalised.empty()) {
        size_t first_ms = normalised.find(kMetaSpace);
        size_t end0     = first_ms == std::string::npos ? normalised.size() : first_ms;
        if (end0 > 0) {
            bpe_encode_word(normalised.substr(0, end0), ids);
        }
        size_t pos = first_ms;
        while (pos != std::string::npos && pos < normalised.size()) {
            size_t next = normalised.find(kMetaSpace, pos + kMetaSpace.size());
            if (next == std::string::npos) next = normalised.size();
            std::string word = normalised.substr(pos, next - pos);
            bpe_encode_word(word, ids);
            pos = next;
        }
    }

    if (add_eos) ids.push_back(eos_id_);
    return ids;
}

std::string Gemma3Tokenizer::decode(const std::vector<int32_t>& ids) const {
    std::string out;
    for (int32_t id : ids) {
        if (id < 0 || id >= (int32_t)pieces_.size()) continue;
        const auto& p = pieces_[id];
        if (p.type == CONTROL) continue;   // skip BOS/EOS/pad
        std::string piece = p.text;
        // Convert back: meta-space → regular space.
        size_t pos = 0;
        while ((pos = piece.find(kMetaSpace, pos)) != std::string::npos) {
            piece.replace(pos, kMetaSpace.size(), " ");
            pos += 1;
        }
        out += piece;
    }
    return out;
}
