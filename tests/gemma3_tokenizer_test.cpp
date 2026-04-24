// Manual test: tokenise a prompt with our Gemma-3 BPE tokenizer.
//
// Build:
//   c++ -std=c++17 -O2 -Isrc \
//       tests/gemma3_tokenizer_test.cpp \
//       src/tokenizers/gemma3_tokenizer.cpp \
//       -o /tmp/gemma3_tok_test
//
// Run:
//   /tmp/gemma3_tok_test /path/to/tokenizer.model "a cat walking across a grassy field"
//
// Compare output to the reference printed by
//   python - <<'PY'
//   from transformers import AutoTokenizer
//   tok = AutoTokenizer.from_pretrained("google/gemma-3-12b-it")
//   print(tok.encode("a cat walking across a grassy field"))
//   PY

#include <cstdio>
#include <cstdlib>
#include <string>

#include "tokenizers/gemma3_tokenizer.h"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <tokenizer.model> <prompt> [add_eos=0|1]\n", argv[0]);
        return 1;
    }
    Gemma3Tokenizer tok;
    std::string err;
    if (!tok.load_from_spm(argv[1], &err)) {
        std::fprintf(stderr, "load failed: %s\n", err.c_str());
        return 2;
    }
    std::fprintf(stderr, "loaded %d pieces (bos=%d eos=%d pad=%d unk=%d)\n",
                 (int)tok.vocab_size(), (int)tok.bos_id(),
                 (int)tok.eos_id(), (int)tok.pad_id(), (int)tok.unk_id());

    bool add_eos = argc > 3 && std::atoi(argv[3]) != 0;
    auto ids = tok.encode(argv[2], /*add_bos=*/true, /*add_eos=*/add_eos);
    for (size_t i = 0; i < ids.size(); ++i) {
        std::printf("%s%d", i ? "," : "", ids[i]);
    }
    std::printf("\n");
    std::fprintf(stderr, "count=%zu  decoded=\"%s\"\n", ids.size(),
                 tok.decode(ids).c_str());
    return 0;
}
