// Tokenizer parity test for Gemma 3.
//
// Encodes a handful of fixed strings with our GemmaTokenizer and compares to the
// token IDs produced by transformers' AutoTokenizer (google/gemma-3-12b-it). The
// expected IDs below are hard-coded from a Python reference run — if the HF vocab
// ever changes they must be regenerated.

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "tokenizers/gemma_tokenizer.h"

namespace {

struct Case {
    std::string input;
    std::vector<int> expected_no_bos;
};

bool run_case(GemmaTokenizer& tk, const Case& c, int idx) {
    std::vector<int> got = tk.encode(c.input);
    bool ok              = (got == c.expected_no_bos);
    std::printf("  [%2d] ", idx);
    if (ok) {
        std::printf("PASS  %zu tokens: ", got.size());
    } else {
        std::printf("FAIL  got=%zu exp=%zu  ", got.size(), c.expected_no_bos.size());
    }
    for (size_t i = 0; i < got.size() && i < 8; i++) std::printf("%d ", got[i]);
    if (got.size() > 8) std::printf("...");
    std::printf("\n");
    if (!ok) {
        std::printf("       input    : %s\n", c.input.c_str());
        std::printf("       expected : ");
        for (int x : c.expected_no_bos) std::printf("%d ", x);
        std::printf("\n       got      : ");
        for (int x : got) std::printf("%d ", x);
        std::printf("\n");
    }
    return ok;
}

}  // namespace

int main(int argc, char** argv) {
    const char* default_path =
        "/home/ilintar/.cache/huggingface/hub/models--google--gemma-3-12b-it/"
        "snapshots/96b6f1eccf38110c56df3a15bffe176da04bfd80/tokenizer.json";
    std::string path = (argc > 1) ? argv[1] : default_path;

    GemmaTokenizer tk;
    std::printf("[load] %s\n", path.c_str());
    if (!tk.load_from_file(path)) {
        std::fprintf(stderr, "fatal: could not load tokenizer\n");
        return 1;
    }
    std::printf("[load] vocab=%d bos=%d eos=%d pad=%d unk=%d\n",
                tk.vocab_size(), tk.BOS_TOKEN_ID, tk.EOS_TOKEN_ID, tk.PAD_TOKEN_ID, tk.UNK_TOKEN_ID);

    // Ground truth from transformers.AutoTokenizer("google/gemma-3-12b-it") with
    // add_special_tokens=False.
    std::vector<Case> cases = {
        {"hello",          {23391}},
        {"hello world",    {23391, 1902}},
        {"  a  b",         {138, 236746, 138, 236763}},
        {"naïve",          {1789, 238527, 560}},
        {"你好",           {144626}},
        {"→ a",            {238183, 496}},
        {"The quick brown fox jumps over the lazy dog.",
                           {818, 3823, 8864, 37423, 38167, 1024, 506, 31770, 4799, 236761}},
        {"",               {}},
        {" ",              {236743}},
        {"\n\n\ttabs and\nnewlines",
                           {108, 255968, 39218, 532, 107, 208697}},
        {"mixed: ABCdef 123 !@# UNK char: \xe2\x80\x8b",
                           {63258, 236787, 21593, 2063, 236743, 236770, 236778, 236800,
                            1717, 236940, 236865, 7866, 236855, 1577, 236787, 36504}},
    };

    int pass = 0;
    for (size_t i = 0; i < cases.size(); i++) {
        if (run_case(tk, cases[i], (int)i)) pass++;
    }
    std::printf("\n%d / %zu cases passed.\n", pass, cases.size());
    return (pass == (int)cases.size()) ? 0 : 3;
}
