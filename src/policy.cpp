#include "policy.h"

#include <cstring>
#include <fstream>
#include <vector>

using namespace chess;

namespace eng::policy {

namespace {

struct Net {
    bool loaded = false;
    int Hp = 0;
    int NF = 0;
    std::vector<float> W1;   // [NF*Hp]
    std::vector<float> b1;   // [Hp]
    std::vector<float> Wf;   // [Hp*64]
    std::vector<float> bf;   // [64]
    std::vector<float> Wt;   // [Hp*64]
    std::vector<float> bt;   // [64]
};

Net g;

// Must match policy_io / nnue_io feature indexing.
inline int feature_index(int color, int pt, int sq, int persp) {
    int rel_color = (color == persp) ? 0 : 1;
    int rel_sq = (persp == 0) ? sq : (sq ^ 56);
    return (rel_color * 6 + pt) * 64 + rel_sq;
}

}  // namespace

bool is_loaded() { return g.loaded; }

bool load(const std::string& path) {
    g.loaded = false;
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    char magic[4];
    f.read(magic, 4);
    if (std::memcmp(magic, "TPL1", 4) != 0) return false;
    int32_t hp, nf;
    f.read(reinterpret_cast<char*>(&hp), 4);
    f.read(reinterpret_cast<char*>(&nf), 4);
    if (!f || hp <= 0 || hp > 4096 || nf != 768) return false;

    Net n;
    n.Hp = hp;
    n.NF = nf;
    auto rd = [&](std::vector<float>& v, size_t count) {
        v.resize(count);
        f.read(reinterpret_cast<char*>(v.data()), count * sizeof(float));
    };
    rd(n.W1, static_cast<size_t>(nf) * hp);
    rd(n.b1, hp);
    rd(n.Wf, static_cast<size_t>(hp) * 64);
    rd(n.bf, 64);
    rd(n.Wt, static_cast<size_t>(hp) * 64);
    rd(n.bt, 64);
    if (!f) return false;

    n.loaded = true;
    g = std::move(n);
    return true;
}

void compute(const Board& board, Eval& out) {
    const int Hp = g.Hp;
    const int persp = static_cast<int>(board.sideToMove());

    std::vector<float> hid(g.b1);  // start from bias
    for (int sq = 0; sq < 64; ++sq) {
        Piece pc = board.at(Square(sq));
        if (pc == Piece::NONE) continue;
        int color = static_cast<int>(pc.color());
        int pt = static_cast<int>(pc.type());
        const float* col = &g.W1[static_cast<size_t>(feature_index(color, pt, sq, persp)) * Hp];
        for (int h = 0; h < Hp; ++h) hid[h] += col[h];
    }
    for (int h = 0; h < Hp; ++h)
        if (hid[h] < 0.f) hid[h] = 0.f;  // ReLU

    for (int s = 0; s < 64; ++s) {
        float fa = g.bf[s], ta = g.bt[s];
        for (int h = 0; h < Hp; ++h) {
            fa += hid[h] * g.Wf[h * 64 + s];
            ta += hid[h] * g.Wt[h * 64 + s];
        }
        out.from_logit[s] = fa;
        out.to_logit[s] = ta;
    }
}

float move_score(const Board& board, const Eval& e, Move m) {
    int persp = static_cast<int>(board.sideToMove());
    int from = m.from().index();
    int to = m.to().index();
    int rf = (persp == 0) ? from : (from ^ 56);
    int rt = (persp == 0) ? to : (to ^ 56);
    return e.from_logit[rf] + e.to_logit[rt];
}

}  // namespace eng::policy
