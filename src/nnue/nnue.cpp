#include "nnue.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <vector>

using namespace chess;

// Perspective NNUE inference: (768 -> H) x2 -> 1, quantized. This mirrors
// training/nnue_io.py byte-for-byte and arithmetic-for-arithmetic so a trained
// network evaluates identically here and in the trainer. Evaluation is a full
// refresh over the pieces on the board (<=32 of them), which is cheap and keeps
// the result trivially correct; an incremental accumulator is a future speedup.
namespace eng::nnue {

namespace {

struct Network {
    bool loaded = false;
    int H = 0;
    int NF = 0;
    int SCALE = 0;
    int QA = 0;
    int QB = 0;
    int32_t out_b = 0;
    std::vector<int16_t> ft_w;   // [NF * H], feature-major
    std::vector<int16_t> ft_b;   // [H]
    std::vector<int16_t> out_w;  // [2H]
};

Network g_net;

// Must match nnue_io.feature_index exactly.
inline int feature_index(int color, int pt, int sq, int persp) {
    int rel_color = (color == persp) ? 0 : 1;
    int rel_sq = (persp == 0) ? sq : (sq ^ 56);
    return (rel_color * 6 + pt) * 64 + rel_sq;
}

}  // namespace

bool is_loaded() { return g_net.loaded; }

bool load(const std::string& path) {
    g_net.loaded = false;
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;

    char magic[4];
    f.read(magic, 4);
    if (std::memcmp(magic, "TND1", 4) != 0) return false;

    int32_t hdr[5];
    f.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
    int32_t out_b;
    f.read(reinterpret_cast<char*>(&out_b), sizeof(out_b));
    if (!f) return false;

    Network n;
    n.H = hdr[0];
    n.NF = hdr[1];
    n.SCALE = hdr[2];
    n.QA = hdr[3];
    n.QB = hdr[4];
    n.out_b = out_b;
    if (n.H <= 0 || n.H > 1024 || n.NF != 768 || n.SCALE <= 0 || n.QA <= 0 || n.QB <= 0)
        return false;

    n.ft_w.resize(static_cast<size_t>(n.NF) * n.H);
    n.ft_b.resize(n.H);
    n.out_w.resize(static_cast<size_t>(2) * n.H);
    f.read(reinterpret_cast<char*>(n.ft_w.data()), n.ft_w.size() * sizeof(int16_t));
    f.read(reinterpret_cast<char*>(n.ft_b.data()), n.ft_b.size() * sizeof(int16_t));
    f.read(reinterpret_cast<char*>(n.out_w.data()), n.out_w.size() * sizeof(int16_t));
    if (!f) return false;

    n.loaded = true;
    g_net = std::move(n);
    return true;
}

Value evaluate(const Board& board) {
    const Network& net = g_net;
    const int H = net.H;

    // Stack buffers (no per-eval heap allocation). Full refresh over the pieces
    // on the board (<=32), which is the bulk of the cost; the compiler
    // vectorizes the column adds under -O3 -march=native.
    constexpr int MAX_H = 1024;
    alignas(32) int32_t accw[MAX_H];
    alignas(32) int32_t accb[MAX_H];
    const int16_t* ftb = net.ft_b.data();
    for (int h = 0; h < H; ++h) {
        accw[h] = ftb[h];
        accb[h] = ftb[h];
    }

    const int16_t* ftw = net.ft_w.data();
    for (int sq = 0; sq < 64; ++sq) {
        Piece pc = board.at(Square(sq));
        if (pc == Piece::NONE) continue;
        int color = static_cast<int>(pc.color());
        int pt = static_cast<int>(pc.type());
        const int16_t* wcol = ftw + static_cast<size_t>(feature_index(color, pt, sq, 0)) * H;
        const int16_t* bcol = ftw + static_cast<size_t>(feature_index(color, pt, sq, 1)) * H;
        for (int h = 0; h < H; ++h) {
            accw[h] += wcol[h];
            accb[h] += bcol[h];
        }
    }

    const bool white = (board.sideToMove() == Color::WHITE);
    const int32_t* own = white ? accw : accb;
    const int32_t* opp = white ? accb : accw;
    const int16_t* ow = net.out_w.data();

    int64_t out = net.out_b;
    const int QA = net.QA;
    for (int h = 0; h < H; ++h) {
        int32_t o = std::clamp(own[h], 0, QA);
        int32_t p = std::clamp(opp[h], 0, QA);
        out += static_cast<int64_t>(o) * ow[h];
        out += static_cast<int64_t>(p) * ow[H + h];
    }

    int64_t cp = (out * net.SCALE) / (static_cast<int64_t>(net.QA) * net.QB);  // trunc toward 0
    if (cp > 29000) cp = 29000;       // keep clear of mate scores
    if (cp < -29000) cp = -29000;
    return static_cast<Value>(cp);
}

}  // namespace eng::nnue
