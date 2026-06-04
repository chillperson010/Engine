#include "see.h"

#include <algorithm>

using namespace chess;

namespace eng {

// P, N, B, R, Q, K  (king huge so a defended king-capture is never "won").
static const int SEE_VAL[7] = {100, 320, 330, 500, 900, 10000, 0};

static Bitboard attackers_to(const Board& b, Square sq, Bitboard occ) {
    Bitboard a(0ULL);
    a |= attacks::pawn(Color::BLACK, sq) & b.pieces(PieceType::PAWN, Color::WHITE);
    a |= attacks::pawn(Color::WHITE, sq) & b.pieces(PieceType::PAWN, Color::BLACK);
    a |= attacks::knight(sq) & b.pieces(PieceType::KNIGHT);
    a |= attacks::king(sq) & b.pieces(PieceType::KING);
    Bitboard bq = b.pieces(PieceType::BISHOP) | b.pieces(PieceType::QUEEN);
    a |= attacks::bishop(sq, occ) & bq;
    Bitboard rq = b.pieces(PieceType::ROOK) | b.pieces(PieceType::QUEEN);
    a |= attacks::rook(sq, occ) & rq;
    return a & occ;
}

int see(const Board& b, Move m) {
    const Square from = m.from();
    const Square to = m.to();

    PieceType target = (m.typeOf() == Move::ENPASSANT) ? PieceType(PieceType::PAWN)
                                                       : b.at<PieceType>(to);
    int gain[32];
    int d = 0;
    gain[0] = (target == PieceType::NONE) ? 0 : SEE_VAL[static_cast<int>(target)];

    PieceType attackerPT = b.at<PieceType>(from);
    Bitboard occ = b.occ();
    occ.clear(from.index());
    if (m.typeOf() == Move::ENPASSANT) {
        int epsq = (b.sideToMove() == Color::WHITE) ? to.index() - 8 : to.index() + 8;
        occ.clear(epsq);
    }

    Color side = ~b.sideToMove();
    Bitboard attackers = attackers_to(b, to, occ);
    int aval = SEE_VAL[static_cast<int>(attackerPT)];

    while (true) {
        ++d;
        gain[d] = aval - gain[d - 1];

        Bitboard myatt = attackers & b.us(side);
        if (myatt.empty() || d >= 31) break;

        // Least valuable attacker for `side`.
        int pt = 0;
        Bitboard lva(0ULL);
        for (; pt < 6; ++pt) {
            lva = myatt & b.pieces(PieceType(static_cast<PieceType::underlying>(pt)), side);
            if (!lva.empty()) break;
        }
        occ.clear(lva.lsb());
        aval = SEE_VAL[pt];
        attackers = attackers_to(b, to, occ);   // recompute -> picks up x-ray attackers
        side = ~side;
    }

    while (--d > 0) gain[d - 1] = -std::max(-gain[d - 1], gain[d]);
    return gain[0];
}

}  // namespace eng
