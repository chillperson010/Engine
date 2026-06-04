#include <chrono>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "../third_party/chess-library/chess.hpp"
#include "search.h"
#include "timeman.h"

using namespace chess;

int uci_loop();

// ---- perft (move-generation correctness) ----------------------------------

static uint64_t perft(Board& b, int depth) {
    if (depth == 0) return 1;
    Movelist moves;
    movegen::legalmoves(moves, b);
    if (depth == 1) return moves.size();
    uint64_t nodes = 0;
    for (const auto& m : moves) {
        b.makeMove(m);
        nodes += perft(b, depth - 1);
        b.unmakeMove(m);
    }
    return nodes;
}

struct PerftCase {
    std::string fen;
    int depth;
    uint64_t expected;
    std::string name;
};

static int run_perft_suite() {
    std::vector<PerftCase> cases = {
        {constants::STARTPOS, 6, 119060324ULL, "startpos d6"},
        {"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 5, 193690690ULL,
         "kiwipete d5"},
        {"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 7, 178633661ULL, "position3 d7"},
        {"r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 5, 15833292ULL,
         "position4 d5"},
        {"rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 5, 89941194ULL, "position5 d5"},
        {"r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 5, 164075551ULL,
         "position6 d5"},
    };

    int failures = 0;
    for (const auto& c : cases) {
        Board b(c.fen);
        auto t0 = std::chrono::steady_clock::now();
        uint64_t got = perft(b, c.depth);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::steady_clock::now() - t0).count();
        bool ok = (got == c.expected);
        std::cout << (ok ? "[ OK ] " : "[FAIL] ") << c.name << ": got " << got << " expected "
                  << c.expected << " (" << ms << " ms)\n";
        if (!ok) ++failures;
    }
    std::cout << (failures ? "PERFT FAILED\n" : "PERFT PASSED\n");
    return failures == 0 ? 0 : 1;
}

// ---- bench (fixed-depth node signature) -----------------------------------

static int run_bench(int depth) {
    std::vector<std::string> fens = {
        constants::STARTPOS,
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
    };
    uint64_t total = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (const auto& fen : fens) {
        Board b(fen);
        eng::SearchLimits lim;
        lim.depth = depth;
        eng::g_searcher.clear();
        eng::g_searcher.think(b, lim);
        total += eng::g_searcher.nodes();
    }
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::steady_clock::now() - t0).count();
    uint64_t nps = ms > 0 ? total * 1000 / ms : total;
    std::cout << "\nbench: " << total << " nodes " << nps << " nps\n";
    return 0;
}

int main(int argc, char** argv) {
    attacks::initAttacks();  // ensure sliding-piece tables are ready

    if (argc > 1) {
        std::string mode = argv[1];
        if (mode == "perft") return run_perft_suite();
        if (mode == "bench") {
            int d = (argc > 2) ? std::stoi(argv[2]) : 11;
            return run_bench(d);
        }
    }
    return uci_loop();
}
