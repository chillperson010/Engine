// extract — stream PGN on stdin, keep only games that pass the titled/human
// filter, and emit WDL-labeled training positions.
//
// Output (stdout), one sample per line:   <FEN>\t<white_score>
//   white_score in {0.0, 0.5, 1.0}  (game result from White's point of view;
//   the FEN's side-to-move field lets the trainer derive the stm perspective).
//
// Usage:
//   zstd -dc lichess_2024-01.pgn.zst | extract [flags] > out.txt
//
// Flags:
//   --no-require-titles      keep all games (for OTB sets); default requires both titled
//   --allow-lm               count Lichess honorary LM as a title (default off)
//   --min-elo N              require both Elo >= N (useful with --no-require-titles)
//   --skip-plies N           skip the first N plies of each game (default 8)
//   --sample-every K         emit 1 of every K eligible plies (default 4)
//   --max-per-game M         cap positions emitted per game (default 20)
//   --max-games G            stop after scanning G qualifying games (0 = no cap)
//   --manifest FILE          write a stats manifest here (also printed to stderr)
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "../third_party/chess-library/chess.hpp"
#include "filter.h"

using namespace chess;

namespace {

struct Config {
    titled::FilterConfig filter;
    int skip_plies = 8;
    int sample_every = 4;
    int max_per_game = 20;
    uint64_t max_games = 0;
    std::string manifest;
    bool policy = false;   // emit "FEN\t<uci_move>" (human move) instead of WDL labels
};

struct Stats {
    uint64_t games_scanned = 0;
    uint64_t games_kept = 0;
    uint64_t positions = 0;
    uint64_t games_parse_errors = 0;
};

class ExtractVisitor : public pgn::Visitor {
public:
    ExtractVisitor(const Config& cfg, std::ostream& out, Stats& stats)
        : cfg_(cfg), out_(out), stats_(stats) {}

    bool done() const { return cfg_.max_games && stats_.games_kept >= cfg_.max_games; }

    struct StopExtraction {};

    void startPgn() override {
        if (done()) throw StopExtraction{};  // hard stop once max_games reached
        h_ = titled::GameHeaders{};
        start_fen_ = constants::STARTPOS;
        white_score_ = 0.5f;
        plies_ = 0;
        emitted_ = 0;
        keeping_ = false;
        broken_ = false;
    }

    void header(std::string_view key, std::string_view value) override {
        if (key == "WhiteTitle") h_.white_title = value;
        else if (key == "BlackTitle") h_.black_title = value;
        else if (key == "WhiteElo") h_.white_elo = atoi_sv(value);
        else if (key == "BlackElo") h_.black_elo = atoi_sv(value);
        else if (key == "Result") { h_.result = value; h_.has_result = true; }
        else if (key == "FEN") start_fen_ = std::string(value);
    }

    void startMoves() override {
        ++stats_.games_scanned;
        keeping_ = titled::game_qualifies(h_, cfg_.filter) &&
                   titled::result_to_score(h_.result, white_score_);
        if (!keeping_) {
            skipPgn(true);   // skip this game's move list entirely
            return;
        }
        ++stats_.games_kept;
        if (!board_.setFen(start_fen_)) {  // bad FEN header -> drop game
            keeping_ = false;
            broken_ = true;
            skipPgn(true);
        }
    }

    void move(std::string_view san, std::string_view) override {
        if (!keeping_ || broken_) return;
        Move m;
        try {
            m = uci::parseSan(board_, san);
        } catch (...) {
            broken_ = true;
            ++stats_.games_parse_errors;
            skipPgn(true);
            return;
        }
        if (m == Move::NO_MOVE) { broken_ = true; skipPgn(true); return; }

        // Policy mode: record (position-before-move, human move) for move-prediction
        // training. Sampling is based on the ply about to be played.
        if (cfg_.policy) {
            int next_ply = plies_ + 1;
            bool sample = next_ply > cfg_.skip_plies && emitted_ < cfg_.max_per_game &&
                          (next_ply - cfg_.skip_plies) % cfg_.sample_every == 0 && !board_.inCheck();
            if (sample) {
                out_ << board_.getFen() << '\t' << uci::moveToUci(m) << '\n';
                ++emitted_;
                ++stats_.positions;
            }
            board_.makeMove(m);
            ++plies_;
            return;
        }

        board_.makeMove(m);
        ++plies_;

        if (plies_ <= cfg_.skip_plies) return;
        if (emitted_ >= cfg_.max_per_game) return;
        if ((plies_ - cfg_.skip_plies) % cfg_.sample_every != 0) return;
        if (board_.inCheck()) return;                 // keep "quiet" samples only
        if (board_.isInsufficientMaterial()) return;

        out_ << board_.getFen() << '\t' << white_score_ << '\n';
        ++emitted_;
        ++stats_.positions;
    }

    void endPgn() override {}

private:
    static int atoi_sv(std::string_view v) {
        int n = 0;
        for (char c : v) {
            if (c < '0' || c > '9') break;
            n = n * 10 + (c - '0');
        }
        return n;
    }

    const Config& cfg_;
    std::ostream& out_;
    Stats& stats_;
    Board board_;
    titled::GameHeaders h_;
    std::string start_fen_;
    float white_score_ = 0.5f;
    int plies_ = 0;
    int emitted_ = 0;
    bool keeping_ = false;
    bool broken_ = false;
};

Config parse_args(int argc, char** argv) {
    Config c;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--no-require-titles") c.filter.require_titles = false;
        else if (a == "--allow-lm") c.filter.allow_lm = true;
        else if (a == "--min-elo") c.filter.min_elo = std::stoi(next());
        else if (a == "--skip-plies") c.skip_plies = std::stoi(next());
        else if (a == "--sample-every") c.sample_every = std::max(1, std::stoi(next()));
        else if (a == "--max-per-game") c.max_per_game = std::stoi(next());
        else if (a == "--max-games") c.max_games = std::stoull(next());
        else if (a == "--policy") c.policy = true;
        else if (a == "--manifest") c.manifest = next();
        else {
            std::cerr << "extract: unknown flag '" << a << "'\n";
            std::exit(2);
        }
    }
    return c;
}

}  // namespace

int main(int argc, char** argv) {
    attacks::initAttacks();
    std::ios::sync_with_stdio(false);

    Config cfg = parse_args(argc, argv);
    Stats stats;

    ExtractVisitor vis(cfg, std::cout, stats);
    pgn::StreamParser parser(std::cin);
    try {
        auto err = parser.readGames(vis);
        (void)err;  // EOF / NotEnoughData at stream end is expected
    } catch (const ExtractVisitor::StopExtraction&) {
        // Reached --max-games; stop reading the stream.
    }

    std::ostringstream manifest;
    manifest << "games_scanned=" << stats.games_scanned << "\n"
             << "games_kept=" << stats.games_kept << "\n"
             << "positions=" << stats.positions << "\n"
             << "games_parse_errors=" << stats.games_parse_errors << "\n"
             << "require_titles=" << cfg.filter.require_titles << "\n"
             << "allow_lm=" << cfg.filter.allow_lm << "\n"
             << "min_elo=" << cfg.filter.min_elo << "\n"
             << "skip_plies=" << cfg.skip_plies << "\n"
             << "sample_every=" << cfg.sample_every << "\n"
             << "max_per_game=" << cfg.max_per_game << "\n";
    std::cerr << manifest.str();
    if (!cfg.manifest.empty()) {
        std::ofstream mf(cfg.manifest);
        mf << manifest.str();
    }
    return 0;
}
