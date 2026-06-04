#include <iostream>
#include <sstream>
#include <string>
#include <thread>

#include "../third_party/chess-library/chess.hpp"
#include "eval.h"
#include "nnue/nnue.h"
#include "policy.h"
#include "search.h"
#include "timeman.h"

using namespace chess;
using namespace eng;

namespace {

const std::string ENGINE_NAME = "TitledNNUE";
const std::string ENGINE_VERSION = "0.1";
const std::string ENGINE_AUTHOR = "chillperson010";

Board g_board;          // current position
std::thread g_thread;   // active search thread
size_t g_hash_mb = 16;
int g_threads = 1;      // Lazy SMP worker count
int g_human_style = 0;  // 0 = pure strength; >0 = human-style move preference
std::string g_eval_file = "<none>";
std::string g_human_file = "<none>";

void join_search() {
    if (g_thread.joinable()) g_thread.join();
}

void stop_search() {
    eng::request_stop();
    join_search();
}

void cmd_uci() {
    std::cout << "id name " << ENGINE_NAME << " " << ENGINE_VERSION << "\n";
    std::cout << "id author " << ENGINE_AUTHOR << "\n";
    std::cout << "option name Hash type spin default 16 min 1 max 65536\n";
    std::cout << "option name Threads type spin default 1 min 1 max 256\n";
    std::cout << "option name UseNNUE type check default true\n";
    std::cout << "option name EvalFile type string default <none>\n";
    std::cout << "option name HumanStyle type spin default 0 min 0 max 100\n";
    std::cout << "option name HumanFile type string default <none>\n";
    std::cout << "uciok\n" << std::flush;
}

// Parse a "setoption name <Name> value <Value>" line.
void cmd_setoption(std::istringstream& is) {
    std::string token, name, value;
    is >> token;  // "name"
    while (is >> token && token != "value") {
        if (!name.empty()) name += ' ';
        name += token;
    }
    while (is >> token) {
        if (!value.empty()) value += ' ';
        value += token;
    }

    if (name == "Hash") {
        g_hash_mb = std::max<size_t>(1, std::stoul(value));
        g_searcher.resize_tt(g_hash_mb);
    } else if (name == "Threads") {
        g_threads = std::max(1, std::min(256, std::stoi(value)));
    } else if (name == "HumanStyle") {
        g_human_style = std::max(0, std::min(100, std::stoi(value)));
    } else if (name == "HumanFile") {
        g_human_file = value;
        if (value != "<none>" && !value.empty()) {
            if (eng::policy::load(value))
                std::cout << "info string loaded human policy " << value << "\n" << std::flush;
            else
                std::cout << "info string failed to load human policy " << value << "\n" << std::flush;
        }
    } else if (name == "EvalFile") {
        g_eval_file = value;
        if (value != "<none>" && !value.empty()) {
            if (nnue::load(value))
                std::cout << "info string loaded NNUE network " << value << "\n" << std::flush;
            else
                std::cout << "info string failed to load NNUE network " << value
                          << " (using hand-crafted eval)\n" << std::flush;
        }
    }
    // UseNNUE accepted but not acted on (NNUE loads via EvalFile).
}

// Apply "position [startpos | fen <fen>] [moves m1 m2 ...]".
void cmd_position(std::istringstream& is) {
    std::string token;
    is >> token;
    if (token == "startpos") {
        g_board.setFen(constants::STARTPOS);
        is >> token;  // maybe "moves"
    } else if (token == "fen") {
        std::string fen;
        while (is >> token && token != "moves") fen += token + ' ';
        g_board.setFen(fen);
        token = (is.eof() || token != "moves") ? token : "moves";
        if (token != "moves") token.clear();
        else token = "moves";
    }

    if (token == "moves") {
        std::string mv;
        while (is >> mv) g_board.makeMove(uci::uciToMove(g_board, mv));
    }
}

// Parse a "go" command into SearchLimits.
SearchLimits parse_go(std::istringstream& is) {
    SearchLimits lim;
    std::string token;
    while (is >> token) {
        if (token == "wtime") is >> lim.time[0];
        else if (token == "btime") is >> lim.time[1];
        else if (token == "winc") is >> lim.inc[0];
        else if (token == "binc") is >> lim.inc[1];
        else if (token == "movestogo") is >> lim.movestogo;
        else if (token == "movetime") is >> lim.movetime;
        else if (token == "depth") is >> lim.depth;
        else if (token == "nodes") is >> lim.nodes;
        else if (token == "infinite") lim.infinite = true;
    }
    return lim;
}

void cmd_go(std::istringstream& is) {
    stop_search();  // ensure no previous search is running
    SearchLimits lim = parse_go(is);
    lim.human_style = g_human_style;
    Board board = g_board;  // search on a copy
    int threads = g_threads;
    eng::clear_stop();      // clear synchronously before launching (avoids go/stop race)
    g_thread = std::thread([board, lim, threads]() mutable {
        Move best = eng::search_best(board, lim, threads);
        std::cout << "bestmove " << uci::moveToUci(best) << "\n" << std::flush;
    });
}

}  // namespace

int uci_loop() {
    std::ios::sync_with_stdio(false);
    g_board.setFen(constants::STARTPOS);
    g_searcher.resize_tt(g_hash_mb);

    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream is(line);
        std::string cmd;
        is >> cmd;

        if (cmd == "uci") {
            cmd_uci();
        } else if (cmd == "isready") {
            std::cout << "readyok\n" << std::flush;
        } else if (cmd == "ucinewgame") {
            stop_search();
            g_searcher.clear();
            g_board.setFen(constants::STARTPOS);
        } else if (cmd == "setoption") {
            cmd_setoption(is);
        } else if (cmd == "position") {
            cmd_position(is);
        } else if (cmd == "go") {
            cmd_go(is);
        } else if (cmd == "stop") {
            stop_search();
        } else if (cmd == "ponderhit") {
            // No pondering in M1.
        } else if (cmd == "quit" || cmd == "exit") {
            stop_search();
            break;
        }
    }
    stop_search();
    return 0;
}
