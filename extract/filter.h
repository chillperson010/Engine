#pragma once
#include <string>
#include <string_view>

// The data-restriction guarantee: decides whether a game is eligible to enter
// the NNUE training set. Keep this logic small and testable — it is the reason
// the engine's learned evaluation contains no engine/cheater play.
namespace titled {

// Recognised FIDE / online title tokens (case-sensitive, as they appear in PGN).
// "BOT" and (by default) the honorary Lichess "LM" are NOT titles for our purpose.
bool is_titled_token(std::string_view title, bool allow_lm);

struct FilterConfig {
    bool require_titles = true;  // both players must carry a recognised title
    bool allow_lm = false;       // count Lichess honorary LM as a title
    int  min_elo = 0;            // if >0, both Elo ratings must be >= this
};

// Per-game header fields the filter inspects.
struct GameHeaders {
    std::string white_title;
    std::string black_title;
    int white_elo = 0;
    int black_elo = 0;
    std::string result;          // "1-0" / "0-1" / "1/2-1/2" / "*"
    bool has_result = false;
};

// Returns true if the game should be included in training.
bool game_qualifies(const GameHeaders& h, const FilterConfig& cfg);

// Map a PGN Result string to a white-POV score. Returns false if unknown ("*").
bool result_to_score(std::string_view result, float& white_score);

}  // namespace titled
