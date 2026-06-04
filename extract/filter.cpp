#include "filter.h"

#include <array>

namespace titled {

bool is_titled_token(std::string_view t, bool allow_lm) {
    static const std::array<std::string_view, 9> kTitles = {
        "GM", "IM", "FM", "CM", "NM", "WGM", "WIM", "WFM", "WCM"};
    for (auto x : kTitles)
        if (t == x) return true;
    if (allow_lm && t == "LM") return true;
    return false;  // BOT, empty, LM (unless allowed), and anything else -> not titled
}

bool game_qualifies(const GameHeaders& h, const FilterConfig& cfg) {
    // A usable training label requires a decisive/drawn result.
    float dummy;
    if (!h.has_result || !result_to_score(h.result, dummy)) return false;

    if (cfg.require_titles) {
        if (!is_titled_token(h.white_title, cfg.allow_lm)) return false;
        if (!is_titled_token(h.black_title, cfg.allow_lm)) return false;
    }

    if (cfg.min_elo > 0) {
        if (h.white_elo < cfg.min_elo || h.black_elo < cfg.min_elo) return false;
    }

    return true;
}

bool result_to_score(std::string_view r, float& white_score) {
    if (r == "1-0") {
        white_score = 1.0f;
        return true;
    }
    if (r == "0-1") {
        white_score = 0.0f;
        return true;
    }
    if (r == "1/2-1/2") {
        white_score = 0.5f;
        return true;
    }
    return false;  // "*" or anything unknown
}

}  // namespace titled
