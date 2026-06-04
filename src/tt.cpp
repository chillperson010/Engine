#include "tt.h"

namespace eng {

TranspositionTable TT;

static size_t next_pow2_entries(size_t mb) {
    size_t bytes = mb * 1024ULL * 1024ULL;
    size_t n = bytes / sizeof(TTEntry);
    if (n < 1024) n = 1024;
    size_t p = 1;
    while (p * 2 <= n) p *= 2;   // largest power of two <= n
    return p;
}

void TranspositionTable::resize(size_t mb) {
    size_t n = next_pow2_entries(mb);
    table_.assign(n, TTEntry{});
    mask_ = n - 1;
    gen_  = 0;
}

void TranspositionTable::clear() {
    std::fill(table_.begin(), table_.end(), TTEntry{});
    gen_ = 0;
}

TTEntry* TranspositionTable::probe(uint64_t key, bool& hit) {
    TTEntry* e = &table_[key & mask_];
    hit = (e->bound != BOUND_NONE) && (e->key16 == static_cast<uint16_t>(key >> 48));
    return e;
}

void TranspositionTable::store(uint64_t key, uint16_t move, Value value, Value eval, int depth,
                               Bound bound, int ply) {
    TTEntry* e = &table_[key & mask_];
    uint16_t k16 = static_cast<uint16_t>(key >> 48);

    // Replace if empty, same position, exact bound, deeper, or from an older search.
    bool replace = (e->bound == BOUND_NONE) || (e->key16 == k16) || (bound == BOUND_EXACT) ||
                   (e->gen != gen_) || (depth + 2 > e->depth);
    if (!replace) return;

    // Preserve an existing best move if this store has none.
    if (move == 0 && e->key16 == k16) move = e->move;

    e->key16 = k16;
    e->move  = move;
    e->value = static_cast<int16_t>(value_to_tt(value, ply));
    e->eval  = static_cast<int16_t>(eval);
    e->depth = static_cast<uint8_t>(depth);
    e->bound = bound;
    e->gen   = gen_;
}

int TranspositionTable::hashfull() const {
    int used = 0;
    int sampled = 1000;
    for (int i = 0; i < sampled && i < (int)table_.size(); ++i)
        if (table_[i].bound != BOUND_NONE && table_[i].gen == gen_) ++used;
    return used;
}

Value value_to_tt(Value v, int ply) {
    if (v >= VALUE_MATE_MAX) return v + ply;
    if (v <= -VALUE_MATE_MAX) return v - ply;
    return v;
}

Value value_from_tt(Value v, int ply) {
    if (v == VALUE_NONE) return VALUE_NONE;
    if (v >= VALUE_MATE_MAX) return v - ply;
    if (v <= -VALUE_MATE_MAX) return v + ply;
    return v;
}

}  // namespace eng
