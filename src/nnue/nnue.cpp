#include "nnue.h"

// M1 stub. Replaced in M3 by a real HalfKP feature-transformer + quantized
// network. Keeping the symbols here means the rest of the engine links and
// runs today on the hand-crafted evaluation.
namespace eng::nnue {

bool is_loaded() { return false; }

bool load(const std::string&) { return false; }

Value evaluate(const chess::Board&) { return 0; }

}  // namespace eng::nnue
