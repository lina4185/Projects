class LoadAnalyzer:

    def compute_penalty(self, passengers, cargo_weight):

        base_weight = passengers * 70
        total_weight = base_weight + cargo_weight

        penalty = total_weight * 0.0003

        return {
            "load_penalty": penalty,
            "total_weight": total_weight
        }