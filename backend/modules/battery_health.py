class BatteryHealthMonitor:

    def compute_health(self,
                       battery_age=2,
                       cycle_count=300,
                       temperature=25):

        degradation = battery_age * 0.02
        cycle_penalty = cycle_count * 0.00005

        health_score = 1 - degradation - cycle_penalty

        return {
            "battery_health": round(health_score, 2)
        }