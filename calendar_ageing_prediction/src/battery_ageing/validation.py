import math
from .config import MIN_TEMPERATURE_C, MAX_TEMPERATURE_C, MIN_SOC_PERCENT, MAX_SOC_PERCENT, MIN_DURATION_HOURS, MAX_DURATION_HOURS

def _number(value, label):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite.")
    return result

def validate_inputs(temperature_c, soc_percent, duration_hours):
    temperature = _number(temperature_c, "Temperature")
    soc = _number(soc_percent, "SOC")
    duration = _number(duration_hours, "Storage duration")
    if not MIN_TEMPERATURE_C <= temperature <= MAX_TEMPERATURE_C:
        raise ValueError("Temperature must be between 0 and 60 °C.")
    if not MIN_SOC_PERCENT <= soc <= MAX_SOC_PERCENT:
        raise ValueError("SOC must be between 0 and 100%.")
    if not MIN_DURATION_HOURS <= duration <= MAX_DURATION_HOURS:
        raise ValueError("Storage duration must be between 0 and 21,241 hours.")
    return temperature, soc, duration
