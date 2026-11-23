"""Physical-time schedule for unchanged nine-frame sampling windows."""


def cycling_schedule(*, num_times: int, assimilation_period: int,
                     forecast_span: int, stride: int = 1) -> tuple[list[int], int]:
    if stride not in (1, 2):
        raise ValueError("Only primary stride 1 and terminal-nowcast stride 2 are supported")
    if assimilation_period <= 0 or forecast_span < 0 or num_times <= assimilation_period:
        raise ValueError("Invalid cycling dimensions")
    if stride == 2 and forecast_span != assimilation_period:
        raise ValueError("Stride 2 requires terminal lead to equal one assimilation interval")
    centers = list(range(assimilation_period, num_times, stride * assimilation_period))
    feedback_frame = assimilation_period if stride == 1 else assimilation_period + forecast_span
    return centers, feedback_frame


def next_window_start(*, center: int, assimilation_period: int,
                      forecast_span: int, stride: int) -> int:
    # Forecast anchors always begin one physical interval before the next center.
    return center if stride == 1 else center + forecast_span


def analysis_indices(*, num_times: int, assimilation_period: int) -> list[int]:
    if num_times <= 0:
        raise ValueError("num_times must be positive")
    if assimilation_period <= 0:
        raise ValueError("assimilation_period must be positive")
    return list(range(assimilation_period, num_times, assimilation_period))
