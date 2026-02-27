# RUN: %PYTHON %s | FileCheck %s


from mlir_quantum.dialects import arith
from mlir_quantum.dialects.builtin import Context, IndexType, InsertionPoint, Location

from frontend.qasm.qasmtoquantum import Interval, IntervalMap


def run(f):
    print("\nTEST:", f.__name__)
    f()
    return f


def mock_value(v: int = 0, *, context: Context | None = None, loc: Location | None = None, ip: InsertionPoint | None = None):
    return arith.ConstantOp(
        IndexType.get(context),
        v,
        loc=loc,
    ).result


# CHECK-LABEL: TEST: test_interval_contains
@run
def test_interval_contains():
    with Context(), Location.unknown():
        interval = Interval(5, 10, mock_value())

        # Lower bound inclusive
        assert interval.contains(5)
        # Upper bound exclusive
        assert interval.contains(9)
        # Upper bound exclusive
        assert not interval.contains(10)
        # Below lower bound
        assert not interval.contains(4)


# CHECK-LABEL: TEST: test_interval_length
@run
def test_interval_length():
    with Context(), Location.unknown():
        interval = Interval(5, 10, mock_value())
        # Length is end - start
        assert len(interval) == 5


# CHECK-LABEL: TEST: test_interval_equality
@run
def test_interval_equality():
    with Context(), Location.unknown():
        interval1 = Interval(5, 10, mock_value(1))
        # Different value, same bounds
        interval2 = Interval(5, 10, mock_value(2))
        # Equality based on start and end
        assert interval1 == interval2


# CHECK-LABEL: TEST: test_add_interval
@run
def test_add_interval():
    with Context(), Location.unknown():
        map = IntervalMap()
        mock = mock_value()
        map.add(0, 5, mock)
        assert len(map) == 1
        assert map.get(0) == mock
        assert map.get(4) == mock
        # Outside the interval
        assert map.get(5) is None


# CHECK-LABEL: TEST: test_add_interval_overlap
@run
def test_add_interval_overlap():
    with Context(), Location.unknown():
        map = IntervalMap()
        map.add(0, 5, mock_value())
        try:
            # Overlaps with existing interval
            map.add(4, 10, mock_value())
        except ValueError:
            pass
        else:
            raise AssertionError("Expected ValueError due to overlapping interval")


# CHECK-LABEL: TEST: test_add_interval_overlap_upper_lower
@run
def test_add_interval_overlap_upper_lower():
    with Context(), Location.unknown():
        map = IntervalMap()
        map.add(0, 5, mock_value())
        try:
            # Touching the end of the first interval
            map.add(5, 10, mock_value())
        except Exception as e:
            raise AssertionError("Unexpected exception") from e


# CHECK-LABEL: TEST: test_remove_interval
@run
def test_remove_interval():
    with Context(), Location.unknown():
        map = IntervalMap()
        mock1 = mock_value()
        map.add(0, 5, mock1)
        map.add(10, 15, mock_value())
        assert len(map) == 2
        assert map.remove(0, 5) == mock1
        assert len(map) == 1
        try:
            # Interval already removed
            map.remove(0, 5)
        except KeyError:
            pass
        else:
            raise AssertionError("Expected KeyError due to no-existing interval")


# CHECK-LABEL: TEST: test_get_interval
@run
def test_get_interval():
    with Context(), Location.unknown():
        map = IntervalMap()
        mock1 = mock_value()
        mock2 = mock_value()
        map.add(0, 5, mock1)
        map.add(10, 15, mock2)
        assert map.get(0) == mock1
        assert map.get(4) == mock1
        # Outside any interval
        assert map.get(5) is None
        assert map.get(10) == mock2


# CHECK-LABEL: TEST: test_interval_containing
@run
def test_interval_containing():
    with Context(), Location.unknown():
        map = IntervalMap()
        mock1 = mock_value()
        mock2 = mock_value()
        map.add(0, 5, mock1)
        map.add(10, 15, mock2)
        interval = map.interval_containing(3)
        assert interval.start == 0
        assert interval.end == 5
        assert interval.value == mock1
        try:
            # No interval contains this key
            map.interval_containing(6)
        except KeyError:
            pass
        else:
            raise AssertionError("Expected KeyError due to non-existing interval")


# CHECK-LABEL: TEST: test_replace_interval
@run
def test_replace_interval():
    with Context(), Location.unknown():
        map = IntervalMap()
        mock1 = mock_value()
        mock2 = mock_value()
        mock3 = mock_value()
        map.add(0, 5, mock1)
        old_interval = map.interval_containing(3)
        new_intervals = [
            Interval(0, 2, mock2),
            Interval(2, 5, mock3),
        ]
        map.replace_interval(old_interval, new_intervals)
        assert len(map) == 2
        assert map.get(1) == mock2
        assert map.get(4) == mock3
        # Outside any interval
        assert map.get(5) is None
