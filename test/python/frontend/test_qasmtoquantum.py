# RUN: %PYTHON %s | FileCheck %s

from mlir_quantum.dialects import arith
from mlir_quantum.dialects.builtin import Context, IndexType, InsertionPoint, Location

from frontend.qasm.qasmtoquantum import Interval


def run(f):
    print("\nTEST:", f.__name__)
    f()
    return f


def mock_value(v: int = 0, context: Context | None = None, loc: Location | None = None, ip: InsertionPoint | None = None):
    return arith.ConstantOp(
        IndexType.get(context),
        v,
        loc=loc,
    ).result


# CHECK-LABEL: TEST: test_interval_contains
@run
def test_interval_contains():
    with Context() as ctx, Location.unknown() as loc:
        interval = Interval(5, 10, mock_value())

        assert interval.contains(5)  # Lower bound inclusive
        assert interval.contains(9)  # Upper bound exclusive
        assert not interval.contains(10)  # Upper bound exclusive
        assert not interval.contains(4)  # Below lower bound

# CHECK-LABEL: TEST: test_interval_length
@run
def test_interval_length():
    interval = Interval(5, 10, mock_value())
    assert(len(interval) == 5)  # Length is end - start

@run
def test_interval_equality():
    interval1 = Interval(5, 10, mock_value(1))
    interval2 = Interval(5, 10, mock_value(2))  # Different value, same bounds
    assert(interval1 == interval2)  # Equality based on start and end


# class TestIntervalMap(unittest.TestCase):
#     def setUp(self):
#         self.map = IntervalMap()

#     def test_add_interval(self):
#         self.map.add(0, 5, "value1")
#         self.assertEqual(len(self.map), 1)
#         self.assertEqual(self.map.get(0), "value1")
#         self.assertEqual(self.map.get(4), "value1")
#         self.assertIsNone(self.map.get(5))  # Outside the interval

#     def test_add_overlapping_intervals(self):
#         self.map.add(0, 5, "value1")
#         with self.assertRaises(ValueError):
#             self.map.add(4, 10, "value2")  # Overlaps with existing interval
#         with self.assertRaises(ValueError):
#             self.map.add(5, 10, "value3")  # Touching the end of the first interval

#     def test_remove_interval(self):
#         self.map.add(0, 5, "value1")
#         self.map.add(10, 15, "value2")
#         self.assertEqual(self.map.remove(0, 5), "value1")
#         self.assertEqual(len(self.map), 1)
#         with self.assertRaises(KeyError):
#             self.map.remove(0, 5)  # Interval already removed

#     def test_get_interval(self):
#         self.map.add(0, 5, "value1")
#         self.map.add(10, 15, "value2")
#         self.assertEqual(self.map.get(0), "value1")
#         self.assertEqual(self.map.get(4), "value1")
#         self.assertIsNone(self.map.get(5))  # Outside any interval
#         self.assertEqual(self.map.get(10), "value2")

#     def test_interval_containing(self):
#         self.map.add(0, 5, "value1")
#         self.map.add(10, 15, "value2")
#         interval = self.map.interval_containing(3)
#         self.assertEqual(interval.start, 0)
#         self.assertEqual(interval.end, 5)
#         self.assertEqual(interval.value, "value1")
#         with self.assertRaises(KeyError):
#             self.map.interval_containing(6)  # No interval contains this key

#     def test_replace_interval(self):
#         self.map.add(0, 5, "value1")
#         old_interval = self.map.interval_containing(3)
#         new_intervals = [
#             Interval(0, 2, "new_value1"),
#             Interval(3, 5, "new_value2"),
#         ]
#         self.map.replace_interval(old_interval, new_intervals)
#         self.assertEqual(len(self.map), 2)
#         self.assertEqual(self.map.get(1), "new_value1")
#         self.assertEqual(self.map.get(4), "new_value2")
#         self.assertIsNone(self.map.get(5))  # Outside any interval

#     def test_repr(self):
#         self.map.add(0, 5, "value1")
#         self.map.add(10, 15, "value2")
#         self.assertEqual(
#             repr(self.map),
#             "IntervalMap([Interval(start=0, end=5, value='value1'), Interval(start=10, end=15, value='value2')])"
#         )
