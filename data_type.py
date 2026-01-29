from typing import Tuple
from typing import Dict, Optional, Iterable
from dataclasses import dataclass

# Constants
DAYS_5 = ["MON", "TUE", "WED", "THU", "FRI"]
ALL_DAYS_7 = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
Cell = Tuple[int, str]  # (week, day)

DAY_MAP_MIN = {
    "MONDAY": "MON", "TUESDAY": "TUE", "WEDNESDAY": "WED",
    "THURSDAY": "THU", "FRIDAY": "FRI",
    "SATURDAY": "SAT", "SUNDAY": "SUN",
}

WEEKS = [1, 2, 3, 4]

# Aggregation key
KEY_COLS = ["custno", "volume", "material_typ"]

# Types
WeekTuple = Tuple[int, ...]
DayBits = Tuple[int, int, int, int, int, int, int]
SchedTuple = Tuple[WeekTuple, DayBits]


@dataclass(frozen=True)
class ScheduleView:
    week_tuple: WeekTuple
    day_bits: DayBits

    def week_key(self) -> WeekTuple:
        return self.week_tuple

    def day_key(self) -> DayBits:
        return self.day_bits


@dataclass
class Stop:
    custno: int
    xcoord: float
    ycoord: float
    qty: Optional[float]
    volume: float
    weight: float
    sos: Optional[int]
    dowcd: str
    dowlockcd: Optional[int]
    wccd_flag: Optional[int]
    material_typ: Optional[str]
    clusterid: Optional[int]
    frequency: int

    baseline: ScheduleView
    chosen: Optional[ScheduleView] = None
    changed: Optional[int] = None

# 같은 튜플을 하나의 객체로 공유하기 위한 장치 -> 모든 stop마다 week, day pattern을 위한 객체를 만드는 것을 방지하기 위해
class TuplePools:
    def __init__(self):
        self._week_pool: Dict[WeekTuple, WeekTuple] = {}
        self._day_pool: Dict[DayBits, DayBits] = {}
        self._sched_pool: Dict[SchedTuple, SchedTuple] = {}

    def week(self, weeks: Iterable[int]) -> WeekTuple:
        t = tuple(sorted(tuple(weeks)))
        self._week_pool.setdefault(t, t)
        # 해당 week pattern을 본 적 있으면 불러오고, 없다면 t 값을 등록한 뒤에 가져와라
        return self._week_pool[t]

    def day(self, bits: DayBits) -> DayBits:
        self._day_pool.setdefault(bits, bits)
        return self._day_pool[bits]

    def schedule(self, week_tuple: WeekTuple, day_bits: DayBits) -> SchedTuple:
        st = (week_tuple, day_bits)
        self._sched_pool.setdefault(st, st)
        return self._sched_pool[st]