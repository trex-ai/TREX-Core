# ruff: noqa: N999

from datetime import datetime
from typing import cast

from TREX_Core.utils import utils

type TouConfig = dict[str, dict[str, float]]
type TouSchedule = dict[tuple[int, ...], dict[tuple[int, ...], float]]


class Market:
    """The Grid should emulate the retail price schedule as closely as possible.

    The Grid should always be available as an option to buy energy from
    and sell energy to.
    """

    def __init__(self, price, fee_ratio=None, **kwargs):
        self.id = "grid"
        self.__default_price = price
        self.tou: TouSchedule | None = None
        self.source_info = {
            # load pays fees
            "grid": {
                "price": price,  # energy price in $/kWh
                "fees": fee_ratio if fee_ratio is not None else 0,  # multiplier
            }
        }

        tou_config = kwargs.get("tou")
        if tou_config is not None:
            self.tou = self.parse_tou(cast(TouConfig, tou_config))

    def parse_tou(self, tou_config: TouConfig) -> TouSchedule:
        """Parse a weekday-only time-of-use schedule.

        Weekends fall back to the default flat rate.
        """
        tou: TouSchedule = {}
        for months, hour_config in tou_config.items():
            months_range = tuple(int(month) for month in months.split(","))
            for hours, price in hour_config.items():
                hours_range = tuple(int(hour) for hour in hours.split(","))
                if months_range not in tou:
                    tou[months_range] = {}
                tou[months_range][hours_range] = price
        return tou

    def get_tou_price(
        self, local_time: datetime, tou_schedule: TouSchedule
    ) -> float | None:
        """Return the configured TOU price for a local time."""
        # weekends are generally exempt from TOU prices
        day_of_week = local_time.isoweekday()
        if day_of_week in (6, 7):
            return None

        month = local_time.month
        hour = local_time.hour
        for months in tou_schedule:
            if month in months:
                for hours in tou_schedule[months]:
                    if hour in hours:
                        return tou_schedule[months][hours]
        return None

    def update_price(self, timestamp, timezone):
        if self.tou is None:
            return
        local_time = utils.timestamp_to_local(timestamp, timezone)
        price = self.get_tou_price(local_time, self.tou)
        self.source_info["grid"]["price"] = (
            price if price is not None else self.__default_price
        )

    def buy_price(self):
        """Unit cost to buy energy from the grid, in $/kWh

        Returns:
            [type]: [description]
        """
        # price to buy from grid
        kwh_price = self.source_info["grid"]["price"] * (
            1 + self.source_info["grid"]["fees"]
        )
        return round(kwh_price, 4)

    def sell_price(self):
        """Unit profit to sell energy to the grid, in $/kWh

        Returns:
            [type]: [description]
        """
        kwh_price = self.source_info["grid"]["price"]
        return round(kwh_price, 4)


if __name__ == "__main__":
    pass
