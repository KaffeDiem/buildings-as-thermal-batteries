from datetime import date, datetime, timedelta
from typing import List, Literal
from dataclasses import dataclass
import urllib.request
import urllib.error
import json
import logging
import numpy as np

class FCRService:
    def __init__(self):
        pass

    def get_fcr_prices(self) -> tuple[float, float]:
        url = f"https://api.energidataservice.dk/dataset/FcrNdDK2?limit=100"
        response = urllib.request.urlopen(url)
        data = json.loads(response.read())

        # Filter for DK2 and AuctionType "Total"
        filtered_records = [
            r for r in data['records']
            if r["PriceArea"] == "DK2" and r["AuctionType"] == "Total"
        ]
        
        filtered_records.sort(key=lambda r: r["HourUTC"], reverse=True)
        fcr_d_down_price: float | None = None
        fcr_d_up_price: float | None = None
        
        for r in filtered_records:
            if r["ProductName"] == "FCR-D ned" and fcr_d_down_price is None:
                fcr_d_down_price = r["PriceTotalEUR"]
            elif r["ProductName"] == "FCR-D upp" and fcr_d_up_price is None:
                fcr_d_up_price = r["PriceTotalEUR"]
            
            # Break early if we have both prices from the most recent timestamp
            if fcr_d_down_price is not None and fcr_d_up_price is not None:
                break
        
        if fcr_d_down_price is None or fcr_d_up_price is None:
            print("Could not find FCR-D prices for DK2")
            return 0.0, 0.0
        
        return fcr_d_down_price / 1_000, fcr_d_up_price / 1_000