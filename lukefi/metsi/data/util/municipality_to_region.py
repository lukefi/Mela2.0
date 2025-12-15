import csv
from pathlib import Path


class MunicipalityToRegionConversion:
    mun_to_reg_map: dict[int, int]

    def __init__(self, mapping_key_csv_path: str | Path) -> None:
        with open(mapping_key_csv_path, encoding="utf-8") as csv_file:
            reader = csv.reader(csv_file, delimiter=";")
            # Skip header row
            next(reader)
            for row in reader:
                source = int(row[0].strip(" \"\'"))
                target = int(row[2].strip(" \"\'"))
                self.mun_to_reg_map[source] = target

    def __getitem__(self, source: int) -> int:
        return self.mun_to_reg_map[source]
