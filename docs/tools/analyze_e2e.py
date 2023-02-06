import click
import pandas as pd

from eyepy.io import HeE2eReader


def main(path):
    folder_headers = []

    for chunk in HeE2eReader(path).parsed_file.chunks:
        for folder in chunk.folders:
            folder_headers.append(folder.header)

    # Get all folder headers but only the second slice if the data is assigned to a slice
    folder_meta = [(f.patient_id, f.study_id, f.series_id, f.slice_id,
                    int(f.type), f.size) for f in folder_headers
                   if ((0 > f.slice_id) or (f.slice_id > 1000)) or (f.slice_id == 2)]

    data = pd.DataFrame.from_records(folder_meta,
                                     columns=[
                                         "patient_id", "study_id", "series_id",
                                         "slice_id", "type", "size"
                                     ])

    gr = data.groupby(
        ["patient_id", "study_id", "series_id", "slice_id", "type",
         "size"]).count()


if __name__ == "__main__":
    # get the path to the file to be analyzed
    #path = click.prompt("Enter path to file to be analyzed")
    path = "/home/morelle/Data/MACUSTAR-313-001-0001-V1-SDOCT_cSLO.e2e"
    main(path)
