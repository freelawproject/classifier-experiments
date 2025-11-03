# Local Data

This directory includes local files used by this library. You can configure the location of this directory by setting the `LOCAL_DATA_DIR` environment variable. To access a pathlib object use `from clx.settings import LOCAL_DATA_DIR`.

## Docket Sample

The `docket_sample` directory includes the docket sample used for the docket viewer application. You can run `clx generate-docket-sample` to reproduce the steps for creating the sample. The following files are created:

- `recap_dockets.csv.bz2`: The raw bulk docket data from CourtListener's [bulk data archive](https://com-courtlistener-storage.s3-us-west-2.amazonaws.com/list.html?prefix=bulk-data/).
- `recap_dockets.csv`: The unzipped bulk docket data.
- `document_coverage.csv`: The document coverage data exported from CourtListener's database via [#5107](https://github.com/freelawproject/courtlistener/pull/5107).
- `recap_dockets_reduced.csv`: A reduced version of the bulk data, filtered to relevant columns and cases with at least one docket entry.
- `docket_index.csv`: The `docket_id` index for the sample.

