# CLI Commands

## `clx manage`

This command is a wrapper around Django's `manage.py` command. It ensures that Django is initialized before running the command.

For example, to run the development server:

```bash
clx manage runserver
```

## `clx generate-docket-sample`

This command generates a sample of dockets for use in the docket viewer application. Running this command will execute the following steps:

1. Download the raw bulk docket data from CourtListener's [bulk data archive](https://com-courtlistener-storage.s3-us-west-2.amazonaws.com/list.html?prefix=bulk-data/). The url for this is hardcoded to the bulk data exported on `2025-10-28`. To use an updated version, set the `BULK_DOCKETS_URL` environment variable.
2. Generates `recap_dockets_reduced.csv`, a preprocessed version of the bulk data. This version:
    - Merges bulk data with document coverage data, showing the number of RECAP documents (i.e. docket entries / attachments) for each case, plus the number of pdfs available for each.
    - Filters to cases with at least one docket entry.
    - Adds a `crude_case_type` column, which attempts to assign a two letter case type based on the `docket_number`.
    - Scopes columns to those used for sampling (`id`, `date_filed`, `court_id`, `nature_of_suit`, etc.)
3. Generates an index of `docket_id`s to use for the sample. We use a variety of sampling strategies to ensure the sample is representative, diverse, and includes edge cases. The sample is saved to `docket_index.csv`. Sampling approaches include:
    - Short cases by court and filing year: Cases with fewer than 20 docket entries.
    - Medium cases by court and filing year: Cases with 20-500 docket entries.
    - Long cases by filing year only: Cases with more than 500 docket entries.
    - Cases where at least 90% of the main documents are available by court and filing year.
    - Crude case type: Cases with a two letter case type.
    - Nature of suit: Cases with a nature of suit (only applicable to civil cases)
