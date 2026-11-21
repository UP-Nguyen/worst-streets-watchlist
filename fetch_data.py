def fetch_socrata(
    dataset_id,
    where=None,
    select=None,
    order=None,
    page_size=50000
):
    """
    Pull a Socrata dataset with pagination.
    Returns a pandas DataFrame.
    """

    url = f"{NYC_DOMAIN}/resource/{dataset_id}.json"

    all_rows = []
    offset = 0

    while True:
        params = {
            "$limit": page_size,
            "$offset": offset,
        }

        if where:
            params["$where"] = where

        if select:
            params["$select"] = select

        if order:
            params["$order"] = order

        response = requests.get(
            url,
            params=params,
            headers=HEADERS,
            timeout=60
        )
        response.raise_for_status()

        rows = response.json()

        if not rows:
            break

        all_rows.extend(rows)

        print(f"Fetched {len(all_rows):,} rows")

        if len(rows) < page_size:
            break

        offset += page_size

    return pd.DataFrame(all_rows)