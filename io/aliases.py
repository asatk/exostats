import multiprocessing as mp
import pandas as pd
import re
import requests

# number of worker threads for multiprocessing thread pool
num_threads = 10

# location of alias lookup resource
url = "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/Lookup/nph-aliaslookup.py"

# tags (prefixes) of aliases to identify each object by
tag_list = ["KOI", "KIC", "EPIC", "TOI", "TIC"]


def fix_ID(tag: str):
    ...


def get_aliases(name: str, i: int, tag_list: list[str]=tag_list):
    """
    Retrieve all aliases for the object with the requested name `name`. The
    returned aliases may be customized through `tag_list`. Default aliases are:
        KOI, KIC, EPIC, TOI, TIC
    """

    # make request to alias lookup resource
    params = dict(objname=name)
    r = requests.get(url, params)
    req = r.json()

    # failed request
    if r.status_code != 200:
        print(f"[{i}] Request to NEA alias lookup encountered an issue: ({r.status_code})", flush=True)
        return

    # failed lookup but successful HTTP request
    if req["manifest"]["lookup_status"] != "OK":
        print(f"[{i}] Lookup of object '{name}' unsuccessful", flush=True)
        return

    # get PSCP default name
    name_default = req["manifest"]["resolved_name"]
    aliases = req["system"]["objects"]["stellar_set"]["stars"][name_default]["alias_set"]["aliases"]

    # progress print
    print(f"[{i}] Request processed for '{name}' = ('{name_default}')", flush=True)

    # make a new row
    data = {t: None for t in tag_list}
    data["hostname"] = name_default

    # pattern that matches desired Kepler or TESS catalog names
    tag_list_str = "|".join(tag_list)
    p = re.compile(rf"^({tag_list_str})(-|\s+)?(\d+)")

    # identify KOI, KIC, EPIC, TOI, or TIC identifiers/aliases
    for al in aliases:

        # match alias name to catalog pattern
        m = p.match(al)

        # alias does not match catalog pattern
        if m is None:
            continue

        # alias prefix
        pfx = m[1]

        # alias ID (just the number)
        num = m[3]

        # fill host's row with ID number
        for tag in tag_list:
            if pfx == tag:
                data[tag] = int(num)
                break

    df = pd.Series(data).to_frame().T
    df = df[["hostname", *tag_list]]

    return df



def iter_aliases(hostnames: pd.Series, tag_list: list[str]=tag_list):
    """
    Fetch all aliases of specified objects iteratively. The aliases may belong
    to the following catalogs:
        KOI, KIC, EPIC, TOI, TIC
    """

    row_list = []
    for i, name in enumerate(hostnames):
        row = get_aliases(name, i, tag_list)
        row_list.append(row)

    df = pd.concat(row_list, ignore_index=True)
    #df = pd.concat(row_list)
    #df = df.reset_index()

    return df



def pool_aliases(hostnames: pd.Series, tag_list: list[str]=tag_list):
    """
    Fetch all aliases of specified objects in parallel. The aliases may belong
    to the following catalogs:
        KOI, KIC, EPIC, TOI, TIC
    """

    n = len(hostnames)
    map_args = zip(hostnames, range(n), [tag_list] * n)

    # get alias for all unique host names in PSCP
    with mp.Pool(num_threads) as p:
        row_list = p.starmap(get_aliases, map_args)
    
    #df = pd.concat(row_list)
    df = pd.concat(row_list, ignore_index=True)
    #df = df.reset_index()

    return df

