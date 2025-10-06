import pyvo as vo

url_NEA = "https://exoplanetarchive.ipac.caltech.edu/TAP"



if __name__ == "__main__":
    # create object that will service TAP queries to NEA TAP server
    service = vo.dal.TAPService(url_NEA)

    cols = "*"
    table = "pscomppars"

    query = f"SELECT TOP 10 {cols} FROM {table}"

    res = service.search(query)

    print(res)
