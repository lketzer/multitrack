import os
import pandas as pd
import requests
from io import BytesIO
from io import StringIO
from astropy.table import Table
from astropy.table import join
from astropy.io import ascii
import astropy.constants as c

def get_catalog(name, basepath="data"):
    """ Function to get NASA Exoplanet Archive catalogs.
        From Dan Foreman-Mackey (https://github.com/dfm/exopop)"""
    fn = os.path.join(basepath, "{0}.h5".format(name))
    if os.path.exists(fn):
        return pd.read_hdf(fn, name)
    if not os.path.exists(basepath):
        os.makedirs(basepath)
    print("Downloading {0}...".format(name))
    url = ("http://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/"
           "nph-nstedAPI?table={0}&select=*").format(name)
    r = requests.get(url)
    if r.status_code != requests.codes.ok:
        r.raise_for_status()
        
    fh = BytesIO(r.content)
    df = pd.read_csv(fh)
    df.to_hdf(fn, name, format="t")
    return df