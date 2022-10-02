import pandas as pd
import urllib.request

url = "https://waterservices.usgs.gov/nwis/dv/?format=rdb,1.0&bBox=-72.095076,41.358143,-70.095076,43.358143&startDT=2019-01-01&endDT=2022-01-01&siteStatus=all"
urllib.request.urlretrieve(url, 'testTsv.tsv') # any dir to save

df = pd.read_csv('testTsv.tsv', sep='\t')

print(df.head())