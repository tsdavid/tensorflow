import numpy as np
import pandas as pd
import pandas_datareader.data as web
import datetime

start = datetime.datetime(2009,3,14)
end = datetime.datetime(2018,12,26)
goog = web.get_data_yahoo(['GOOG'],start=start,end=end)

print(goog)
