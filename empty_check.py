import os
import pandas  as pd
import shutil

files_list = os.listdir('stock')
error = []
for i in range(0, len(files_list)):
    try:
        raw_data = pd.read_csv('./stock/{0}/{0}.csv'.format(files_list[i]))
    except pd.io.common.EmptyDataError:
        print(files_list[i], " is empty and has been skipped.")
        shutil.rmtree('./stock/{0}'.format(files_list[i]))
    except:
        error.append(files_list[i])
        continue

print(error)
