from src.system_trading import SystemTrading
import os
import json
from src.utils import report_error

os.chdir(os.getcwd()+'/../')
SUBJECT = sorted(os.listdir('./stock/'))
cost = {}
start = 0

for idx, s in enumerate(SUBJECT):
    try:
        st = SystemTrading(s)
        print("{0} / {1} {2} is executing..".format(idx, len(SUBJECT), s))
        cost = st.analysis()
        st.predict()
    except Exception as e:
        print('ERROR')
        error = '{0} Error. {1} \n'.format(s, e)
        report_error(error, 'errors.log')
    with open('costs.json', 'a+', encoding='UTF-8') as costJson:
        json_str = json.dumps([{'name': k, 'cost': v} for k, v in cost.items()], indent=4, ensure_ascii=False)
        costJson.write(json_str)
        costJson.close()
