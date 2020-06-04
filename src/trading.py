from system_trading import SystemTrading
import os
import json

os.chdir(os.getcwd()+'/../')
SUBJECT = sorted(os.listdir('./stock/'))
cost = {}

for idx, s in enumerate(SUBJECT[51:100]):
    st = SystemTrading(s)
    print("{0} / {1} {2} is executing..".format(idx, len(SUBJECT), s))
    cost = st.analysis()
    st.predict()

    with open('costs.json', 'a+', encoding='UTF-8') as costJson:
        json_str = json.dumps([{'name': k, 'cost': v} for k, v in cost.items()], indent=4, ensure_ascii=False)
        costJson.write(json_str)
        costJson.close()
   # except Exception as e:
    #    print('ERROR')
     #   error = '{0} Error. {1} \n'.format(s, e)
      #  with open('errors.log', 'a+') as f:
       #     f.write(error)
        #    f.close()
