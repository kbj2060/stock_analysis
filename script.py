import os

x1 = sorted(os.listdir('./stock'))
for x in x1:
    fs = os.listdir('./stock/{0}/'.format(x))
    print([os.rename('./stock/{0}/{1}'.format(x, f), './stock/{0}/{1}'.format(x, f[:-3])) for f in fs if str(f)[-2:] == 'h5'])
