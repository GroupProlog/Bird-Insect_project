import pandas as pd

def createSlist(file):
	st = ''
	
	metadata = pd.read_csv(file, sep='\t', lineterminator='\n')
	
	slist = []

	for index, row in metadata.iterrows():
		mid = row['gen'] + ' ' + row['sp'] + '_' + row['en'] + '\n'
		if mid not in slist:
			st += mid
			slist.append(mid)

	with open('slist.txt', 'w') as file:
		file.write(st)

	return st.split('\n')

#createSlist(metadata)