import pandas as pd
import numpy as np
import librosa

def extract_features(file):
	try:
		y, sr = librosa.load(file, res_type = 'kaiser_fast')
		mfcc = librosa.feature.mfcc(y = y, sr = sr, n_mfcc = 40)
		scaled_mfcc = np.mean(mfcc.T, axis = 0)
		return scaled_mfcc
	except FileNotFoundError:
		print('file ' + file + ' does not exist.')
		FailedCount += 1
		return 'ERROR'

metadata = pd.read_csv('metadata.csv')
count = 0
FailedCount = 0
tagList = []
featList = []

for index, row in metadata.iterrows():
	wavPath = 'wav/' + row['itemid'] + '.wav'
	feats = extract_features(wavPath)
	tag = row['hasbird']
	
	if feats != 'ERROR':
		tagList.append(tag)
		featList.append(feats)

	count += 1
	print(count)


print(f"Failed on {FailedCount} files")

feats_array = np.asarray(featList)
tags_array = np.asarray(tagList)

np.save('feats array.npy', feats_array)
np.save('tags array.npy', tags_array)