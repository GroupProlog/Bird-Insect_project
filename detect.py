import pickle
import librosa
from keras.models import load_model
import numpy as np

loaded_model = load_model('model 0.4.h5')

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

def detectbird(FILE):
	feat = np.asarray(extract_features(FILENAME))
	feat = feat.reshape((1, 40))
	preds = loaded_model.predict(feat)
	pred = [round(i) for x in preds for i in x]
	if pred == [1]:
		return True
	else:
		return False
