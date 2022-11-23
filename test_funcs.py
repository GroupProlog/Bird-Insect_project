import slist, cleaner, data

def test_slist():
	splist = slist.createSlist('xcmeta.csv')
	assert splist[0].split('_')[1] == 'Common Redpoll'

def test_cleaner():
	feats, tags = cleaner.processFiles()
	assert feats.shape[1] == 40

def test_cluster():
	feats, tags = data.cluster()
	assert tags.shape[0] == 2059