import bayes

listOPosts, listClasses = bayes.loadDataset()
myVocabList = bayes.createVocabList(listOPosts)
word2Vec = bayes.setOfWords2Vec(myVocabList, listOPosts[0])
print(myVocabList, '\n', listOPosts[0])
print(word2Vec)