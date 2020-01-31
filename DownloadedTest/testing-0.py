from gensim.models import Word2Vec
import gensim.downloader as api
testing_list = [['fund','defund'], ['short', 'long'], ['bond', 'stock'], ['capitalize', 'subsidize'], ['capitalize', 'fund'], ['hedge', 'security'], ['securities', 'commodities'],['bank','savings'],['company', 'organization'],['money', 'stock'], ['tax','irs'], ['charity','nonprofit'], ['llc', '501(c)']]
#testing_list = []
new_model = Word2Vec.load('core-word2vec.model')
#Antonyms
for pair in testing_list:
    result = new_model.similarity(pair[0],pair[1])
    print(result)
print('end')

pretrained_model = api.load("word2vec-google-news-300")
for pair in testing_list:
    result2 = pretrained_model.similarity(pair[0],pair[1])
    print(result2)