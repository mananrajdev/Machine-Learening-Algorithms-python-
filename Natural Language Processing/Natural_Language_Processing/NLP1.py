#Natural Language processing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv("Restaurant_Reviews.tsv", delimiter='\t', quoting=3)

#cleaning the text
import re
import nltk

review=re.sub('[^a-zA-Z]',' ',dataset['Review'][0])
review=review.lower()

nltk.download('stopwords')
from nltk.corpus import stopwords
review=review.split()

from nltk.stem.porter import PorterStemmer
ps= PorterStemmer()
review= [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]

