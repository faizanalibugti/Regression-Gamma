import numpy as np
import pandas as pd
#from collections import Counter
#from random import shuffle

train_data = np.load('training_data.npy')
df = pd.DataFrame(train_data, columns = ['Yaverage', 'Correct Parameter'])
print(df)


# for data in train_data:
#     Yaverage = data[0]
#     print("Yaverage: {}".format(Yaverage))
#     correct_param = data[1]
#     print("correct_param: {}".format(correct_param))
    