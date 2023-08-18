import torch
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD

# ID processing
ids =  open("interpro_test_term_ids.txt", "r")
lines = ids.read().split()
test_id_dict = {}
reverse_test_id_dict = {}
for i in range(0,len(lines),2):
    # print(lines[i])
    # print(lines[i+1])
    test_id_dict[i/2] = lines[i]
    reverse_test_id_dict[lines[i]] = [i/2]
ids.close()

ids =  open("interpro_term_ids.txt", "r")
lines = ids.read().split()
train_id_dict = {}
reverse_train_id_dict = {}
for i in range(0,len(lines),2):
    # print(lines[i])
    # print(lines[i+1])
    train_id_dict[i/2] = lines[i]
    reverse_train_id_dict[lines[i]] = [i/2]
ids.close()

# train Term Processing 
terms = open("interpro_test_terms.txt", "r")
lines = terms.read().split('\n')
test_term_dict = {}
test_num_categories = 1
saved = ""
for i in range(len(lines)):
    line_array = lines[i].split()
    test_term_dict[line_array[0]] = list(map(int, line_array[1:]))
    # if num_categories < max(term_dict[line_array[0]]):
    #     saved = line_array
    test_num_categories = max(test_num_categories, max(test_term_dict[line_array[0]]))
    
#     if i > 10:
#         break
# print(term_dict)
test_num_categories = test_num_categories+1
print("num test term categories:")
print(test_num_categories)


# Train Term Processing
terms = open("interpro_terms.txt", "r")
lines = terms.read().split('\n')
train_term_dict = {}
train_num_categories = 1
saved = ""
for i in range(len(lines)):
    line_array = lines[i].split()
    train_term_dict[line_array[0]] = list(map(int, line_array[1:]))
    # if num_categories < max(term_dict[line_array[0]]):
    #     saved = line_array
    train_num_categories = max(train_num_categories, max(train_term_dict[line_array[0]]))

train_num_categories = train_num_categories+1
print("num train term categories:")
print(train_num_categories)

# test_id_dict
# keys: numbers (int)
# values: interpro class IDs (string) 

# test_term_dict
# keys: Uniprot ids (string)
# Values : list of Ids (list of int)

# 1. Let train_id_dict be the main dict
main_id_dict = train_id_dict
# 2. For id in test_id_dict, create a dict that maps from the old value to the new:
# e.g. old id for a class is 3 -> 27 bc 27 is the id for that class in train_id_dict
old_test_to_new_test = {}
count = 0
for key in test_id_dict.keys():
    cur_ID = test_id_dict[key]  
    try:
        # new number
        new_num = reverse_train_id_dict[cur_ID]
        old_test_to_new_test[key] = int(new_num[0])
        count = count+1
    except:
        new_num = len(train_id_dict)
        reverse_train_id_dict[cur_ID] = new_num
        train_id_dict[new_num] = cur_ID
        old_test_to_new_test[key] = new_num
print("number of duplicate interpro terms")
print(count)
num_categories = len(train_id_dict)+1
print(num_categories)
# If ID is not contained within dict, create new and assign it the last number
# 3. Reassign the ids numbers in test_term_dict
new_test_term_dict = {}
for key in test_term_dict.keys():
    cur_list = test_term_dict[key]
    new_list = []
    for num in cur_list:
        new_list.append(old_test_to_new_test[num])
    new_test_term_dict[key] = new_list
print("testing:")
print(cur_list)
print(new_list)
print(old_test_to_new_test[cur_list[0]])
test_term_dict = new_test_term_dict

# Encode test term dicts
print("dims of test tensor:")
print(len(test_term_dict))
print(num_categories)
test_multi_hot_labels = torch.zeros((len(test_term_dict),num_categories),dtype=torch.float32)
count = 0
for prot in test_term_dict:
    # temp = torch.zeros(num_categories)
    int_terms = test_term_dict[prot]
    # temp[int_terms] = 1
    test_multi_hot_labels[count][int_terms] = 1
    # multi_hot_labels = torch.cat((multi_hot_labels,temp),1)
    count += 1
print(test_multi_hot_labels.shape)
print(torch.sum(test_multi_hot_labels))

# Encode train term dicts
print("dims of train tensor:")

print(len(train_term_dict))
print(num_categories)
train_multi_hot_labels = torch.zeros((len(train_term_dict),num_categories),dtype=torch.float32)
count = 0
for prot in train_term_dict:
    # temp = torch.zeros(num_categories)
    int_terms = train_term_dict[prot]
    # temp[int_terms] = 1
    train_multi_hot_labels[count][int_terms] = 1
    # multi_hot_labels = torch.cat((multi_hot_labels,temp),1)
    count += 1
print(train_multi_hot_labels.shape)
print(torch.sum(train_multi_hot_labels))


combined_ten = torch.cat((test_multi_hot_labels,train_multi_hot_labels),dim=0)
print("combined tensor shape:")
print(combined_ten.shape)

# model = NMF(n_components=1024, init= 'random', random_state=0)
# W = model.fit_transform(multi_hot_labels)
svd = TruncatedSVD(n_components=1024)
svd.fit(combined_ten)

test_trans_Interpro = svd.transform(test_multi_hot_labels)
np.save("test_embeds.npy",test_trans_Interpro)
test_ids = [*test_term_dict]
np.save("test_ids.npy",test_ids)

train_trans_Interpro = svd.transform(train_multi_hot_labels)
np.save("train_embeds.npy",train_trans_Interpro)
train_ids = [*train_term_dict]
np.save("train_ids.npy", train_ids)