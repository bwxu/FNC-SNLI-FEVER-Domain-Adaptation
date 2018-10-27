import var
from util import get_fnc_data

fnc_headlines_train, fnc_bodies_train, fnc_labels_train, fnc_body_ids_train = get_fnc_data(var.FNC_TRAIN_STANCES, var.FNC_TRAIN_BODIES)
fnc_headlines_test, fnc_bodies_test, fnc_labels_test, _ = get_fnc_data(var.FNC_TEST_STANCES, var.FNC_TEST_BODIES)

print(len(fnc_labels_train))

labels = [0, 0, 0, 0]
for label in fnc_labels_train:
    labels[label] += 1

print(labels)

avg_len = 0.0
for headline in fnc_headlines_train:
    avg_len += len(headline.split())

print(avg_len/len(fnc_headlines_train))

avg_body_len = 0.0
for body in fnc_bodies_train:
    avg_body_len += len(body.split())
    
print(avg_body_len/len(fnc_bodies_train))
