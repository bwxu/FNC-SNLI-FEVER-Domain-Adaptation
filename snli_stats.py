import var
from util import get_snli_data

# Extracts statistics about the SNLI dataset

snli_s1_train, snli_s2_train, snli_labels_train = get_snli_data(var.SNLI_TRAIN)
snli_s1_val, snli_s2_val, snli_labels_val = get_snli_data(var.SNLI_VAL)
snli_s1_test, snli_s2_test, snli_labels_test = get_snli_data(var.SNLI_TEST)

print(len(snli_labels_train))

labels = [0, 0, 0, 0]
for label in snli_labels_train:
    labels[label] += 1

print(labels)

avg_len = 0.0
for headline in snli_s1_train:
    avg_len += len(headline.split())

print(avg_len/len(snli_labels_train))

avg_body_len = 0.0
for body in snli_s2_train:
    avg_body_len += len(body.split())
    
print(avg_body_len/len(snli_labels_train))


print(len(snli_labels_val))

labels = [0, 0, 0, 0]
for label in snli_labels_val:
    labels[label] += 1

print(labels)

avg_len = 0.0
for headline in snli_s1_val:
    avg_len += len(headline.split())

print(avg_len/len(snli_labels_val))

avg_body_len = 0.0
for body in snli_s2_val:
    avg_body_len += len(body.split())
    
print(avg_body_len/len(snli_labels_val))


print(len(snli_labels_test))

labels = [0, 0, 0, 0]
for label in snli_labels_test:
    labels[label] += 1

print(labels)

avg_len = 0.0
for headline in snli_s1_test:
    avg_len += len(headline.split())

print(avg_len/len(snli_labels_test))

avg_body_len = 0.0
for body in snli_s2_test:
    avg_body_len += len(body.split())
    
print(avg_body_len/len(snli_labels_test))

