import var
from util import get_fever_data

# Extracts statistics about the FEVER dataset

fever_headlines, fever_bodies, fever_labels, fever_claim_set = get_fever_data(var.FEVER_TRAIN, var.FEVER_WIKI)
fever_domains = [2 for _ in range(len(fever_headlines))]

print(len(fever_labels))

labels = [0, 0, 0, 0]
for label in fever_labels:
    labels[label] += 1

print(labels)

avg_len = 0.0
for headline in fever_headlines:
    avg_len += len(headline.split())

print(avg_len/len(fever_headlines))

avg_body_len = 0.0
for body in fever_bodies:
    avg_body_len += len(body.split())
    
print(avg_body_len/len(fever_bodies))

