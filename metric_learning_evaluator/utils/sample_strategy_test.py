from sample_strategy import SampleStrategy


instance_ids = [1,2,3,4,5]
label_ids = [1,1,2,2,1]
print (instance_ids, label_ids)
sampler = SampleStrategy(instance_ids, label_ids)

print(sampler._sample('A', 'A', 2, 2, 2))