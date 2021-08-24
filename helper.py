
"""### Helper to check generator dimensions"""

# check generator dimensions
for i in range(len(train_generator)):
    x, y = train_generator[i]
    print(x.shape, y.shape)

# check generator dimensions
for i in range(len(test_generator)):
    x, y = test_generator[i]
    print(x.shape, y.shape)

#test_generator[0]

# check generator dimensions
for i in range(len(val_generator)):
    x, y = val_generator[i]
    print(x.shape, y.shape)