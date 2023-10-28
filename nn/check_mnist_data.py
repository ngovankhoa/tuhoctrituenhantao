from datasets import load_dataset
mnist = load_dataset('mnist')
print(mnist)
'''
DatasetDict({
    train: Dataset({
        features: ['image', 'label'],
        num_rows: 60000
    })
    test: Dataset({
        features: ['image', 'label'],
        num_rows: 10000
    })
})
'''
print(mnist['train']['label'][0]) #5
img0 = mnist['train']['image'][0]
print(img0.size) #(28, 28)
img0.show()