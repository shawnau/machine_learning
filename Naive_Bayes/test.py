import naive_bayes as nb

text_list, text_class = nb.create_dataset()
vocab_list = nb.create_vocab_dic(text_list)
input_text = ['stupid', 'garbage']

predict = nb.naive_bayes(text_list, text_class, input_text)
print(predict)