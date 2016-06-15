import test_toolkit as tt
import bagging as bg


data_matrix = tt.load_data('test_data/housingdata.txt')
feature_list = ['age', 'have job', 'have house', 'credit situation']

print('--------------Bagging Test-------------------')
model_list = bg.bagging(data_matrix, feature_list)
prediction = bg.bagging_classify([2, 1, 0, 2], feature_list, model_list)
for i in range(len(model_list)):
    print(model_list[i])
print('bagging prediction:', prediction)

print('--------------Random Forest Test-------------------')
rf_model_list = bg.bagging(data_matrix, feature_list, 1)
rf_prediction = bg.bagging_classify([2, 0, 0, 1], feature_list, rf_model_list)
for i in range(len(rf_model_list)):
    print(rf_model_list[i])
print('random forest prediction:', rf_prediction)