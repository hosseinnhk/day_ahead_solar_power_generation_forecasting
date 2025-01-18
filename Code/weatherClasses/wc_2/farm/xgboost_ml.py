import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, PredictionErrorDisplay, mean_squared_error, mean_absolute_percentage_error

plt.rc('font', family='Times New Roman')
plt.rcParams.update({'font.size': 14})
plt.rcParams['legend.frameon'] = 'False'
y_test = y_test.reset_index(drop=True)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test)
parameters = {'max_depth': 9,
              'objective': 'reg:squarederror',
              'eval_metric': 'rmse',
              # 'n_estimators': 3600,
              'eta': 0.01280835351496272,
              'subsample': 0.7,
              'colsample_bytree': 0.9,
              'colsample_bylevel': 0.8,
              'min_child_weight': 0.5603877044205587,
              'reg_lambda': 18.08031125352632,
              'reg_alpha': 0.8110321676302101,
              'gamma': 0.005144217103386925,
              }

xg = xgb.train(parameters, dtrain, 2000)
y_train_pred = xg.predict(dtrain)
dtest = xgb.DMatrix(X_test)
y_pred_xgb = xg.predict(dtest)
y_pred_xgb[y_pred_xgb < 0] = 0
y_pred_xgb[y_pred_xgb > y_test.max()] = y_test.max()
print("rmse score with XGBoost = ", mean_squared_error(y_train_pred, y_train, squared=False))
print("rmse score with XGBoost= ", mean_squared_error(y_pred_xgb, y_test, squared=False))
print('mae score with XGBoost= ', np.mean(np.abs(y_pred_xgb - y_test)))
print('r2 score=', r2_score(y_pred_xgb, y_test))

# fig = plt.figure(dpi=532) # figsize=(20, 20),
# display = PredictionErrorDisplay.from_predictions(
#     y_true=y_test,
#     y_pred=y_pred_xgb,
#     kind="actual_vs_predicted",
#     scatter_kwargs={"alpha": 0.2, "color": "tab:blue"},
#     line_kwargs={"color": "tab:red"},
# )
# plt.grid(linewidth=.5)
# plt.xlabel('Predicted values', fontsize=18)
# plt.ylabel('Actual values', fontsize=18)
# plt.title('XGBoost', fontsize=20)
# legend= plt.legend(['R2 score = {:.2f}'.format(r2_score(y_pred_xgb, y_test))], loc='lower right', fontsize=18) # frameon=False
# for handle in legend.legendHandles:
#     handle.set_visible(False)
# legend.get_frame().set_facecolor('None')
#
# plt.tight_layout()
# plt.savefig('xgboost_scatter_pred_truth.svg')
# plt.show()


# plt.figure()
# plt.figure(figsize=(20, 10), dpi=180)
# plt.plot(y_pred_xgb[:50], label='XGBM')  #, 'gd'
# plt.plot(y_test[:50], label='Actual')  #, 'c^'
# plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
# plt.ylabel('predicted')
# plt.xlabel('training samples')
# plt.legend(loc="best")
# plt.title('Regressor predictions and their average')
# plt.show()
